"""Contrato comum de dados, proveniência e avaliação da rodada corrigida."""
import argparse
import hashlib
import json
from pathlib import Path

CLASSES = ['Parasitized', 'Uninfected']
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 123
VERSAO = 'preprocessamento-corrigido-v1'
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / 'dados' / 'protocolo_v1'
SAIDA = RAIZ / 'codigos' / 'resultados' / 'corrigido_seed123'
EXTENSOES = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, 'rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def ler_manifesto(dados):
    """Confere arquivos, hashes, classes, partições e grupos antes de usar dados."""
    dados = Path(dados)
    caminho = dados / 'manifesto.json'
    manifesto = json.loads(caminho.read_text(encoding='utf-8'))
    if manifesto['classes'] != CLASSES:
        raise ValueError('Classes ou ordem de classes incompatíveis.')
    arquivos, hashes, grupos = set(), set(), {}
    contagens = {(s, c): 0 for s in ('train', 'val', 'test') for c in CLASSES}
    for item in manifesto['imagens']:
        relativo = Path(item['caminho'])
        s, c = item['conjunto'], item['classe']
        if (s, c) not in contagens or relativo.parts[:2] != (s, c):
            raise ValueError('Partição/classe inválida no manifesto.')
        if relativo.is_absolute() or '..' in relativo.parts:
            raise ValueError('Caminho inválido no manifesto.')
        if str(relativo) in arquivos or item['sha256'] in hashes:
            raise ValueError('Arquivo ou conteúdo duplicado no manifesto.')
        grupo = item.get('grupo')
        if grupo is not None:
            if grupo in grupos and grupos[grupo] != s:
                raise ValueError('Grupo compartilhado entre partições.')
            grupos[grupo] = s
        if sha256(dados / relativo) != item['sha256']:
            raise ValueError(f'Imagem alterada: {relativo}')
        arquivos.add(relativo.as_posix())
        hashes.add(item['sha256'])
        contagens[s, c] += 1
    existentes = {p.relative_to(dados).as_posix()
                  for s in ('train', 'val', 'test')
                  for p in (dados / s).rglob('*')
                  if p.is_file() and p.suffix.lower() in EXTENSOES}
    if existentes != arquivos or not all(contagens.values()):
        raise ValueError('Manifesto incompleto, imagens extras ou classe vazia.')
    return manifesto, sha256(caminho)


def carregar_conjunto(dados, conjunto, seed=SEED):
    from tensorflow.keras.utils import image_dataset_from_directory
    return image_dataset_from_directory(
        str(Path(dados) / conjunto), class_names=CLASSES,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=conjunto == 'train', seed=seed,
        interpolation='bilinear',
    )


def argumentos_avaliacao():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dados', type=Path, default=DADOS)
    parser.add_argument('--modelos-dir', type=Path, default=SAIDA)
    parser.add_argument('--conjunto', choices=['val', 'test'], default='test')
    args = parser.parse_args()
    _, args.manifesto_sha256 = ler_manifesto(args.dados)
    args.saida = args.modelos_dir / ('avaliacao_' + args.conjunto)
    args.saida.mkdir(parents=True, exist_ok=True)
    return args


def conferir_modelo(caminho, manifesto_sha256):
    """Recusa pesos antigos ou provenientes de outra divisão."""
    meta = json.loads(Path(caminho).with_suffix('.json').read_text(encoding='utf-8'))
    if (meta['protocolo'] != VERSAO or meta['classes'] != CLASSES
            or meta['manifesto_sha256'] != manifesto_sha256
            or meta['modelo_sha256'] != sha256(caminho)):
        raise ValueError(f'Modelo/proveniência incompatível: {caminho}')
    return meta


def predizer(caminho, conjunto):
    import numpy as np
    import tensorflow as tf
    import preprocessamento  # registra a camada antes de load_model (sem Lambda)
    modelo = tf.keras.models.load_model(caminho, compile=False)
    reais, probabilidades = [], []
    for imagens, labels in conjunto:
        probabilidades.append(modelo(imagens, training=False).numpy())
        reais.append(np.argmax(labels.numpy(), axis=1))
    y, prob = np.concatenate(reais), np.concatenate(probabilidades)
    if (prob.shape != (len(y), len(CLASSES)) or not np.isfinite(prob).all()
            or (prob < 0).any() or not np.allclose(prob.sum(axis=1), 1., atol=1e-5)):
        raise ValueError('Saída inválida do modelo.')
    del modelo
    tf.keras.backend.clear_session()
    return y, np.argmax(prob, axis=1), prob


def salvar_predicoes(args, nome, caminho, y, pred, prob, arquivos):
    import numpy as np
    np.savez_compressed(args.saida / f'predicoes_{nome}.npz',
                        y_true=y, y_pred=pred, probabilidades=prob,
                        arquivos=np.asarray(arquivos), classes=np.asarray(CLASSES))
    meta = conferir_modelo(caminho, args.manifesto_sha256)
    (args.saida / f'proveniencia_{nome}.json').write_text(
        json.dumps(dict(conjunto=args.conjunto, preliminar=args.conjunto == 'val',
                        criterio_decisao='argmax softmax; sem ajuste no teste',
                        treinamento=meta), ensure_ascii=False, indent=2),
        encoding='utf-8')
