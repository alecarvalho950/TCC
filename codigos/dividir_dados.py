"""Cria NOVA divisão aproximada 70/15/15 sem mover a fonte histórica.

A origem deve conter Parasitized/ e Uninfected/. Não reutilize os pesos antigos
com esta divisão. Sem --executar, apenas valida e mostra as contagens.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from protocolo import CLASSES, DADOS, EXTENSOES, SEED, ler_manifesto, sha256


def montar_plano(origem, grupos_csv=None, por_imagem=False, seed=SEED):
    origem = Path(origem)
    if not grupos_csv and not por_imagem:
        raise ValueError('Informe --grupos CSV ou aceite --por-imagem (exploratório).')
    grupos = {}
    if grupos_csv:
        with open(grupos_csv, newline='', encoding='utf-8-sig') as arquivo:
            for linha in csv.DictReader(arquivo):
                nome, grupo = linha['arquivo'], linha['grupo'].strip()
                if not grupo or nome in grupos:
                    raise ValueError('Grupo vazio ou arquivo repetido no CSV.')
                grupos[nome] = grupo
    itens, vistos = [], {}
    for classe in CLASSES:
        for p in sorted((origem / classe).rglob('*')):
            if not p.is_file() or p.suffix.lower() not in EXTENSOES:
                continue
            relativo = p.relative_to(origem).as_posix()
            digest = sha256(p)
            if digest in vistos:
                raise ValueError(f'Conteúdo duplicado: {relativo} e {vistos[digest]}')
            vistos[digest] = relativo
            if grupos_csv and relativo not in grupos:
                raise ValueError(f'Imagem sem grupo: {relativo}')
            itens.append(dict(origem=relativo, classe=classe, sha256=digest,
                              grupo=grupos.get(relativo)))
    if grupos_csv and set(grupos) != {i['origem'] for i in itens}:
        raise ValueError('CSV contém arquivos que não estão na origem.')
    indices = list(range(len(itens)))
    y = [i['classe'] for i in itens]
    if grupos_csv:
        g = [i['grupo'] for i in itens]
        # Grupo inteiro numa só partição; proporções de IMAGENS são aproximadas.
        treino, reserva = next(GroupShuffleSplit(
            n_splits=1, test_size=0.30, random_state=seed).split(indices, y, g))
        val_i, test_i = next(GroupShuffleSplit(
            n_splits=1, test_size=0.50, random_state=seed + 1).split(
                reserva, groups=[g[i] for i in reserva]))
        val, teste = reserva[val_i], reserva[test_i]
    else:
        treino, reserva = train_test_split(indices, test_size=0.30,
                                          stratify=y, random_state=seed)
        val, teste = train_test_split(reserva, test_size=0.50,
                                      stratify=[y[i] for i in reserva],
                                      random_state=seed + 1)
    for conjunto, ids in [('train', treino), ('val', val), ('test', teste)]:
        if {itens[i]['classe'] for i in ids} != set(CLASSES):
            raise ValueError(f'{conjunto} sem ambas as classes; reveja grupos/tamanho.')
        for i in ids:
            itens[i]['conjunto'] = conjunto
            itens[i]['caminho'] = conjunto + '/' + itens[i]['origem']
    return dict(classes=CLASSES, seed=seed,
                unidade='grupo' if grupos_csv else 'imagem_exploratoria',
                fracao_alvo=dict(train=0.70, val=0.15, test=0.15), imagens=itens)


def executar(origem, destino, manifesto):
    destino = Path(destino)
    if destino.exists():
        raise ValueError('Destino já existe. Use pasta nova; não redistribua dados.')
    destino.mkdir(parents=True)
    for item in manifesto['imagens']:
        saida = destino / item['caminho']
        saida.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(origem) / item['origem'], saida)
    (destino / 'manifesto.json').write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding='utf-8')
    ler_manifesto(destino)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origem', type=Path, required=True)
    parser.add_argument('--dados', type=Path, default=DADOS)
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument('--grupos', type=Path, help='CSV: arquivo,grupo; caminho relativo à origem')
    modo.add_argument('--por-imagem', action='store_true', help='somente análise exploratória')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--executar', action='store_true')
    args = parser.parse_args()
    if args.dados.resolve().is_relative_to(args.origem.resolve()):
        parser.error('O destino deve ficar fora da origem.')
    plano = montar_plano(args.origem, args.grupos, args.por_imagem, args.seed)
    for conjunto in ('train', 'val', 'test'):
        print(conjunto, {c: sum(i['conjunto'] == conjunto and i['classe'] == c
                               for i in plano['imagens']) for c in CLASSES})
    print('Unidade de separação:', plano['unidade'])
    if args.executar:
        executar(args.origem, args.dados, plano)
        print('Divisão criada; treine NOVOS modelos. Fonte preservada.')
    else:
        print('Simulação: nenhuma imagem copiada.')


if __name__ == '__main__':
    main()
