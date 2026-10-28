import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import FuncFormatter, MaxNLocator
from tensorflow.keras import applications, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessamento import PreprocessamentoImageNet
from protocolo import (BATCH_SIZE, CLASSES, DADOS, IMG_SIZE, SAIDA, SEED, VERSAO, carregar_conjunto,
                       ler_manifesto, sha256)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -- CONFIGURAÇÕES GERAIS --
EPOCHS = 20

LISTA_MODELOS = ['cnn_propria', 'vgg16', 'resnet50', 'efficientnet']

# -- CONSTRUÇÃO DAS ARQUITETURAS --
def construir_modelo(tipo_modelo, input_shape=(128, 128, 3), num_classes=2,
                     weights='imagenet', seed=SEED):
    inputs = layers.Input(shape=input_shape, name='rgb_0_255')
    # Instâncias e sementes novas por modelo; augmentation só atua no treino.
    augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1, seed=seed),
        layers.RandomFlip('horizontal_and_vertical', seed=seed + 1),
    ], name='augmentation')
    x = augmentation(inputs)

    if tipo_modelo == 'cnn_propria':
        x = layers.Rescaling(1.0 / 255, name='escala_0_1')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    elif tipo_modelo == 'vgg16':
        x = PreprocessamentoImageNet('vgg16', name='preprocess_vgg16')(x)
        base_model = applications.VGG16(
            weights=weights, include_top=False, input_shape=input_shape
        )
        base_model.trainable = False
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    elif tipo_modelo == 'resnet50':
        x = PreprocessamentoImageNet('resnet50', name='preprocess_resnet50')(x)
        base_model = applications.ResNet50(
            weights=weights, include_top=False, input_shape=input_shape
        )
        base_model.trainable = False
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    elif tipo_modelo == 'efficientnet':
        # EfficientNet exige pixels de 0 a 255 (sem Rescaling)
        base_model = applications.EfficientNetB0(
            weights=weights, include_top=False, input_shape=input_shape
        )
        base_model.trainable = False
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    else:
        raise ValueError(
            "Modelo não reconhecido! Escolha 'cnn_propria', 'vgg16', 'resnet50' ou"
            " 'efficientnet'."
        )

    return models.Model(inputs=inputs, outputs=outputs)


# -- GRÁFICO DO HISTÓRICO DE TREINAMENTO --
COR_TREINO = '#1F5FA8'
COR_VALID = '#E67E22'
COR_MELHOR = '#2E7D32'


def plotar_historico(history, nome_modelo, caminho_saida):
    """Gráfico de acurácia e perda (treino x validação) fácil de interpretar.

    - Marcadores em cada época (aparece mesmo com 1 época).
    - Seta verde indicando o sentido de "melhor" em cada gráfico.
    - Estrela na época do modelo salvo (menor perda de validação).
    - Resumo com os valores finais no rodapé.
    """
    h = history.history
    acc, val_acc = h['accuracy'], h['val_accuracy']
    loss, val_loss = h['loss'], h['val_loss']
    n = len(loss)
    epocas = list(range(1, n + 1))
    melhor = int(np.argmin(val_loss))
    x_melhor = melhor + 1

    def fmt_pct(v, _=None):
        return f'{v * 100:.1f}%'.replace('.', ',')

    def fmt_loss(v, _=None):
        return f'{v:.2f}'.replace('.', ',')

    def desenhar(ax, treino, valid, titulo, maior_melhor, fmt, pad_min):
        tam = 9 if n <= 3 else 5
        ax.plot(epocas, treino, color=COR_TREINO, marker='o', ms=tam, lw=2.2,
                label='Treino')
        ax.plot(epocas, valid, color=COR_VALID, marker='o', ms=tam, lw=2.2,
                label='Validação')

        ax.axvline(x_melhor, color=COR_MELHOR, ls='--', lw=1.4, alpha=0.8,
                   zorder=1)
        ax.scatter([x_melhor], [valid[melhor]], marker='*', s=320,
                   color=COR_MELHOR, edgecolor='white', zorder=6)

        v_cima = valid[melhor] >= treino[melhor]
        for serie, cor, acima in (
                (valid, COR_VALID, v_cima),
                (treino, COR_TREINO, not v_cima),
        ):
            ax.annotate(
                fmt(serie[melhor]), (x_melhor, serie[melhor]),
                xytext=(0, 13 if acima else -19), textcoords='offset points',
                ha='center', fontsize=10.5, weight='bold', color=cor,
            )

        ax.set_title(titulo, fontsize=13, weight='bold', pad=12)
        ax.set_xlabel('Época', fontsize=11)
        ax.set_xlim(0.5, n + 0.5)
        if n <= 20:
            ax.set_xticks(epocas)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))

        todos = list(treino) + list(valid)
        lo, hi = min(todos), max(todos)
        folga = max((hi - lo) * 0.25, pad_min)
        topo = hi + folga
        if fmt is fmt_pct:
            topo = min(topo, 1.02)
        ax.set_ylim(lo - folga, topo)
        ax.grid(linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        for lado in ('top', 'right'):
            ax.spines[lado].set_visible(False)

        # Seta verde: sentido de "melhor" à direita
        y0, y1 = (0.60, 0.95) if maior_melhor else (0.40, 0.05)
        ax.annotate(
            '', xy=(1.03, y1), xytext=(1.03, y0), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>', color=COR_MELHOR, lw=2.5),
            annotation_clip=False,
        )
        ax.text(1.065, (y0 + y1) / 2, 'melhor', transform=ax.transAxes,
                rotation=90, va='center', ha='center', fontsize=10,
                weight='bold', color=COR_MELHOR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))
    desenhar(ax1, acc, val_acc, 'Acurácia', True, fmt_pct, 0.03)
    desenhar(ax2, loss, val_loss, 'Perda / Loss', False, fmt_loss, 0.08)

    fig.suptitle(f'Histórico de treinamento - {nome_modelo.upper()}',
                 fontsize=15, weight='bold', y=0.995)

    handles, rotulos = ax1.get_legend_handles_labels()
    fig.legend(handles, rotulos, loc='upper center', ncol=2, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, 0.955))

    resumo = (
        f'Modelo salvo na época {x_melhor} de {n} (menor perda de validação)   |'
        f'   Acurácia: treino {fmt_pct(acc[melhor])}, validação '
        f'{fmt_pct(val_acc[melhor])}   |   Perda: treino '
        f'{fmt_loss(loss[melhor])}, validação {fmt_loss(val_loss[melhor])}'
    )
    if n == 1:
        resumo += ('\nTreinamento com apenas 1 época: aumente EPOCHS para ver a '
                   'evolução ao longo do tempo.')
    fig.text(0.5, 0.01, resumo, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa',
                       edgecolor='#cccccc'))

    fig.tight_layout(rect=(0, 0.1, 0.97, 0.9))
    fig.savefig(caminho_saida, dpi=200, bbox_inches='tight', pad_inches=0.3,
                facecolor='white')
    plt.close(fig)  # Libera memória no loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dados', type=Path, default=DADOS)
    parser.add_argument('--saida', type=Path, default=SAIDA)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--epocas', type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epocas < 1:
        parser.error('--epocas deve ser positivo')
    manifesto, digest = ler_manifesto(args.dados)
    if args.saida.exists():
        parser.error('Pasta de saída já existe; use outra para preservar a rodada.')
    args.saida.mkdir(parents=True)
    tf.config.experimental.enable_op_determinism()
    ambiente = {p: importlib.metadata.version(p) for p in
                ('keras', 'numpy', 'pandas', 'matplotlib', 'scikit-learn')}
    ambiente['tensorflow'] = tf.__version__
    ambiente['python'] = platform.python_version()
    ambiente['plataforma'] = platform.platform()
    fontes = {p.name: sha256(p) for p in Path(BASE_DIR).glob('*.py')}
    for nome in LISTA_MODELOS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        # Recria o dataset para reiniciar o embaralhamento em cada arquitetura.
        treino = carregar_conjunto(args.dados, 'train', args.seed).prefetch(tf.data.AUTOTUNE)
        validacao = carregar_conjunto(args.dados, 'val').prefetch(tf.data.AUTOTUNE)
        modelo = construir_modelo(nome, seed=args.seed)
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss='categorical_crossentropy', metrics=['accuracy'])
        caminho = args.saida / f'melhor_modelo_{nome}.keras'
        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', patience=5,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=str(caminho), monitor='val_loss', mode='min',
                            save_best_only=True, verbose=1),
        ]
        historia = modelo.fit(treino, validation_data=validacao,
                              epochs=args.epocas, callbacks=callbacks)
        plotar_historico(historia, nome, args.saida / f'grafico_{nome}.png')
        (args.saida / f'historico_{nome}.json').write_text(
            json.dumps(historia.history, indent=2), encoding='utf-8')
        meta = dict(protocolo=VERSAO, modelo=nome, classes=CLASSES,
                    manifesto_sha256=digest, modelo_sha256=sha256(caminho),
                    unidade=manifesto['unidade'], seed=args.seed,
                    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
                    epocas_maximas=args.epocas,
                    melhor_epoca=int(np.argmin(historia.history['val_loss'])) + 1,
                    criterio='min val_loss', patience=5, learning_rate=1e-4,
                    transferencia='base ImageNet congelada' if nome != 'cnn_propria'
                    else 'treinamento integral, inicialização aleatória',
                    parametros=modelo.count_params(),
                    parametros_treinaveis=int(sum(np.prod(w.shape) for w in modelo.trainable_weights)),
                    ambiente=ambiente, fontes_sha256=fontes)
        caminho.with_suffix('.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Rodada concluída. Teste não utilizado no treinamento.')


if __name__ == '__main__':
    main()
