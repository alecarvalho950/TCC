import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import FuncFormatter, MaxNLocator
from tensorflow.keras import applications, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_IMAGENS = os.path.join(BASE_DIR, '../dados/train')

# ==========================================
# 1. CONFIGURAÇÕES GERAIS
# ==========================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5  # Ajuste para o total de épocas desejado por modelo

# Lista com todos os modelos que serão treinados em sequência automática
LISTA_MODELOS = ['cnn_propria', 'vgg16', 'resnet50', 'efficientnet']

# ==========================================
# 2. CARREGAMENTO DOS DADOS (80% Treino / 20% Validação)
# ==========================================
print('Carregando conjunto de treino...')
train_gen = image_dataset_from_directory(
    PASTA_IMAGENS,
    validation_split=0.2,
    subset='training',
    seed=123,  # Semente fixa para manter a mesma ordem
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
)

# Validação: Mantém shuffle=True para garantir o split proporcional correto
val_gen = image_dataset_from_directory(
    PASTA_IMAGENS,
    validation_split=0.2,
    subset='validation',
    seed=123,  # Mesma semente do treino!
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
)

# Otimização de leitura em memória
train_gen = train_gen.prefetch(buffer_size=tf.data.AUTOTUNE)
val_gen = val_gen.prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 3. CAMADA DE PRÉ-PROCESSAMENTO & AUGMENTATION
# ==========================================
data_augmentation = tf.keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.RandomRotation(0.1),
        layers.RandomFlip('horizontal_and_vertical'),
    ],
    name='augmentation',
)

data_augmentation_efficientnet = tf.keras.Sequential(
    [
        layers.RandomRotation(0.1),
        layers.RandomFlip('horizontal_and_vertical'),
    ],
    name='augmentation_efficientnet',
)


# ==========================================
# 4. CONSTRUÇÃO DAS ARQUITETURAS
# ==========================================
def construir_modelo(tipo_modelo, input_shape=(128, 128, 3), num_classes=2):
  inputs = layers.Input(shape=input_shape)

  if tipo_modelo == 'cnn_propria':
    x = data_augmentation(inputs)
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
    x = data_augmentation(inputs)
    base_model = applications.VGG16(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base_model.trainable = False
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

  elif tipo_modelo == 'resnet50':
    x = data_augmentation(inputs)
    base_model = applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base_model.trainable = False
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

  elif tipo_modelo == 'efficientnet':
    # EfficientNet exige pixels de 0 a 255 (sem Rescaling)
    x = data_augmentation_efficientnet(inputs)
    base_model = applications.EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base_model.trainable = False
    x = base_model(x)
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


# ==========================================
# 5. GRÁFICO DO HISTÓRICO DE TREINAMENTO
# ==========================================
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
  # Mesmo critério do ModelCheckpoint/EarlyStopping (monitor='val_loss')
  melhor = int(np.argmin(val_loss))
  x_melhor = melhor + 1

  def fmt_pct(v, _=None):
    return f'{v * 100:.1f}%'.replace('.', ',')

  def fmt_loss(v, _=None):
    return f'{v:.2f}'.replace('.', ',')

  def desenhar(ax, treino, valid, titulo, ylabel, maior_melhor, fmt,
               pad_min):
    tam = 9 if n <= 3 else 5
    ax.plot(epocas, treino, color=COR_TREINO, marker='o', ms=tam, lw=2.2,
            label='Treino')
    ax.plot(epocas, valid, color=COR_VALID, marker='o', ms=tam, lw=2.2,
            label='Validação')

    # Época do modelo salvo
    ax.axvline(x_melhor, color=COR_MELHOR, ls='--', lw=1.4, alpha=0.8,
               zorder=1)
    ax.scatter([x_melhor], [valid[melhor]], marker='*', s=320,
               color=COR_MELHOR, edgecolor='white', zorder=6,
               label=f'Melhor época ({x_melhor}) - modelo salvo')

    # Valores na melhor época (o maior fica acima, o menor abaixo)
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
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0.5, n + 0.5)
    if n <= 20:
      ax.set_xticks(epocas)
    else:
      ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    # Margem vertical com folga mínima (evita eixo "apertado" com 1 época)
    todos = list(treino) + list(valid)
    lo, hi = min(todos), max(todos)
    folga = max((hi - lo) * 0.25, pad_min)
    topo = hi + folga
    if fmt is fmt_pct:  # acurácia não passa de 100%
      topo = min(topo, 1.02)
    ax.set_ylim(lo - folga, topo)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    for lado in ('top', 'right'):
      ax.spines[lado].set_visible(False)

    # Seta verde: sentido de "melhor" (fora da área do gráfico, à direita)
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
  desenhar(ax1, acc, val_acc,
           'Acurácia  (quanto MAIOR, melhor)', 'Acurácia', True, fmt_pct,
           0.03)
  desenhar(ax2, loss, val_loss,
           'Perda / Loss  (quanto MENOR, melhor)', 'Perda/Loss', False, fmt_loss,
           0.08)

  fig.suptitle(f'Histórico de treinamento - {nome_modelo.upper()}',
               fontsize=15, weight='bold', y=0.995)
  handles, rotulos = ax1.get_legend_handles_labels()
  fig.legend(handles, rotulos, loc='upper center', ncol=3, fontsize=10.5,
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


# ==========================================
# 6. LOOP DE TREINAMENTO AUTOMATIZADO
# ==========================================
for MODELO in LISTA_MODELOS:
  print(f"\n{'='*60}")
  print(f'INICIANDO PIPELINE DO MODELO: {MODELO.upper()}')
  print(f"{'='*60}")

  model = construir_modelo(
      MODELO, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=2
  )

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
      loss='categorical_crossentropy',
      metrics=[
          'accuracy',
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
      ],
  )

  callbacks = [
      EarlyStopping(
          monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
      ),
      ModelCheckpoint(
          filepath=os.path.join(BASE_DIR, f'melhor_modelo_{MODELO}.keras'),
          monitor='val_loss',
          save_best_only=True,
          verbose=1,
      ),
  ]

  print(f'\nIniciando treinamento do modelo: {MODELO.upper()}...')
  history = model.fit(
      train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks
  )

  # Gráfico de desempenho individual salvo para cada arquitetura
  plotar_historico(
      history, MODELO, os.path.join(BASE_DIR, f'grafico_{MODELO}.png')
  )

  print(f'\nTreinamento de {MODELO} concluído e modelo salvo com sucesso!')

print(f"\n{'='*60}")
print('TODOS OS MODELOS FORAM TREINADOS E SALVOS COM SUCESSO!')
print(f"{'='*60}")