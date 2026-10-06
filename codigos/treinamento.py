import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory

# ==========================================
# 1. CONFIGURAÇÕES GERAIS
# ==========================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
DATA_DIR = "../dados"

# Escolha o modelo para rodar: "cnn_propria", "vgg16" ou "resnet50"
MODELO = "cnn_propria"

# ==========================================
# 2. CARREGAMENTO DOS DADOS (80% Treino / 20% Validação)
# ==========================================
PASTA_IMAGENS = "../dados/train"

print("Carregando conjunto de treino...")
train_gen = image_dataset_from_directory(
    PASTA_IMAGENS,
    validation_split=0.2,
    subset="training",
    seed=123,  # Semente fixa para manter a mesma ordem
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
)

# Validação: TAMBÉM DEVE TER shuffle=True durante a criação para garantir que o split pegue ambas as classes
val_gen = image_dataset_from_directory(
    PASTA_IMAGENS,
    validation_split=0.2,
    subset="validation",
    seed=123,  # Mesma semente do treino!
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,  # O Keras usará isso para separar as duas classes proporcionalmente
)

# Otimização de leitura em memória
train_gen = train_gen.prefetch(buffer_size=tf.data.AUTOTUNE)
val_gen = val_gen.prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 3. CAMADA DE PRÉ-PROCESSAMENTO & AUGMENTATION
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1. / 255),
    layers.RandomRotation(0.1),
    layers.RandomFlip("horizontal_and_vertical")
], name="augmentation")


# ==========================================
# 4. CONSTRUÇÃO DAS ARQUITETURAS
# ==========================================
def construir_modelo(tipo_modelo, input_shape=(128, 128, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)

    if tipo_modelo == "cnn_propria":
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

    elif tipo_modelo == "vgg16":
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

    elif tipo_modelo == "resnet50":
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    else:
        raise ValueError("Modelo não reconhecido! Escolha 'cnn_propria', 'vgg16' ou 'resnet50'.")

    return models.Model(inputs=inputs, outputs=outputs)


model = construir_modelo(MODELO, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=2)

# ==========================================
# 5. COMPILAÇÃO E MÉTRICAS
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# ==========================================
# 6. CALLBACKS E TREINAMENTO
# ==========================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=f"melhor_modelo_{MODELO}.keras", monitor='val_loss', save_best_only=True, verbose=1)
]

print(f"\nIniciando treinamento do modelo: {MODELO.upper()}...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# 7. GRÁFICOS DE DESEMPENHO
# ==========================================
plt.figure(figsize=(12, 4))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title(f'Acurácia - {MODELO}')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

# Gráfico de Perda (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title(f'Perda (Loss) - {MODELO}')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f"grafico_{MODELO}.png")
plt.show()

print(f"\nTreinamento de {MODELO} concluído e modelo salvo com sucesso!")