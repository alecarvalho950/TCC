import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
PASTA_IMAGENS = r'C:\Users\GAMER\OneDrive\Documentos\Desenvolvendo Sites\TCC-Imagens celulares\estudos-imagens-celulares\dados\train'
MODEL_PATH = r'melhor_modelo_cnn_propria.keras'

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print('Carregando modelo e dados de validação/teste...')

# Carregar o modelo treinado
model = load_model(MODEL_PATH)

# ==========================================
# 2. CARREGAR DATASET DE VALIDAÇÃO (COM SHUFFLE=TRUE PARA FAZER A DIVISÃO CORRETA)
# ==========================================
# Importante: shuffle=True e a MESMA seed (123) do treinamento são essenciais
# para que o split 80/20 selecione 50% de cada classe na validação.
val_gen = image_dataset_from_directory(
    PASTA_IMAGENS,
    validation_split=0.2,
    subset='validation',
    seed=123,  # Mesma semente usada no treino
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,  # Garante que a validação receba ambas as classes!
)

class_names = val_gen.class_names
print(f'Classes encontradas: {class_names}')

# ==========================================
# 3. EXTRAIR RÓTULOS REAIS (y_true) E PREDIÇÕES (y_pred)
# ==========================================
print('Calculando predições...')

y_true = []
y_pred = []

# Iteramos lote a lote para garantir o alinhamento exato entre imagem e rótulo
for images, labels in val_gen:
  preds = model.predict(images, verbose=0)

  # Pegar os índices reais e preditos de cada lote
  y_true.extend(np.argmax(labels.numpy(), axis=1))
  y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==========================================
# 4. EXIBIR RESULTADOS
# ==========================================
print('\n--- RELATÓRIO DE AVALIAÇÃO DO ARTIGO ---')
print(
    classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
)

cm = confusion_matrix(y_true, y_pred)
print('--- MATRIZ DE CONFUSÃO ---')
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title('Matriz de Confusão - CNN Própria')
plt.ylabel('Classe Real')
plt.xlabel('Classe Predita')
plt.tight_layout()
plt.savefig('matriz_confusao_cnn_propria.png', dpi=300)
plt.show()