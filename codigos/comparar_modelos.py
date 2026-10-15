import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_IMAGENS = os.path.join(BASE_DIR, '../dados/train')

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
# Mesma pasta de imagens usada no treinamento.py / avaliacao.py
# Pasta onde estão os arquivos .keras (também procura na pasta deste script)
PASTA_MODELOS = BASE_DIR

# Nome exibido no gráfico -> arquivo salvo pelo treinamento.py
MODELOS = {
    'CNN Própria': 'melhor_modelo_cnn_propria.keras',
    'VGG16': 'melhor_modelo_vgg16.keras',
    'ResNet50': 'melhor_modelo_resnet50.keras',
    'EfficientNet': 'melhor_modelo_efficientnet.keras',
}

# Devem ser IGUAIS aos usados no treinamento.py e no avaliacao.py
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 123
VALIDATION_SPLIT = 0.2

# Classe considerada "positiva" (a que queremos detectar)
CLASSE_POSITIVA = 'Parasitized'

# Tipo de média usada em Precisão / Recall / F1:
#   'weighted avg' -> média ponderada pelo nº de imagens de cada classe
#                     (o Recall médio fica igual à acurácia)
#   'macro avg'    -> média simples entre as duas classes
TIPO_MEDIA = 'weighted avg'

# Métrica usada para ordenar o ranking (critério de "mais efetivo")
CRITERIO = 'F1-Score Médio (%)'

METRICAS = ['Acurácia (%)', 'Precisão Média (%)', 'Recall Médio (%)',
            'F1-Score Médio (%)']
NOMES_CURTOS = {
    'Acurácia (%)': 'Acurácia',
    'Precisão Média (%)': 'Precisão',
    'Recall Médio (%)': 'Recall',
    'F1-Score Médio (%)': 'F1-Score',
}

ARQ_CSV = 'tabela_comparativa_modelos.csv'
ARQ_GRAFICO = 'grafico_comparativo_geral.png'

# Cores
COR_MELHOR = '#2E7D32'
COR_OUTROS = '#5B8DB8'
COR_LINHA = '#C5D3E0'
COR_CEL_MELHOR = '#A5D6A7'
COR_CEL_NORMAL = '#F1F4F8'
COR_TEXTO = '#1B1B1B'


# ==========================================
# 2. FUNÇÕES AUXILIARES
# ==========================================
def pct(v, casas=2):
    """96.17 -> '96,17%' (padrão brasileiro)."""
    return f'{v:.{casas}f}%'.replace('.', ',')


def pp(v):
    """Diferença em pontos percentuais: 3.34 -> '−3,34 p.p.'"""
    return f'−{abs(v):.2f}'.replace('.', ',') + ' p.p.'


def localizar_modelo(arquivo):
    """Procura o .keras na PASTA_MODELOS e na pasta do script."""
    pastas = [PASTA_MODELOS, os.path.dirname(os.path.abspath(__file__))]
    for pasta in pastas:
        caminho = os.path.join(pasta, arquivo)
        if os.path.exists(caminho):
            return caminho
    return None


def calcular_resultados(y_true, y_pred, class_names):
    """Calcula as métricas de um modelo a partir de y_true e y_pred."""
    from sklearn.metrics import classification_report, confusion_matrix

    rotulos = list(range(len(class_names)))
    rel = classification_report(
        y_true, y_pred, labels=rotulos, target_names=class_names,
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=rotulos)

    pos = class_names.index(CLASSE_POSITIVA)
    neg = 1 - pos
    tp, fn = int(cm[pos, pos]), int(cm[pos, neg])
    fp, tn = int(cm[neg, pos]), int(cm[neg, neg])

    media = rel[TIPO_MEDIA]
    return {
        'Acurácia (%)': rel['accuracy'] * 100,
        'Precisão Média (%)': media['precision'] * 100,
        'Recall Médio (%)': media['recall'] * 100,
        'F1-Score Médio (%)': media['f1-score'] * 100,
        # Extras (úteis para o TCC; aparecem no CSV, não no gráfico)
        'Sensibilidade (%)': 100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
        'Especificidade (%)': 100.0 * tn / (tn + fp) if (tn + fp) else 0.0,
        'Falsos Negativos': fn,
        'Falsos Positivos': fp,
        'Imagens de Validação': tp + fn + fp + tn,
    }


def carregar_validacao():
    """Carrega o conjunto de validação (mesma seed/proporção do treino)."""
    from tensorflow.keras.utils import image_dataset_from_directory

    if not os.path.isdir(PASTA_IMAGENS):
        print(f'\n[ERRO] Pasta de imagens não encontrada:\n       {PASTA_IMAGENS}')
        print('       Ajuste a variável PASTA_IMAGENS no início do script.')
        sys.exit(1)

    print('Carregando conjunto de validação padronizado '
          f'(seed={SEED}, split={VALIDATION_SPLIT})...')
    val_gen = image_dataset_from_directory(
        PASTA_IMAGENS,
        validation_split=VALIDATION_SPLIT,
        subset='validation',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,  # necessário para o split correto (igual ao treino)
    )
    class_names = list(val_gen.class_names)
    print(f'Classes encontradas: {class_names}')

    if CLASSE_POSITIVA not in class_names or len(class_names) != 2:
        print(f"\n[ERRO] Esperava 2 classes incluindo '{CLASSE_POSITIVA}', "
              f'mas encontrei: {class_names}')
        sys.exit(1)
    return val_gen, class_names


def avaliar_modelo(caminho, val_gen, class_names):
    """Carrega um .keras, prediz o conjunto de validação e devolve métricas."""
    import tensorflow as tf
    from sklearn.metrics import classification_report
    from tensorflow.keras.models import load_model

    model = load_model(caminho)

    # y_true e y_pred coletados NO MESMO loop: como val_gen tem shuffle=True,
    # percorrê-lo duas vezes geraria ordens diferentes e os rótulos não
    # corresponderiam às predições.
    y_true, y_pred = [], []
    for images, labels in val_gen:
        preds = model(images, training=False).numpy()
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(classification_report(y_true, y_pred, target_names=class_names,
                                digits=4, zero_division=0))

    resultados = calcular_resultados(y_true, y_pred, class_names)

    # Libera memória antes do próximo modelo
    del model
    tf.keras.backend.clear_session()
    return resultados


# ==========================================
# 3. GRÁFICO: RANKING (PIRULITO) + TABELA DE MÉTRICAS
# ==========================================
def gerar_grafico(df, caminho_saida):
    n = len(df)
    melhor = df.iloc[0]
    nome_criterio = NOMES_CURTOS[CRITERIO]

    fig = plt.figure(figsize=(15.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.55)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ---------- Painel esquerdo: ranking (pirulito) ----------
    valores = df[CRITERIO].values
    xmin = max(0, 5 * math.floor((valores.min() - 3) / 5))
    ys = np.arange(n)[::-1]  # 1º lugar no topo

    for i, (y, v) in enumerate(zip(ys, valores)):
        cor = COR_MELHOR if i == 0 else COR_OUTROS
        ax1.hlines(y, xmin, v, color=COR_LINHA, lw=4, zorder=1)
        ax1.scatter(v, y, s=520 if i == 0 else 380, color=cor, zorder=2,
                    edgecolor='white', linewidth=2)
        ax1.text(v + 1.1, y + (0.08 if i else 0), pct(v), va='center',
                 ha='left', fontsize=13, weight='bold', color=COR_TEXTO)
        if i > 0:
            ax1.text(v + 1.1, y - 0.22, f'{pp(valores[0] - v)} vs. melhor',
                     va='center', ha='left', fontsize=9, color='#666666')

    ax1.set_yticks(ys)
    ax1.set_yticklabels([f'{i + 1}º  {m}' for i, m in enumerate(df['Modelo'])],
                        fontsize=12)
    ax1.get_yticklabels()[0].set_weight('bold')
    ax1.set_ylim(-0.7, n - 0.3)
    ax1.set_xlim(xmin, 112)
    ax1.set_xticks(range(xmin, 101, 5))
    ax1.set_xticklabels([f'{t}%' for t in range(xmin, 101, 5)], fontsize=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.tick_params(length=0)
    for lado in ('top', 'right', 'left'):
        ax1.spines[lado].set_visible(False)
    ax1.set_xlabel(f'{nome_criterio} (%) — o eixo começa em {xmin}% '
                   'para evidenciar as diferenças', fontsize=10)
    ax1.set_title(f'Ranking por {nome_criterio}', fontsize=13, weight='bold',
                  pad=14)

    # ---------- Painel direito: tabela com destaque do melhor por coluna ----
    ncols = len(METRICAS)
    ax2.set_xlim(-1.45, ncols)
    ax2.set_ylim(n, -0.9)
    ax2.axis('off')

    for j, met in enumerate(METRICAS):
        ax2.text(j + 0.5, -0.45, NOMES_CURTOS[met], ha='center', va='center',
                 fontsize=12, weight='bold')

    for i, row in df.iterrows():
        ax2.text(-0.08, i + 0.5, f'{i + 1}º  {row["Modelo"]}', ha='right',
                 va='center', fontsize=12,
                 weight='bold' if i == 0 else 'normal')
        for j, met in enumerate(METRICAS):
            eh_melhor = np.isclose(row[met], df[met].max())
            ax2.add_patch(Rectangle(
                (j, i), 1, 1,
                facecolor=COR_CEL_MELHOR if eh_melhor else COR_CEL_NORMAL,
                edgecolor='white', lw=4))
            ax2.text(j + 0.5, i + 0.5, pct(row[met]), ha='center',
                     va='center', fontsize=12.5,
                     weight='bold' if eh_melhor else 'normal',
                     color=COR_TEXTO)

    ax2.set_title('Todas as métricas',
                  fontsize=13, weight='bold', pad=14)

    if n > 1:
        titulo = (f'{melhor["Modelo"]} foi a arquitetura mais efetiva na '
                  f'detecção de malária ({nome_criterio} de '
                  f'{pct(melhor[CRITERIO])})')
        total_val = f'{int(melhor["Imagens de Validação"]):,}'.replace(',', '.')
        subtitulo = ('Comparação entre as arquiteturas de CNN no conjunto de '
                     f'validação ({total_val} imagens)')
    else:
        titulo = (f'Desempenho do modelo {melhor["Modelo"]} na detecção de '
                  f'malária ({nome_criterio} de {pct(melhor[CRITERIO])})')
        subtitulo = ('Apenas um modelo avaliado - treine os demais para '
                     'habilitar a comparação')

    fig.suptitle(titulo, fontsize=15, weight='bold', y=1.03)
    fig.text(0.5, 0.965, subtitulo, ha='center', fontsize=11, color='#555555')

    fig.savefig(caminho_saida, dpi=300, bbox_inches='tight', pad_inches=0.3,
                facecolor='white')
    plt.close(fig)


# ==========================================
# 4. EXECUÇÃO PRINCIPAL
# ==========================================
def main():
    print('=' * 75)
    print('COMPARAÇÃO AUTOMÁTICA DE MODELOS - TCC (detecção de malária)')
    print('=' * 75)

    # 1) Descobre quais modelos já foram treinados (antes de carregar o TF)
    encontrados = {}
    for nome, arquivo in MODELOS.items():
        caminho = localizar_modelo(arquivo)
        if caminho is None:
            print(f"[AVISO] Modelo '{nome}' ainda não foi treinado "
                  f"('{arquivo}' não encontrado). Pulando...")
        else:
            encontrados[nome] = caminho

    if not encontrados:
        print('\n[ERRO] Nenhum arquivo .keras foi encontrado.')
        print('       Execute o treinamento.py primeiro ou ajuste a variável '
              'PASTA_MODELOS.')
        sys.exit(1)

    # 2) Carrega o conjunto de validação uma única vez
    val_gen, class_names = carregar_validacao()

    # 3) Avalia cada modelo encontrado
    linhas = []
    for nome, caminho in encontrados.items():
        print('\n' + '=' * 75)
        print(f'Avaliando modelo: [{nome}]  ({os.path.basename(caminho)})')
        print('=' * 75)
        try:
            resultados = avaliar_modelo(caminho, val_gen, class_names)
        except Exception as erro:  # arquivo corrompido, versão incompatível...
            print(f"[AVISO] Falha ao avaliar '{nome}': {erro}\n"
                  '        Pulando este modelo...')
            continue
        linhas.append({'Modelo': nome, **resultados})

    if not linhas:
        print('\n[ERRO] Nenhum modelo pôde ser avaliado.')
        sys.exit(1)

    # 4) Tabela final ordenada pelo critério
    df = pd.DataFrame(linhas)
    df = df.sort_values(CRITERIO, ascending=False).reset_index(drop=True)

    print('\n' + '=' * 75)
    print(f'TABELA COMPARATIVA FINAL (ordenada por {CRITERIO})')
    print('=' * 75)
    print(df.round(2).to_string(index=False))
    print('=' * 75)

    df.round(2).to_csv(os.path.join(BASE_DIR, ARQ_CSV), index=False)
    print(f"\n[arquivo] Tabela salva como '{ARQ_CSV}'")

    # 5) Gráfico
    gerar_grafico(df, os.path.join(BASE_DIR, ARQ_GRAFICO))
    print(f"[arquivo] Gráfico salvo como '{ARQ_GRAFICO}'")

    plt.show()


if __name__ == '__main__':
    main()