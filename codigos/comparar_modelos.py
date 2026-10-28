import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from protocolo import (CLASSES, argumentos_avaliacao, carregar_conjunto,
                       conferir_modelo, predizer, salvar_predicoes)

# -- CONFIGURAÇÕES --

# Nome exibido no gráfico
MODELOS = {
    'CNN Própria': 'melhor_modelo_cnn_propria.keras',
    'VGG16': 'melhor_modelo_vgg16.keras',
    'ResNet50': 'melhor_modelo_resnet50.keras',
    'EfficientNetB0': 'melhor_modelo_efficientnet.keras',
}

# Classe considerada "positiva"
CLASSE_POSITIVA = 'Parasitized'

TIPO_MEDIA = 'weighted avg'

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

# -- FUNÇÕES AUXILIARES --
def pct(v, casas=2):
    """96.17 -> '96,17%' (padrão brasileiro)."""
    return f'{v:.{casas}f}%'.replace('.', ',')


def pp(v):
    """Diferença em pontos percentuais: 3.34 -> '−3,34 p.p.'"""
    return f'−{abs(v):.2f}'.replace('.', ',') + ' p.p.'


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
        'Imagens Avaliadas': tp + fn + fp + tn,
        'F1 Macro (%)': rel['macro avg']['f1-score'] * 100,
        'Verdadeiros Positivos': tp, 'Verdadeiros Negativos': tn,
    }


# -- GRÁFICO: RANKING + TABELA DE MÉTRICAS
def gerar_grafico(df, caminho_saida, conjunto='test'):
    n = len(df)
    melhor = df.iloc[0]
    nome_criterio = NOMES_CURTOS[CRITERIO]

    fig = plt.figure(figsize=(15.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.55)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # -- Painel esquerdo: ranking --
    valores = df[CRITERIO].values
    xmin = 0
    ys = np.arange(n)[::-1]  # 1º lugar no topo

    for i, (y, v) in enumerate(zip(ys, valores)):
        eh_maximo = np.isclose(v, valores.max(), atol=1e-10, rtol=0)
        cor = COR_MELHOR if eh_maximo else COR_OUTROS
        ax1.hlines(y, xmin, v, color=COR_LINHA, lw=4, zorder=1)
        ax1.scatter(v, y, s=520 if i == 0 else 380, color=cor, zorder=2,
                    edgecolor='white', linewidth=2)
        ax1.text(v + 1.1, y + (0.08 if i else 0), pct(v), va='center',
                 ha='left', fontsize=13, weight='bold', color=COR_TEXTO)
        if i > 0:
            ax1.text(v + 1.1, y - 0.22, ('Empate no valor' if eh_maximo else f'{pp(valores[0] - v)} vs. máximo'),
                     va='center', ha='left', fontsize=9, color='#666666')

    ax1.set_yticks(ys)
    ax1.set_yticklabels(list(df['Modelo']),
                        fontsize=12)
    ax1.get_yticklabels()[0].set_weight('bold')
    ax1.set_ylim(-0.7, n - 0.3)
    ax1.set_xlim(xmin, 112)
    ax1.set_xticks(range(xmin, 101, 20))
    ax1.set_xticklabels([f'{t}%' for t in range(xmin, 101, 20)], fontsize=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.tick_params(length=0)
    for lado in ('top', 'right', 'left'):
        ax1.spines[lado].set_visible(False)
    ax1.set_xlabel(f'{nome_criterio} ponderado (%)', fontsize=10)
    ax1.set_title(f'{nome_criterio} ponderado observado', fontsize=13, weight='bold',
                  pad=14)

    # -- Painel direito: tabela com destaque do melhor por coluna --
    ncols = len(METRICAS)
    ax2.set_xlim(-1.45, ncols)
    ax2.set_ylim(n, -0.9)
    ax2.axis('off')

    for j, met in enumerate(METRICAS):
        ax2.text(j + 0.5, -0.45, NOMES_CURTOS[met], ha='center', va='center',
                 fontsize=12, weight='bold')

    for i, row in df.iterrows():
        ax2.text(-0.08, i + 0.5, str(row['Modelo']), ha='right',
                 va='center', fontsize=12,
                 weight='bold' if i == 0 else 'normal')
        for j, met in enumerate(METRICAS):
            eh_melhor = np.isclose(row[met], df[met].max(), atol=1e-10, rtol=0)
            ax2.add_patch(Rectangle(
                (j, i), 1, 1,
                facecolor=COR_CEL_MELHOR if eh_melhor else COR_CEL_NORMAL,
                edgecolor='white', lw=4))
            ax2.text(j + 0.5, i + 0.5, pct(row[met]), ha='center',
                     va='center', fontsize=12.5,
                     weight='bold' if eh_melhor else 'normal',
                     color=COR_TEXTO)

    ax2.set_title('Métricas (médias ponderadas, exceto acurácia)',
                  fontsize=13, weight='bold', pad=14)

    titulo = 'Comparação descritiva dos modelos nesta rodada'
    total = f'{int(melhor["Imagens Avaliadas"]):,}'.replace(',', '.')
    rotulo = 'teste' if conjunto == 'test' else 'validação'
    subtitulo = f'{rotulo}: {total} imagens; sem inferência de superioridade estatística'

    if conjunto == 'val':
        subtitulo += ' — PRELIMINAR: validação usada na seleção dos pesos'
    fig.suptitle(titulo, fontsize=15, weight='bold', y=1.03)
    fig.text(0.5, 0.965, subtitulo, ha='center', fontsize=11, color='#555555')

    fig.savefig(caminho_saida, dpi=300, bbox_inches='tight', pad_inches=0.3,
                facecolor='white')
    plt.close(fig)


# -- EXECUÇÃO PRINCIPAL --
def main():
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    args = argumentos_avaliacao()
    caminhos = {nome: args.modelos_dir / arq for nome, arq in MODELOS.items()}
    # Uma comparação entre QUATRO modelos não pode omitir falhas silenciosamente.
    metadados = [conferir_modelo(p, args.manifesto_sha256) for p in caminhos.values()]
    for campo in ('seed', 'image_size', 'batch_size', 'epocas_maximas', 'learning_rate'):
        if any(m[campo] != metadados[0][campo] for m in metadados):
            raise ValueError(f'Rodadas incompatíveis: {campo}')
    conjunto = carregar_conjunto(args.dados, args.conjunto)
    arquivos = [os.path.relpath(p, args.dados) for p in conjunto.file_paths]
    linhas = []
    for nome, caminho in caminhos.items():
        y, pred, prob = predizer(caminho, conjunto)
        stem = caminho.stem.replace('melhor_modelo_', '')
        salvar_predicoes(args, stem, caminho, y, pred, prob, arquivos)
        rel = classification_report(y, pred, labels=[0, 1], target_names=CLASSES,
                                    digits=4, zero_division=0, output_dict=True)
        (args.saida / f'relatorio_{stem}.json').write_text(
            json.dumps(rel, ensure_ascii=False, indent=2), encoding='utf-8')
        resultados = calcular_resultados(y, pred, CLASSES)
        positivos = y == CLASSES.index(CLASSE_POSITIVA)
        score = prob[:, CLASSES.index(CLASSE_POSITIVA)]
        resultados['ROC AUC Parasitized'] = roc_auc_score(positivos, score)
        resultados['Average Precision Parasitized'] = average_precision_score(positivos, score)
        linhas.append({'Modelo': nome, 'Conjunto': args.conjunto, **resultados})
    df = pd.DataFrame(linhas).sort_values(CRITERIO, ascending=False).reset_index(drop=True)
    print(df.round(2).to_string(index=False))
    # Preserva precisão numérica; arredondamento apenas na exibição.
    df.to_csv(args.saida / ARQ_CSV, index=False)
    gerar_grafico(df, args.saida / ARQ_GRAFICO, args.conjunto)
    if args.conjunto == 'val':
        print('PRELIMINAR: validação usada para selecionar pesos; não é teste.')


if __name__ == '__main__':
    main()
