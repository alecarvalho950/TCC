import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from protocolo import (CLASSES, argumentos_avaliacao, carregar_conjunto,
                       conferir_modelo, predizer, salvar_predicoes)
from comparar_modelos import MODELOS

# -- CONFIGURAÇÕES --
# Classe considerada "positiva"
CLASSE_POSITIVA = 'Parasitized'

# Cores da Matriz de Confusão
COR_ACERTO = '#C8E6C9'
COR_ERRO = '#FFCDD2'
COR_ERRO_CRITICO = '#EF9A9A'
COR_TEXTO = '#1B1B1B'
COR_TEXTO_SUAVE = '#555555'

# Cores diferentes para cada métrica de desempenho
CORES_METRICAS = ['#1F5FA8', '#2E7D32', '#8E44AD', '#D35400', '#16A085']
COR_FUNDO_BARRA = '#EAECEE'


# -- FUNÇÕES AUXILIARES DO GRÁFICO --
def fmt_int(n):
    """1469 -> '1.469' (padrão brasileiro)."""
    return f'{int(n):,}'.replace(',', '.')


def fmt_pct(v):
    """54.23 -> '54,2%' (padrão brasileiro)."""
    return f'{v:.1f}%'.replace('.', ',')


def calcular_metricas(tp, fn, fp, tn):
    total = tp + fn + fp + tn

    def div(a, b):
        return 100.0 * a / b if b > 0 else 0.0

    acuracia = div(tp + tn, total)
    sensibilidade = div(tp, tp + fn)
    especificidade = div(tn, tn + fp)
    precisao = div(tp, tp + fp)
    f1 = (
        2 * precisao * sensibilidade / (precisao + sensibilidade)
        if (precisao + sensibilidade) > 0
        else 0.0
    )
    return {
        'acuracia': acuracia,
        'sensibilidade': sensibilidade,
        'especificidade': especificidade,
        'precisao': precisao,
        'f1': f1,
    }


def gerar_grafico(cm, class_names, nome_modelo, caminho_saida, conjunto='test'):
    """Gera figura com a matriz de confusão (esq.) e barras de métricas (dir.).

    Os dois painéis compartilham a mesma caixa (mesmo topo e mesma base) e o
    mesmo sistema de coordenadas verticais, então as 5 barras ficam alinhadas
    com o bloco de 2 linhas da matriz. Os títulos de seção ficam na mesma
    altura, posicionados em coordenadas da figura (sem depender de `pad`).
    """
    pos = class_names.index(CLASSE_POSITIVA)
    neg = 1 - pos

    tp = cm[pos, pos]
    fn = cm[pos, neg]
    fp = cm[neg, pos]
    tn = cm[neg, neg]

    total_pos = tp + fn
    total_neg = fp + tn
    total = total_pos + total_neg
    m = calcular_metricas(tp, fn, fp, tn)

    Y_BASE = 2.0
    Y_TOPO = -0.66

    fig = plt.figure(figsize=(16, 8.8))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.15, 1], wspace=0.36,
        left=0.05, right=0.97, top=0.74, bottom=0.13,
    )
    ax_cm = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])

    # -- Painel esquerdo: matriz de confusão --
    celulas = {
        (0, 0): ('Verdadeiro Positivo (VP)', COR_ACERTO, tp, total_pos),
        (0, 1): ('Falso Negativo (FN)', COR_ERRO_CRITICO, fn, total_pos),
        (1, 0): ('Falso Positivo (FP)', COR_ERRO, fp, total_neg),
        (1, 1): ('Verdadeiro Negativo (VN)', COR_ACERTO, tn, total_neg),
    }

    ax_cm.set_xlim(0, 2)
    ax_cm.set_ylim(Y_BASE, Y_TOPO)
    for (i, j), (titulo, cor, valor, tot_linha) in celulas.items():
        ax_cm.add_patch(
            Rectangle((j, i), 1, 1, facecolor=cor, edgecolor='white', lw=6)
        )
        pct_linha = 100.0 * valor / tot_linha if tot_linha > 0 else 0.0
        cx, cy = j + 0.5, i + 0.5

        ax_cm.text(cx, cy - 0.34, titulo, ha='center', va='center',
                   fontsize=12.5, weight='bold', color=COR_TEXTO)
        ax_cm.text(cx, cy - 0.06, fmt_int(valor), ha='center', va='center',
                   fontsize=32, weight='bold', color=COR_TEXTO)
        ax_cm.text(cx, cy + 0.15, 'imagens', ha='center', va='center',
                   fontsize=11, color=COR_TEXTO)
        ax_cm.text(cx, cy + 0.34, f'({fmt_pct(pct_linha)} da classe real)',
                   ha='center', va='center', fontsize=10.5, weight='bold',
                   color='#333333')

    for j, (nome, tot) in enumerate([('Parasitada', total_pos),
                                     ('Não infectada', total_neg)]):
        ax_cm.text(j + 0.5, -0.31, nome, ha='center', va='center',
                   fontsize=14, weight='bold', color=COR_TEXTO)
        ax_cm.text(j + 0.5, -0.14, 'classe prevista',
                   ha='center', va='center', fontsize=11,
                   color=COR_TEXTO_SUAVE)

    # Rótulo do eixo + fio fino separando-o das colunas
    ax_cm.plot([0.04, 1.96], [-0.45, -0.45], color='#BDBDBD', lw=1.2,
               solid_capstyle='round', clip_on=False)
    ax_cm.text(1.0, -0.56, 'O QUE O MODELO PREVIU', ha='center', va='center',
               fontsize=11, weight='bold', color=COR_TEXTO_SUAVE)

    for i, nome in enumerate(('Parasitada', 'Não infectada')):
        ax_cm.text(-0.04, i + 0.5, f'Real: {nome}', rotation=90,
                   ha='right', va='center', fontsize=11)
    ax_cm.axis('off')

    # -- Painel direito: métricas com barras duplas --
    itens = [
        ('Acurácia', m['acuracia']),
        ('Sensibilidade', m['sensibilidade']),
        ('Especificidade', m['especificidade']),
        ('Precisão (positiva)', m['precisao']),
        ('F1 (positiva)', m['f1']),
    ]
    n_itens = len(itens)
    altura = 0.30
    ys = np.linspace(-0.58 + altura / 2, Y_BASE - altura / 2, n_itens)
    valores = [v for _, v in itens]

    ax_m.set_ylim(Y_BASE, Y_TOPO)
    ax_m.set_xlim(0, 117)

    # Barra de fundo até 100%
    ax_m.barh(ys, [100] * n_itens, color=COR_FUNDO_BARRA, height=altura,
              edgecolor='none')

    # Barra principal por cima
    for y, v, cor in zip(ys, valores, CORES_METRICAS):
        ax_m.barh(y, v, color=cor, height=altura)
        ax_m.text(103, y, fmt_pct(v), va='center', ha='left', fontsize=13,
                  weight='bold', color=COR_TEXTO)

    ax_m.set_yticks(ys)
    ax_m.set_yticklabels([n for n, _ in itens], fontsize=12.5, weight='bold')
    ax_m.set_xticks([0, 25, 50, 75, 100])
    ax_m.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax_m.tick_params(axis='y', length=0, pad=14)
    ax_m.tick_params(axis='x', length=4, pad=8, labelsize=11, color='#888888')

    for lado in ('top', 'right', 'left'):
        ax_m.spines[lado].set_visible(False)
    ax_m.spines['bottom'].set_bounds(0, 100)
    ax_m.spines['bottom'].set_position(('outward', 16))
    ax_m.spines['bottom'].set_color('#888888')

    p_cm = ax_cm.get_position()
    p_m = ax_m.get_position()
    y_secao = p_cm.y1 + 0.055
    fig.text((p_cm.x0 + p_cm.x1) / 2, y_secao, 'Matriz de confusão',
             ha='center', va='bottom', fontsize=15, weight='bold',
             color=COR_TEXTO)
    fig.text((p_m.x0 + p_m.x1) / 2, y_secao, 'Métricas de desempenho',
             ha='center', va='bottom', fontsize=15, weight='bold',
             color=COR_TEXTO)

    fig.text(0.5, 0.965,
             f'Avaliação do modelo {nome_modelo} — detecção de malária em '
             'células sanguíneas',
             ha='center', va='top', fontsize=18, weight='bold',
             color=COR_TEXTO)
    fig.text(0.5, 0.915,
             f'({conjunto}: {fmt_int(total)} imagens; positivo: Parasitized)'
             + (' — PRELIMINAR' if conjunto == 'val' else ''),
             ha='center', va='top', fontsize=13, color=COR_TEXTO_SUAVE)

    fig.savefig(caminho_saida, dpi=300, bbox_inches='tight', pad_inches=0.4,
                facecolor='white')
    plt.close(fig)
    return m


# -- AVALIAÇÃO EM LOTE --
def main():
    import json
    from sklearn.metrics import classification_report, confusion_matrix
    args = argumentos_avaliacao()
    caminhos = [args.modelos_dir / arq for arq in MODELOS.values()]
    for caminho in caminhos:
        conferir_modelo(caminho, args.manifesto_sha256)
    conjunto = carregar_conjunto(args.dados, args.conjunto)
    arquivos = [os.path.relpath(p, args.dados) for p in conjunto.file_paths]
    for caminho in caminhos:
        nome = caminho.stem.replace('melhor_modelo_', '')
        y, pred, prob = predizer(caminho, conjunto)
        salvar_predicoes(args, nome, caminho, y, pred, prob, arquivos)
        print(classification_report(y, pred, labels=[0, 1], target_names=CLASSES,
                                    digits=4, zero_division=0))
        cm = confusion_matrix(y, pred, labels=[0, 1])
        (args.saida / f'matriz_confusao_{nome}.json').write_text(
            json.dumps(dict(classes=CLASSES, linhas='real', colunas='previsto',
                            conjunto=args.conjunto, matriz=cm.tolist()), indent=2),
            encoding='utf-8')
        gerar_grafico(cm, CLASSES, nome.upper(),
                      args.saida / f'matriz_confusao_{nome}.png', args.conjunto)
    if args.conjunto == 'val':
        print('PRELIMINAR: validação usada para selecionar pesos; não é teste.')


if __name__ == '__main__':
    main()
