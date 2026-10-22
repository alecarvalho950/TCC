import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_IMAGENS = os.path.join(BASE_DIR, '../dados/train')

# -- CONFIGURAÇÕES --
LISTA_MODELOS_CAMINHOS = [
    os.path.join(BASE_DIR, 'melhor_modelo_cnn_propria.keras'),
    os.path.join(BASE_DIR, 'melhor_modelo_vgg16.keras'),
    os.path.join(BASE_DIR, 'melhor_modelo_resnet50.keras'),
    os.path.join(BASE_DIR, 'melhor_modelo_efficientnet.keras'),
]

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

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


def gerar_grafico(cm, class_names, nome_modelo, caminho_saida):
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
        ax_cm.text(j + 0.5, -0.14, f'(real: {fmt_int(tot)} imagens)',
                   ha='center', va='center', fontsize=11,
                   color=COR_TEXTO_SUAVE)

    # Rótulo do eixo + fio fino separando-o das colunas
    ax_cm.plot([0.04, 1.96], [-0.45, -0.45], color='#BDBDBD', lw=1.2,
               solid_capstyle='round', clip_on=False)
    ax_cm.text(1.0, -0.56, 'O QUE O MODELO PREVIU', ha='center', va='center',
               fontsize=11, weight='bold', color=COR_TEXTO_SUAVE)

    ax_cm.axis('off')

    # -- Painel direito: métricas com barras duplas --
    itens = [
        ('Acurácia', m['acuracia']),
        ('Sensibilidade', m['sensibilidade']),
        ('Especificidade', m['especificidade']),
        ('Precisão', m['precisao']),
        ('F1-Score', m['f1']),
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
             f'(conjunto de validação: {fmt_int(total)} imagens)',
             ha='center', va='top', fontsize=13, color=COR_TEXTO_SUAVE)

    fig.savefig(caminho_saida, dpi=300, bbox_inches='tight', pad_inches=0.4,
                facecolor='white')
    plt.close(fig)
    return m


# -- AVALIAÇÃO EM LOTE --
def main():
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import image_dataset_from_directory

    print('Carregando conjunto de validação padronizado...')
    val_gen = image_dataset_from_directory(
        PASTA_IMAGENS,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
    )
    class_names = val_gen.class_names
    print(f'Classes encontradas: {class_names}')

    for model_path in LISTA_MODELOS_CAMINHOS:
        if not os.path.exists(model_path):
            print(f"\n[AVISO] Arquivo '{model_path}' não encontrado. Pulando...")
            continue

        nome_modelo = (
            os.path.basename(model_path)
            .replace('melhor_modelo_', '')
            .replace('.keras', '')
            .replace('.h5', '')
            .upper()
        )
        print('\n' + '=' * 60)
        print(f'Avaliando modelo: [{nome_modelo}]')
        print('=' * 60)

        model = load_model(model_path)

        y_true, y_pred = [], []
        for images, labels in val_gen:
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        print(f'\n--- RELATÓRIO DE AVALIAÇÃO: {nome_modelo} ---')
        print(classification_report(y_true, y_pred, target_names=class_names,
                                    digits=4))

        cm = confusion_matrix(y_true, y_pred)
        print('--- MATRIZ DE CONFUSÃO BRUTA ---')
        print(cm)

        nome_img = f'matriz_confusao_{nome_modelo.lower()}.png'
        gerar_grafico(
            cm, class_names, nome_modelo, os.path.join(BASE_DIR, nome_img)
        )
        print(f'\nGráfico salvo em: {nome_img}')

    print('\n' + '=' * 60)
    print('TODAS AS AVALIAÇÕES FORAM CONCLUÍDAS COM SUCESSO!')
    print('=' * 60)


if __name__ == '__main__':
    main()