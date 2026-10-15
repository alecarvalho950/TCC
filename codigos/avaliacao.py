import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_IMAGENS = os.path.join(BASE_DIR, '../dados/train')

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
LISTA_MODELOS_CAMINHOS = [
    os.path.join(BASE_DIR, 'melhor_modelo_cnn_propria.keras'),
    os.path.join(BASE_DIR, 'melhor_modelo_vgg16.keras'),
    os.path.join(BASE_DIR, 'melhor_modelo_resnet50.keras'),
    os.path.join(BASE_DIR, 'melhor_modelo_efficientnet.keras'),
]

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Classe considerada "positiva" (a que queremos detectar)
CLASSE_POSITIVA = 'Parasitized'

# Cores
COR_ACERTO = '#C8E6C9'
COR_ERRO = '#FFCDD2'
COR_ERRO_CRITICO = '#EF9A9A'
COR_TEXTO = '#1B1B1B'
COR_BARRA = '#1F5FA8'


# ==========================================
# 2. FUNÇÕES AUXILIARES DO GRÁFICO
# ==========================================
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
    """Gera figura com a matriz de confusão (esquerda) e métricas (direita)."""
    pos = class_names.index(CLASSE_POSITIVA)
    neg = 1 - pos

    tp = cm[pos, pos]  # parasitada  -> previu parasitada
    fn = cm[pos, neg]  # parasitada  -> previu não infectada
    fp = cm[neg, pos]  # não infect. -> previu parasitada
    tn = cm[neg, neg]  # não infect. -> previu não infectada

    total_pos = tp + fn  # total de células realmente parasitadas
    total_neg = fp + tn  # total de células realmente não infectadas
    total = total_pos + total_neg
    m = calcular_metricas(tp, fn, fp, tn)

    fig = plt.figure(figsize=(15.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.45)
    ax_cm = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])

    # ---------- Painel esquerdo: matriz de confusão ----------
    # (linha, coluna): (título, cor, valor, total da linha, descrição)
    celulas = {
        (0, 0): ('Verdadeiro Positivo (VP)', COR_ACERTO, tp, total_pos,
                 'Parasitada classificada\ncomo parasitada'),
        (0, 1): ('Falso Negativo (FN)', COR_ERRO_CRITICO, fn, total_pos,
                 'Parasitada classificada\ncomo não infectada  [erro crítico]'),
        (1, 0): ('Falso Positivo (FP)', COR_ERRO, fp, total_neg,
                 'Não infectada classificada\ncomo parasitada  [falso alarme]'),
        (1, 1): ('Verdadeiro Negativo (VN)', COR_ACERTO, tn, total_neg,
                 'Não infectada classificada\ncomo não infectada'),
    }
    nomes_linha = {
        0: f'Parasitada\n(real: {fmt_int(total_pos)} imagens)',
        1: f'Não infectada\n(real: {fmt_int(total_neg)} imagens)',
    }

    ax_cm.set_xlim(0, 2)
    ax_cm.set_ylim(2, 0)
    for (i, j), (titulo, cor, valor, tot_linha, desc) in celulas.items():
        ax_cm.add_patch(
            Rectangle((j, i), 1, 1, facecolor=cor, edgecolor='white', lw=5)
        )
        pct_linha = 100.0 * valor / tot_linha if tot_linha > 0 else 0.0
        cx, cy = j + 0.5, i + 0.5
        ax_cm.text(cx, cy - 0.30, titulo, ha='center', va='center',
                   fontsize=11.5, weight='bold', color=COR_TEXTO)
        ax_cm.text(cx, cy - 0.08, fmt_int(valor), ha='center', va='center',
                   fontsize=28, weight='bold', color=COR_TEXTO)
        ax_cm.text(cx, cy + 0.13, 'imagens', ha='center', va='center',
                   fontsize=10, color=COR_TEXTO)
        ax_cm.text(cx, cy + 0.26,
                   f'{fmt_pct(pct_linha)} das ' +
                   ('parasitadas' if i == 0 else 'não infectadas'),
                   ha='center', va='center', fontsize=10.5, weight='bold',
                   color=COR_TEXTO)
        ax_cm.text(cx, cy + 0.42, desc, ha='center', va='center',
                   fontsize=8.5, style='italic', color='#444444',
                   linespacing=1.15)

    ax_cm.set_xticks([0.5, 1.5])
    ax_cm.set_xticklabels(['Parasitada', 'Não infectada'], fontsize=11)
    ax_cm.set_yticks([0.5, 1.5])
    ax_cm.set_yticklabels([nomes_linha[0], nomes_linha[1]], fontsize=11)
    ax_cm.tick_params(length=0)
    ax_cm.xaxis.set_ticks_position('top')
    ax_cm.xaxis.set_label_position('top')
    ax_cm.set_xlabel('O QUE O MODELO PREVIU', fontsize=11, weight='bold',
                     labelpad=12)
    ax_cm.set_ylabel('O QUE A IMAGEM REALMENTE É', fontsize=11, weight='bold',
                     labelpad=12)
    for s in ax_cm.spines.values():
        s.set_visible(False)
    ax_cm.set_title('Matriz de confusão', fontsize=13, weight='bold', pad=75)

    # ---------- Painel direito: métricas ----------
    itens = [
        ('Acurácia', 'acertos sobre o total', m['acuracia']),
        ('Sensibilidade', 'parasitadas detectadas', m['sensibilidade']),
        ('Especificidade', 'não infectadas reconhecidas', m['especificidade']),
        ('Precisão', 'quando diz "parasitada", acerta', m['precisao']),
        ('F1-Score', 'equilíbrio precisão × sensibilidade', m['f1']),
    ]
    ys = np.arange(len(itens))[::-1]
    valores = [v for _, _, v in itens]
    ax_m.barh(ys, valores, color=COR_BARRA, height=0.55)
    for y, v in zip(ys, valores):
        ax_m.text(min(v + 1.5, 97), y, fmt_pct(v), va='center', ha='left',
                  fontsize=12, weight='bold', color=COR_TEXTO)
    ax_m.set_yticks(ys)
    ax_m.set_yticklabels([f'{n}\n({d})' for n, d, _ in itens], fontsize=10.5)
    ax_m.set_xlim(0, 112)
    ax_m.set_xticks([0, 25, 50, 75, 100])
    ax_m.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax_m.axvline(50, color='gray', ls='--', lw=1)
    ax_m.text(50, len(itens) - 0.45, 'chute aleatório (50%)', ha='center',
              va='bottom', fontsize=8.5, color='gray')
    ax_m.tick_params(length=0)
    for lado in ('top', 'right', 'left'):
        ax_m.spines[lado].set_visible(False)
    ax_m.set_title('Métricas de desempenho', fontsize=13, weight='bold',
                   pad=75)

    fig.suptitle(
        f'Avaliação do modelo {nome_modelo} — detecção de malária em células '
        f'sanguíneas\n(conjunto de validação: {fmt_int(total)} imagens)',
        fontsize=14, weight='bold', y=1.14,
    )

    fig.savefig(caminho_saida, dpi=300, bbox_inches='tight', pad_inches=0.3,
                facecolor='white')
    plt.close(fig)
    return m


# ==========================================
# 3. AVALIAÇÃO EM LOTE
# ==========================================
def main():
    # Imports pesados aqui dentro para permitir testar só o gráfico
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import image_dataset_from_directory

    print('Carregando conjunto de validação padronizado...')
    val_gen = image_dataset_from_directory(
        PASTA_IMAGENS,
        validation_split=0.2,
        subset='validation',
        seed=123,  # Mesma semente do treinamento
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,  # Necessário para o split correto (ver y_true abaixo)
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

        # CORREÇÃO: y_true e y_pred são coletados NO MESMO loop.
        # Como val_gen tem shuffle=True, percorrê-lo duas vezes geraria
        # duas ordens diferentes e os rótulos não bateriam com as predições.
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