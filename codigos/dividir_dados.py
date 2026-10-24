"""Redistribui as imagens em dados/train e dados/val (sem etapa de teste).

Estrutura final:
    dados/train/Parasitized  \\
    dados/train/Uninfected   /  ~70% do total
    dados/val/Parasitized    ->  15% do total
    dados/val/Uninfected     ->  15% do total  (mesma quantidade da outra classe)

Como funciona:
  1. Junta todas as imagens que hoje estão em train/ e val/ (por classe).
  2. Calcula N_VAL = round(15% x total de imagens) e sorteia exatamente N_VAL
     imagens de CADA classe para a validação (seed fixa -> reprodutível).
  3. O restante de cada classe vai para o treino.
  4. Move apenas os arquivos que não estão no lugar certo.

Uso (a partir da pasta dos códigos):
    python dividir_dados.py              # SIMULAÇÃO: só mostra o plano
    python dividir_dados.py --executar   # move os arquivos de verdade

O resultado é determinístico: rodar de novo (mesmo que as pastas já estejam
divididas) gera a mesma divisão e não move nada.
"""
import argparse
import os
import random
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(BASE_DIR, '../dados')

CLASSES = ['Parasitized', 'Uninfected']
FRACAO_VAL_POR_CLASSE = 0.15  # 15% do TOTAL de imagens para cada classe
SEED = 123
EXTENSOES = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}


def fmt_int(n):
    return f'{int(n):,}'.replace(',', '.')


def fmt_pct(v):
    return f'{v:.1f}%'.replace('.', ',')


def listar_imagens(dados, classe):
    """Retorna {nome_arquivo: caminho_atual} juntando train/ e val/ da classe."""
    encontradas = {}
    for conjunto in ('train', 'val'):
        pasta = os.path.join(dados, conjunto, classe)
        if not os.path.isdir(pasta):
            continue
        for nome in os.listdir(pasta):
            caminho = os.path.join(pasta, nome)
            if not os.path.isfile(caminho):
                continue
            if os.path.splitext(nome)[1].lower() not in EXTENSOES:
                continue  # ignora Thumbs.db, .csv etc.
            if nome in encontradas:
                sys.exit(
                    f"[ERRO] Arquivo duplicado na classe '{classe}': '{nome}'\n"
                    f'       {encontradas[nome]}\n       {caminho}\n'
                    '       Remova/renomeie um deles e rode novamente.'
                )
            encontradas[nome] = caminho
    return encontradas


def montar_plano(dados):
    """Decide o destino de cada imagem. Retorna (plano, n_val, total)."""
    pool = {c: listar_imagens(dados, c) for c in CLASSES}

    for c in CLASSES:
        if not pool[c]:
            sys.exit(
                f"[ERRO] Nenhuma imagem encontrada para a classe '{c}' em "
                f'{os.path.abspath(dados)} (train/{c} ou val/{c}).'
            )

    total = sum(len(p) for p in pool.values())
    n_val = int(round(FRACAO_VAL_POR_CLASSE * total))
    menor = min(len(p) for p in pool.values())
    if n_val >= menor:
        sys.exit(
            f'[ERRO] Cada classe precisa de {fmt_int(n_val)} imagens para a '
            f'validação, mas a menor classe tem só {fmt_int(menor)}.'
        )

    plano = {}  # classe -> {'val': {nome: destino}, 'train': {nome: destino}}
    for c in CLASSES:
        nomes = sorted(pool[c])  # ordem independente do estado atual das pastas
        random.Random(SEED).shuffle(nomes)
        plano[c] = {'val': nomes[:n_val], 'train': nomes[n_val:]}
    return pool, plano, n_val, total


def calcular_movimentos(dados, pool, plano):
    movimentos = []
    for c in CLASSES:
        for conjunto in ('val', 'train'):
            for nome in plano[c][conjunto]:
                destino = os.path.join(dados, conjunto, c, nome)
                origem = pool[c][nome]
                if os.path.abspath(origem) != os.path.abspath(destino):
                    movimentos.append((origem, destino))
    return movimentos


def mostrar_plano(pool, plano, n_val, total, n_mov):
    n_train = {c: len(plano[c]['train']) for c in CLASSES}
    total_train = sum(n_train.values())
    total_val = 2 * n_val

    print('=' * 66)
    print('PLANO DE DIVISÃO DOS DADOS')
    print('=' * 66)
    print(f'Total de imagens: {fmt_int(total)}')
    for c in CLASSES:
        print(f'  {c:<12} {fmt_int(len(pool[c])):>8}')
    print('-' * 66)
    print(f'{"":<14}{"Parasitized":>14}{"Uninfected":>14}{"Total":>10}{"% total":>10}')
    print(f'{"Treino":<14}{fmt_int(n_train[CLASSES[0]]):>14}'
          f'{fmt_int(n_train[CLASSES[1]]):>14}{fmt_int(total_train):>10}'
          f'{fmt_pct(100 * total_train / total):>10}')
    print(f'{"Validação":<14}{fmt_int(n_val):>14}{fmt_int(n_val):>14}'
          f'{fmt_int(total_val):>10}{fmt_pct(100 * total_val / total):>10}')
    print('-' * 66)
    print(f'Cada classe na validação = {fmt_pct(100 * n_val / total)} do total.')
    desbal = abs(n_train[CLASSES[0]] - n_train[CLASSES[1]])
    if desbal:
        print(f'Obs.: o treino fica com {fmt_int(desbal)} imagens de diferença '
              'entre as classes (reflete o desbalanceamento original).')
    print(f'Arquivos a mover: {fmt_int(n_mov)}')
    print('=' * 66)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--executar', action='store_true',
                    help='move os arquivos (sem esta opção, só simula)')
    ap.add_argument('--dados', default=PASTA_DADOS,
                    help='pasta "dados" (padrão: ../dados)')
    args = ap.parse_args()

    dados = args.dados
    if not os.path.isdir(dados):
        sys.exit(f'[ERRO] Pasta de dados não encontrada: {os.path.abspath(dados)}')

    pool, plano, n_val, total = montar_plano(dados)
    movimentos = calcular_movimentos(dados, pool, plano)
    mostrar_plano(pool, plano, n_val, total, len(movimentos))

    if not args.executar:
        print('\nSIMULAÇÃO: nada foi alterado.')
        print('Para aplicar a divisão: python dividir_dados.py --executar')
        print('(faça um backup da pasta dados antes, por segurança)')
        return

    if not movimentos:
        print('\nAs pastas já estão na divisão correta. Nada a fazer.')
        return

    print('\nMovendo arquivos...')
    for k, (origem, destino) in enumerate(movimentos, start=1):
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        if os.path.exists(destino):
            sys.exit(f'[ERRO] Destino já existe, abortando: {destino}')
        shutil.move(origem, destino)
        if k % 2000 == 0:
            print(f'  {fmt_int(k)} / {fmt_int(len(movimentos))}')

    # Verificação final contando o que realmente ficou em cada pasta
    print('\nVerificação final:')
    for conjunto in ('train', 'val'):
        partes = []
        for c in CLASSES:
            pasta = os.path.join(dados, conjunto, c)
            n = sum(1 for f in os.listdir(pasta)
                    if os.path.splitext(f)[1].lower() in EXTENSOES)
            partes.append(f'{c}: {fmt_int(n)}')
        print(f'  {conjunto:<6} ' + ' | '.join(partes))
    print('\nConcluído! Agora retreine os modelos com o treinamento.py.')


if __name__ == '__main__':
    main()