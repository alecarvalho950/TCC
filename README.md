# Detecção de malária em células sanguíneas usando CNNs

Projeto de Trabalho de Conclusão de Curso (TCC) para classificar imagens
microscópicas de células sanguíneas como parasitadas ou não parasitadas e
comparar o desempenho de diferentes arquiteturas de redes neurais
convolucionais.

**Artigo:** [Inteligência artificial na detecção de malária: estudo comparativo de redes neurais convolucionais em imagens de células sanguíneas](docs/artigo_TCC.pdf)

## Status do Projeto

Os quatro modelos foram treinados e avaliados sob as mesmas condições (até 20
épocas, com 70% das imagens para treinamento e 30% para validação). Os gráficos
de treinamento e o gráfico comparativo estão publicados na pasta `codigos/`, e
as métricas estão resumidas na seção [Resultados](#resultados).

Os resultados são preliminares, pois foram obtidos no conjunto de validação. A
próxima etapa prevê a avaliação em um conjunto de teste independente.

## Visão Geral do Projeto

O projeto realiza uma análise comparativa de quatro arquiteturas:

- **CNN Própria** — arquitetura convolucional desenvolvida para este estudo.
- **VGG16** — arquitetura pré-treinada usada na classificação das imagens.
- **ResNet50** — arquitetura residual pré-treinada.
- **EfficientNet** (EfficientNetB0) — arquitetura eficiente pré-treinada.

O fluxo tem quatro etapas, cada uma em um script da pasta `codigos/`:

1. `dividir_dados.py` separa as imagens em treinamento (`dados/train/`) e
   validação (`dados/val/`).
2. `treinamento.py` treina as quatro arquiteturas e salva os modelos e os
   gráficos de histórico.
3. `avaliacao.py` gera os relatórios e as matrizes de confusão.
4. `comparar_modelos.py` consolida as métricas em uma tabela e em um gráfico de
   ranking.

## Obter o Repositório

```bash
git clone https://github.com/alecarvalho950/TCC.git

cd TCC
```

## Estrutura de Pastas

Depois de baixar o dataset e executar o `dividir_dados.py`, a estrutura fica
assim:

```text
TCC/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── artigo_TCC.pdf
├── codigos/
│   ├── dividir_dados.py
│   ├── treinamento.py
│   ├── avaliacao.py
│   ├── comparar_modelos.py
│   └── grafico_*.png          (gráficos gerados)
└── dados/                     (não versionada)
    ├── train/
    │   ├── Parasitized/
    │   └── Uninfected/
    └── val/
        ├── Parasitized/
        └── Uninfected/
```

Os scripts localizam os dados com base no caminho dos próprios arquivos. Assim,
podem ser executados a partir da raiz do repositório ou de outro diretório,
desde que a estrutura acima seja mantida.

## Download do Dataset

As imagens não acompanham este repositório devido ao tamanho do dataset.
Baixe-as na página oficial do Kaggle:

[Cell Images for Detecting Malaria — Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data)

Depois de baixar e extrair o arquivo:

1. Localize as pastas de classes `Parasitized` e `Uninfected` dentro do dataset.
2. Copie **somente essas duas pastas** para `dados/train/`.
3. Confirme que os caminhos ficaram `dados/train/Parasitized/` e
   `dados/train/Uninfected/`, sem uma pasta intermediária extra.

O arquivo do Kaggle contém uma cópia duplicada do dataset (uma pasta
`cell_images` dentro de outra). Copie as classes de apenas uma delas, para não
duplicar as imagens.

No Windows PowerShell, a pasta de destino pode ser criada com:

```powershell
New-Item -ItemType Directory -Force dados/train
```

Em macOS ou Linux:

```bash
mkdir -p dados/train
```

## Configuração do Ambiente e Instalação

Abra um terminal na raiz do repositório. Recomenda-se usar uma versão do Python
compatível com a versão do TensorFlow disponível para o seu sistema operacional.

### Windows (PowerShell)

Crie e ative um ambiente virtual:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação de scripts, consulte as instruções de
execução de scripts da sua organização ou use o Prompt de Comando:

```bat
venv\Scripts\activate.bat
```

### macOS ou Linux

Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Instalar dependências

Com o ambiente virtual ativado, atualize o `pip` e instale as bibliotecas
listadas no `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Passo a Passo para Execução

Execute os comandos a seguir, na ordem indicada, a partir da raiz do
repositório. Antes de iniciar, confirme que o ambiente virtual está ativado e
que o dataset está em `dados/train/`.

1. **Dividir os dados.** Sem a opção `--executar`, o script apenas mostra o
   plano de divisão, sem mover arquivos. Com ela, move 15% do total de imagens
   de cada classe para `dados/val/` (8.268 imagens no total), mantendo as
   demais em `dados/train/` (19.290 imagens). A divisão usa semente fixa
   (`seed = 123`) e é reprodutível. Recomenda-se fazer uma cópia da pasta
   `dados/` antes.

   ```bash
   python codigos/dividir_dados.py
   python codigos/dividir_dados.py --executar
   ```

2. **Treinar os modelos em lote.** O script treina CNN Própria, VGG16,
   ResNet50 e EfficientNet por até 20 épocas, com parada antecipada. Os
   modelos `.keras` e os gráficos de histórico são gravados em `codigos/`.

   ```bash
   python codigos/treinamento.py
   ```

3. **Avaliar os modelos salvos.** O script imprime os relatórios de avaliação
   e grava as matrizes de confusão em `codigos/`.

   ```bash
   python codigos/avaliacao.py
   ```

4. **Comparar os resultados.** O script gera a tabela consolidada
   `tabela_comparativa_modelos.csv` e o gráfico de ranking
   `grafico_comparativo_geral.png`, também em `codigos/`.

   ```bash
   python codigos/comparar_modelos.py
   ```

O treinamento de redes neurais pode levar algum tempo e requer espaço livre em
disco. O dataset e os modelos treinados (`.keras`) estão listados no
`.gitignore` e não são enviados ao GitHub.

## Resultados

Métricas no conjunto de validação (8.268 imagens), calculadas pelo
`comparar_modelos.py`. Precisão, *recall* e *F1-score* correspondem à média
ponderada das duas classes.

| Modelo | Acurácia | Precisão | Recall | F1-score |
|---|---|---|---|---|
| CNN Própria | 95,82% | 95,84% | 95,82% | 95,81% |
| EfficientNetB0 | 93,55% | 93,58% | 93,55% | 93,55% |
| VGG16 | 93,01% | 93,18% | 93,01% | 93,00% |
| ResNet50 | 63,93% | 65,50% | 63,93% | 63,00% |

### Comparação geral

![Comparação geral dos modelos](codigos/grafico_comparativo_geral.png)

### Histórico de treinamento

| CNN Própria | EfficientNet |
|---|---|
| ![CNN Própria](codigos/grafico_cnn_propria.png) | ![EfficientNet](codigos/grafico_efficientnet.png) |
| **VGG16** | **ResNet50** |
| ![VGG16](codigos/grafico_vgg16.png) | ![ResNet50](codigos/grafico_resnet50.png) |

As matrizes de confusão (`matriz_confusao_*.png`) e a tabela
`tabela_comparativa_modelos.csv` desta rodada serão adicionadas ao repositório.

## Referências Bibliográficas

As arquiteturas utilizadas neste estudo são fundamentadas em trabalhos
acadêmicos que investigam profundidade, conexões residuais e escalabilidade
eficiente em redes neurais convolucionais.

### VGG16 — Very Deep Convolutional Networks for Large-Scale Image Recognition

Empilha convoluções pequenas de 3 × 3 para investigar como o aumento da
profundidade melhora o reconhecimento de imagens em larga escala.

- **Autores:** Karen Simonyan e Andrew Zisserman (2014)
- **Artigo original:** [Acessar no arXiv](https://arxiv.org/abs/1409.1556)

### ResNet50 — Deep Residual Learning for Image Recognition

Introduz conexões residuais, que facilitam o treinamento de redes profundas e
ajudam a lidar com a degradação da acurácia à medida que a profundidade aumenta.

- **Autores:** Kaiming He, Xiangyu Zhang, Shaoqing Ren e Jian Sun (CVPR 2016)
- **Artigo original:** [Acessar no CVF Open Access](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

### EfficientNet — Rethinking Model Scaling for Convolutional Neural Networks

Propõe o escalonamento composto, que ajusta de forma coordenada profundidade,
largura e resolução para equilibrar desempenho e custo computacional.

- **Autores:** Mingxing Tan e Quoc V. Le (ICML 2019)
- **Artigo original:** [Acessar no PMLR](https://proceedings.mlr.press/v97/tan19a.html)

## Autores

- Alejandro Carvalho — [@alecarvalho950](https://github.com/alecarvalho950)
- Ângela Kretschmann — [@angelakretschmann](https://github.com/angelakretschmann)
- Giliard Gomes Martins — [@Gili10](https://github.com/Gili10)

## Orientador

- Prof. Gian Brustolin
