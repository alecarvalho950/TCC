# Detecção de malária em células sanguíneas usando CNNs

Projeto de Trabalho de Conclusão de Curso (TCC) para classificar imagens
microscópicas de células sanguíneas como parasitadas ou não parasitadas e
comparar o desempenho de diferentes arquiteturas de redes neurais
convolucionais.

## Visão Geral do Projeto

O projeto realiza uma análise comparativa de quatro arquiteturas:

- **CNN Própria** — arquitetura convolucional desenvolvida para este estudo.
- **VGG16** — arquitetura pré-treinada usada na classificação das imagens.
- **ResNet50** — arquitetura residual pré-treinada.
- **EfficientNet** — arquitetura eficiente pré-treinada.

Os scripts carregam as imagens de treino e separam automaticamente uma parte
para validação. O treinamento salva os modelos e gráficos de histórico; a
avaliação gera relatórios e matrizes de confusão; e a comparação consolida as
métricas em uma tabela e um gráfico de ranking.

## Obter o Repositório

Clone o projeto e entre na pasta criada:

```bash
git clone https://github.com/alecarvalho950/TCC.git

cd estudos-imagens-celulares
```

## Estrutura de Pastas Esperada

Clone o repositório e organize as imagens do dataset conforme a árvore abaixo.
As pastas `Parasitized` e `Uninfected` devem estar diretamente dentro de
`dados/train/`.

```text
estudos-imagens-celulares/
├── README.md
├── .gitignore
├── codigos/
│   ├── treinamento.py
│   ├── avaliacao.py
│   └── comparar_modelos.py
└── dados/
    └── train/
        ├── Parasitized/
        │   ├── imagem_1.png
        │   └── ...
        └── Uninfected/
            ├── imagem_1.png
            └── ...
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
utilizadas pelos scripts:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow keras matplotlib numpy pandas seaborn scikit-learn
```

## Passo a Passo para Execução

Execute os comandos a seguir, na ordem indicada, a partir da raiz do
repositório. Antes de iniciar, confirme que o ambiente virtual está ativado e
que o dataset está em `dados/train/`.

1. **Treinar os modelos em lote.** O script treina CNN Própria, VGG16,
   ResNet50 e EfficientNet. Os modelos `.keras` e gráficos de histórico são
   gravados em `codigos/`.

   ```bash
   python codigos/treinamento.py
   ```

2. **Avaliar os modelos salvos.** O script imprime os relatórios de avaliação
   e grava as matrizes de confusão em `codigos/`.

   ```bash
   python codigos/avaliacao.py
   ```

3. **Comparar os resultados.** O script gera a tabela consolidada
   `tabela_comparativa_modelos.csv` e o gráfico de ranking
   `grafico_comparativo_geral.png`, também em `codigos/`.

   ```bash
   python codigos/comparar_modelos.py
   ```

O treinamento de redes neurais pode levar algum tempo e requer espaço livre em
disco para os modelos e resultados. Os dados, modelos treinados e arquivos
gerados estão listados no `.gitignore` e não são enviados ao GitHub.

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
- **Artigo original:** [Acessar no PMLR](https://proceedings.mlr.press/v97/tan19a.html?ref=ji)

## Autores

Alejandro Carvalho, Ângela Kretschmann, Giliard Gomes Martins e Gian Brustolin.
