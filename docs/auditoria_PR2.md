# Auditoria técnica da PR #2 — validade da comparação de CNNs

Data: 28/10/2026. Repositório: `alecarvalho950/TCC`.
PR: https://github.com/alecarvalho950/TCC/pull/2.
Ramo examinado: `angelakretschmann/TCC-eng-software:atualiza-readme-resultados`.
Commit auditado: `8d9ae604d5c6c0d0e4747b234ae54fa6c3203141`.
Base da PR: `3f7108339b4175bec4fc99a372f13db32a4db70b`.

## Parecer

A ordenação histórica é uma descrição de uma execução em validação. Não
sustenta superioridade geral de uma arquitetura, conclusão clínica ou inferioridade
intrínseca da ResNet50. Há incompatibilidade comprovada no pré-processamento de
VGG16/ResNet50, reutilização da validação para seleção e comparação, e ausência
de artefatos suficientes para recalcular as métricas históricas.

Recomenda-se integrar as correções de implementação e as ressalvas, mantendo os
números antigos como históricos. A conclusão experimental permanece pendente de
novos treinamentos e avaliação reservada. Mergeabilidade do GitHub não é evidência
de validade metodológica. Esta auditoria não mescla a PR.

## Escopo e evidência disponível

Foram lidos os quatro scripts completos, README, requirements, diff da PR e o
texto do artigo PDF. A PR original altera apenas `.gitignore`, README,
requirements e PDF: os problemas de treinamento/avaliação são herdados da base,
mas afetam a interpretação dos resultados que a PR passa a documentar.

O ramo contém gráficos, mas não o dataset, pesos `.keras`, históricos numéricos,
CSV comparativo ou predições históricas. Os percentuais abaixo são **transcritos
como resultados informados**, não reproduzidos nem certificados por esta auditoria.
Não foram executados treinamentos experimentais com as imagens de malária.

## Achados comprováveis e correções

| Gravidade | Evidência no commit auditado | Implicação | Correção |
|---|---|---|---|
| Alta | `treinamento.py`: VGG16 e ResNet50 usam `data_augmentation` contendo `Rescaling(1/255)` | As bases ImageNet recebem RGB 0–1, sem BGR e centralização esperados | Augmentation em RGB 0–255; camada serializável chama o `preprocess_input` específico de cada base |
| Alta | `EarlyStopping` e `ModelCheckpoint` monitoram `val_loss`; os dois avaliadores leem `dados/val` | Seleção de pesos e comparação não são independentes; números são de validação | Nova divisão train/val/test registrada; avaliadores usam teste por padrão e recusam pesos de outra divisão |
| Média | Só há semente no embaralhamento e na divisão; augmentation é compartilhado entre CNN/VGG/ResNet | Não controla inicialização nem reinicia o estado aleatório por arquitetura | Semente global e augmentation novos por modelo; dataset recriado; operações determinísticas; versões/configuração registradas |
| Média | `comparar_modelos.py` pula arquivos ausentes e captura qualquer erro para continuar | Pode publicar ranking incompleto como comparação final | Exige os quatro modelos, metadados compatíveis e conclusão sem falhas silenciosas |
| Média | Gráfico comparativo declara a primeira colocada mais efetiva; critério é F1 ponderado | Título extrapola uma ordenação observada, sem inferência estatística | Título descritivo, conjunto explícito e ressalva; gráfico histórico preservado com aviso |
| Média | Matriz usa linhas reais e colunas previstas, mas mostra totais reais nos cabeçalhos das colunas | Visual pode atribuir suporte real ao eixo previsto | Eixos e rótulos explícitos; matriz bruta exportada em JSON |
| Média | Sem manifesto, vínculo de pesos com divisão ou versões históricas | Divisão/artefatos podem ser alterados sem rastreio; reprodução histórica incompleta | Hashes de imagens, manifesto, código e pesos; classes fixas, metadados e históricos numéricos |

### Pré-processamentos

| Arquitetura | Implementação original | Exigência/avaliação |
|---|---|---|
| CNN Própria | RGB / 255 | Coerente com treinamento próprio; preservado |
| VGG16 | RGB / 255 | Incompatível com ImageNet: esperado BGR menos `[103.939,116.779,123.68]`, sem /255 |
| ResNet50, versão original (não V2) | RGB / 255 | Mesma incompatibilidade; não confundir com ResNetV2, que usa outra transformação |
| EfficientNetB0 | RGB 0–255 e normalização interna | Coerente; não acrescentar /255 externo |

O erro é constatável sem retreinar. **Seu impacto quantitativo não é.** Não se
pode afirmar que ele explica toda a diferença da ResNet50, nem prever sua acurácia
corrigida. Alterar apenas o pré-processamento na inferência de pesos já treinados
com entradas diferentes não reproduz uma rodada corrigida: é necessário retreinar.

O aumento de dados permanece somente no treino. Na inferência, aplica-se o mesmo
pré-processamento determinístico embutido no modelo salvo; nenhum avaliador
acrescenta outra normalização. O teste funcional de pixels `[10,20,30]` confirma
saída BGR centralizada `[-73.939,-96.779,-113.68]` em VGG16/ResNet50.

### Separação dos dados

O código antigo calcula `round(0.15 * total)` imagens **para cada classe** na
validação. Para 27.558 imagens balanceadas: 4.134 por classe, 8.268 na validação
(30% do total) e 19.290 no treino. Não era uma divisão com 15% de validação.
Em dados desbalanceados, essa fórmula fixa igual número de validação por classe,
não a mesma fração de cada classe. O novo modo por imagem é estratificado.

O código antigo só evita repetição do mesmo nome na mesma classe entre train/val;
não verifica conteúdo duplicado com nomes diferentes, nem separa pacientes.
**Não foi demonstrado vazamento concreto de imagens ou pacientes**: os dados e
metadados não estão disponíveis. O risco é real e é reconhecido no próprio artigo.

O novo divisor copia de uma origem única para pasta nova, recusa SHA-256 repetido
e registra cada imagem. Com CSV `arquivo,grupo`, mantém todos os arquivos de
cada grupo numa única partição. Use paciente como unidade quando disponível;
lâminas do mesmo paciente não devem ter grupos distintos. Proporções são
aproximadas com grupos; ambas as classes são verificadas em cada partição.

Não há um mapa de pacientes no repositório. Sem ele, o modo `--por-imagem` precisa
ser escolhido explicitamente e permanece exploratório. Hash de bytes não detecta
imagens quase duplicadas, recompostas ou recomprimidas; a revisão da origem e da
unidade de aquisição continua necessária.

A nova partição de teste será reservada **aos pesos retreinados**. Como o corpus
já foi explorado pelos autores, ela não transforma a pesquisa anterior em estudo
prospectivo, nem garante independência clínica. Confirmação externa exige novos
pacientes/fontes. Pesos antigos não devem ser avaliados contra essa redistribuição.

### Critérios de avaliação

- Seleção da época: menor entropia cruzada em validação (`val_loss`), paciência 5,
  até 20 épocas. Isso está correto como seleção; não equivale a maximizar F1.
- Decisão: `argmax` de duas probabilidades softmax; sem limiar ajustado no teste.
- Comparação original: F1 ponderado. O README informa corretamente que precisão,
  recall e F1 da tabela são ponderados pelo suporte das duas classes.
- `Parasitized` é a positiva, índice 0 na ordem alfabética. O código original
  calcula sensibilidade/especificidade a partir desse índice corretamente.
- Recall ponderado coincide com acurácia em classificação de rótulo único;
  não é uma segunda confirmação independente do resultado.
- As barras da matriz mostram precisão e F1 da positiva, enquanto a tabela usa
  médias ponderadas. São estimandos diferentes; agora ficam identificados.
- As métricas `tf.keras.metrics.Precision/Recall` do treino, sobre one-hot e
  softmax com limiar padrão, não correspondem automaticamente à sensibilidade
  de Parasitized ou às médias ponderadas. Foram retiradas do log de treino;
  a avaliação explícita em scikit-learn permanece.
- O novo CSV acrescenta F1 macro, VP/VN/FP/FN, sensibilidade, especificidade,
  ROC AUC e average precision de Parasitized. Nenhum desses números foi inferido
  a partir das acurácias históricas. Predições/probabilidades são exportadas.

Não há intervalos de confiança, teste pareado ou repetição histórica. Não é
possível inferir significância com os percentuais arredondados e sem predições.
A classe positiva e falsos negativos devem ser analisados antes de qualquer
interpretação de utilidade em triagem. Classificar recortes celulares não equivale
a diagnosticar pacientes em ambiente clínico.

## Consistência do README e resultados preservados

| Modelo | Acurácia histórica informada | Situação após a correção |
|---|---|---|
| CNN Própria | 95,82% | Sem novo resultado |
| EfficientNetB0 | 93,55% | Sem novo resultado |
| VGG16 | 93,01% | Pré-processamento corrigido; retreinamento pendente |
| ResNet50 | 63,93% | Pré-processamento corrigido; retreinamento pendente |

As demais métricas históricas do README foram mantidas. Não há inconsistência
aritmética demonstrada na igualdade entre acurácia e recall ponderado. Também não
há base para reconstituir as matrizes de confusão únicas a partir dos percentuais.

A expressão “mesmas condições” era excessivamente ampla: tamanho, lote,
otimizador e orçamento de épocas comuns não igualam representações aprendidas,
número de parâmetros treináveis e estratégias de treinamento. As três bases
pré-treinadas estão congeladas, enquanto a CNN própria é integralmente treinada.
Isso é uma comparação válida de configurações se o protocolo for declarado e
corrigido, mas não isola o efeito causal da arquitetura. A resolução de 128 × 128
é permitida nesse uso com `include_top=False`; avaliar outra resolução é opção
experimental, não correção obrigatória de uma entrada inválida.

Os gráficos já versionados permanecem registros históricos. Novas rodadas usam
pasta separada, evitando misturar resultados de modelos, sementes e divisões
diferentes. Arquivos ausentes da execução antiga devem ser recuperados da origem.

## Artigo em PDF

O PDF descreve corretamente a rodada original em §3.3 e reconhece em §4.3 a
reutilização da validação, o risco por paciente e a execução única. O binário
foi preservado como registro da versão submetida; não foi reescrito para aparentar
que os novos procedimentos já tinham sido executados. Esta nota acompanha o link
no README e deve orientar a revisão do texto-fonte antes de nova versão acadêmica.

Correções editoriais necessárias na próxima versão:

1. §4.3: substituir a atribuição de um efeito mais intenso do pré-processamento
   à ResNet50 por: “VGG16 e ResNet50 receberam uma transformação incompatível com
   a entrada esperada de seus pesos ImageNet. Essa diferença é uma hipótese para
   parte da perda de desempenho; o efeito diferencial entre as arquiteturas não
   foi isolado experimentalmente.”
2. §4.2 e considerações finais: descrever a vantagem como maior valor observado
   nessa configuração e execução, sem generalizar superioridade da CNN própria.
   Curvas próximas não excluem sobreajuste à validação nem vazamento por paciente.
3. Não apresentar diferenças entre percentuais de estudos com protocolos distintos
   como comparação controlada. Podem ser contextualização, com essa limitação.
4. Fine-tuning pode ser um braço adicional previamente planejado. Corrigir o
   pré-processamento é obrigatório neste protocolo com pesos ImageNet; fine-tuning
   não é requisito universal para uma comparação legítima de bases congeladas.
5. Quando houver novos resultados, separar a metodologia histórica da corrigida,
   informar a unidade de separação e contagens reais, e substituir figuras/tabelas
   apenas com artefatos efetivamente obtidos.

## Experimentos ainda necessários

1. Recuperar imagens originais, origem/pacientes, pesos, CSV e predições históricos.
   Fixar uma divisão e preservar o manifesto antes de treinar.
2. Para investigar causalidade, executar ablação VGG16/ResNet50 com e sem a
   transformação correta, mantendo divisão, cabeça, congelamento, resolução,
   sementes e orçamento iguais. Usar validação para desenvolver esse diagnóstico;
   não escolher a configuração final consultando repetidamente o teste.
3. Treinar novamente as quatro configurações no protocolo corrigido, repetindo
   sementes de treinamento predefinidas com a mesma divisão por grupos. Informar
   distribuição dos resultados e custo de treinamento/inferência.
4. Após fechar o protocolo, avaliar no teste reservado; guardar predições para
   comparação pareada. Intervalos devem respeitar a unidade independente
   (paciente/grupo), e não tratar células correlacionadas como amostras independentes.
5. Para alegações clínicas, confirmar em pacientes/fontes externos, incluindo
   sensibilidade, especificidade e análise dos erros. Nada disso foi realizado
   por esta correção de software.

## Validação da implementação

- Seis testes automatizados passam no ambiente Python 3.12, TensorFlow CPU 2.20.0,
  Keras 3.15.1, NumPy 2.3.5, pandas 2.2.3, matplotlib 3.10.8 e scikit-learn 1.8.0.
- As quatro arquiteturas foram construídas com `weights=None` nos testes, sem
  baixar ImageNet; inferência antes/depois de salvar e recarregar foi comparada.
- Testes cobrem transformação numérica, saída softmax, duplicatas, preservação
  da fonte, separação por grupos, adulteração do manifesto/dados e incompatibilidade
  dos pesos. Métricas são conferidas com uma matriz de respostas conhecida.
- O fluxo completo também foi executado com 40 imagens sintéticas aleatórias:
  divisão, quatro treinamentos de uma época com `weights=None`, checkpoints,
  metadados, recarga, matrizes e comparação. Figuras foram inspecionadas. Os
  artefatos sintéticos não foram incorporados às tabelas ou ao repositório.
- Esses testes e quaisquer imagens sintéticas de verificação não são resultados
  do estudo. Não confirmam a acurácia de modelos com pesos ImageNet treinados
  sobre o dataset de malária.

## Fontes técnicas

Consultadas em 28/10/2026:

- [VGG16: entrada e pré-processamento](https://keras.io/2/api/applications/vgg/).
- [ResNet original: entrada e pré-processamento](https://keras.io/2/api/applications/resnet/).
- [EfficientNetB0: normalização interna e entrada 0–255](https://keras.io/2/api/applications/efficientnet/).
- [EarlyStopping e restauração da melhor época](https://keras.io/api/callbacks/early_stopping/).
- [Separação de validação e teste](https://scikit-learn.org/stable/modules/cross_validation.html).
- [Métricas de classificação](https://scikit-learn.org/stable/modules/model_evaluation.html).

Código original verificável por SHA:
[treinamento](https://github.com/angelakretschmann/TCC-eng-software/blob/8d9ae604d5c6c0d0e4747b234ae54fa6c3203141/codigos/treinamento.py),
[divisão](https://github.com/angelakretschmann/TCC-eng-software/blob/8d9ae604d5c6c0d0e4747b234ae54fa6c3203141/codigos/dividir_dados.py),
[avaliação](https://github.com/angelakretschmann/TCC-eng-software/blob/8d9ae604d5c6c0d0e4747b234ae54fa6c3203141/codigos/avaliacao.py),
[comparação](https://github.com/angelakretschmann/TCC-eng-software/blob/8d9ae604d5c6c0d0e4747b234ae54fa6c3203141/codigos/comparar_modelos.py).
