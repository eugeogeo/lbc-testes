# Preparação da FMD

O script `fmd_data.py` lê imagens organizadas em uma pasta por classe, por
exemplo `FMD_dataset/image/fabric/`. Arquivos que não têm extensões de imagem
suportadas são ignorados. As classes e imagens são ordenadas para manter o
mapeamento de rótulos e as partições reproduzíveis.

Instale as dependências de `requirements.txt` no ambiente Python e execute:

```bash
python fmd_data.py
```

O caminho padrão é `FMD_dataset/image` ao lado do script. Para usar outra base
ou configurar os carregadores:

```bash
python fmd_data.py --data-dir /caminho/para/imagens --batch-size 32 --num-workers 0 --seed 42
```

As imagens são convertidas para RGB, redimensionadas para 224 × 224 e
normalizadas com a média e o desvio padrão da ImageNet. A validação usa
`StratifiedKFold` com 10 partições, embaralhamento e semente fixa; cada classe
deve ter pelo menos 10 imagens. O conjunto de treino é embaralhado e o de
validação não é. A mesma transformação é aplicada a ambos, sem aumento de dados.

O script `fmd_data.py` apenas prepara os `DataLoader`s e informa os tamanhos
de cada fold.

## Bases e protocolos de validação (`datasets_loader.py`)

`get_dataset_and_loaders(dataset_name, data_dir, batch_size, num_folds, seed)`
retorna `(dataset, num_classes, fold_loaders)` para `'fmd'`, `'dtd'` ou `'kth'`.
Cada base usa o seu protocolo; `num_folds` pode ser omitido e, se informado,
deve coincidir com o da base:

| Base | Protocolo | Folds | Diretório padrão |
|---|---|---|---|
| FMD (10 classes) | `StratifiedKFold` aleatório, semente 42 | 10 | `FMD_dataset/image` |
| DTD (47 classes) | splits oficiais `labels/{train,val,test}N.txt`; treino = train + val, validação = test | 10 | `dataset_texture/test/DTD/dtd/images` |
| KTH-TIPS2b (11 classes) | uma amostra física (a, b, c ou d) valida e as outras três treinam | 4 | `dataset_texture/test/KTH-TIPS2b/KTH-TIPS2-b/images` |

- **DTD:** os arquivos `.txt` são lidos de `labels/`, ao lado de `images/` (ou de
  `labels_dir`). Eles fazem parte do pacote oficial `dtd-r1.0.1`, mas **não estão
  em `dataset_texture/`**; sem eles o carregador falha com uma mensagem explícita.
  Cada split é checado: sem repetições, sem sobreposição entre treino e teste e
  cobrindo as 5640 imagens.
- **KTH-TIPS2b:** a amostra vem da letra no prefixo do arquivo (`16a-...`).
  Cada fold de validação tem as 11 classes (1188 imagens) e nunca compartilha
  amostra física com o treino.
- Também é conferido o número de classes e de imagens de cada base.

Os quatro scripts de treino aceitam `--dataset {fmd,dtd,kth}` (padrão `fmd`) e
incluem a base nos nomes de saída, por exemplo
`results/integrated_scratch_dtd.json` e `saved_models/integrated_scratch_dtd/`:

| Script | JSON | Checkpoints |
|---|---|---|
| `resnet_baseline.py` | `results/resnet18_<base>.json` | — |
| `train_resnet_scratch_cv.py` | `results/resnet_scratch_<base>.json` | `saved_models/resnet_scratch_<base>/` |
| `train_integrated_cv.py` | `results/integrated_<base>.json` | `saved_models/integrated_<base>/` |
| `train_integrated_scratch_cv.py` | `results/integrated_scratch_<base>.json` (+ `_curves.png`) | `saved_models/integrated_scratch_<base>/` |

O JSON registra `dataset` e `num_folds`. `plot_filters.py` e `plot_confusion.py`
aceitam `--dataset` (usam `results/integrated_scratch_<base>.json` e sufixam as
imagens) e reconstroem os mesmos folds a partir do JSON. Os resultados já
gerados da FMD continuam nos nomes antigos e podem ser lidos sem `--dataset`.
O `handcrafted_baseline.py` ainda usa somente a FMD.

## Baseline handcrafted

```bash
python handcrafted_baseline.py
```

São aceitos `--data-dir`, `--seed` e `--output`; por padrão os resultados são
gravados em `results/handcrafted.json`. O JSON contém as métricas por fold,
médias, desvios padrão, semente, número de imagens e mapeamento de classes.

As mesmas partições estratificadas são usadas para o baseline. As imagens RGB
são redimensionadas para 224 × 224 e convertidas para tons de cinza; a
normalização ImageNet não é aplicada aos descritores. Para cada imagem:

- LBP uniforme, com 8 pontos e raio 1: histograma normalizado de 10 posições.
- GLCM simétrica normalizada, com 32 níveis de cinza, distância 1 e ângulos
  0°, 45°, 90° e 135°: média por ângulo de contraste, energia, homogeneidade
  e correlação.

Os 14 atributos são concatenados. O `StandardScaler` é ajustado somente no
treino de cada fold, seguido de um SVM linear (`SVC`, C=1). Os descritores
individuais são reutilizados entre folds, pois não são ajustados aos dados.
São reportadas a acurácia e o F1 **macro** (incluindo todas as classes, com
zero para classes sem predições), cada uma na escala de 0 a 1. O desvio padrão
é populacional (`ddof=0`) sobre os 10 folds. Não há ajuste de hiperparâmetros.

## Baseline ResNet-18

```bash
python resnet_baseline.py
```

O script usa os mesmos folds do baseline handcrafted e as transformações
ImageNet descritas acima. Cada fold começa com uma ResNet-18 nova, carregada
com `ResNet18_Weights.IMAGENET1K_V1` (baixados pelo torchvision se ausentes).
A camada `fc` é substituída por uma camada com uma saída por classe. Todos os
parâmetros são treináveis e um novo otimizador Adam é criado para cada fold.

Por padrão são usadas 10 épocas, batch de 32, taxa de aprendizado 0,0001,
semente 42 e GPU CUDA quando disponível. Esses parâmetros podem ser alterados
por `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--num-workers` e
`--device`; `--data-dir` altera a pasta de imagens. A semente também determina
as partições e a inicialização de cada fold. A reprodução exata entre hardware
ou versões diferentes não é garantida.

A rede retorna logits: `CrossEntropyLoss` aplica internamente log-softmax e
perda de log-verossimilhança; não se adiciona uma camada softmax antes da
perda. Na validação, `argmax` dos logits equivale ao `argmax` do softmax. A
avaliação ocorre uma única vez por fold, após a última época, em modo de
inferência; não é usada para escolher épocas ou hiperparâmetros.

Acurácia e F1 macro são calculados como no baseline handcrafted, assim como
suas médias e desvios padrão populacionais. Os resultados e parâmetros são
salvos em `results/resnet18.json`, ou no caminho indicado por `--output`.
O arquivo é gravado após a conclusão de todos os folds. O script não salva
pesos nem faz ajuste de hiperparâmetros.

## ResNet-18 from scratch

```bash
python train_resnet_scratch_cv.py --num-workers 2
```

O script compartilha o loop e as transformações do baseline ResNet; a única
alteração na arquitetura é usar `resnet18(weights=None)`, sem pesos ImageNet,
e substituir `fc` para o número de classes. São usados os inicializadores
padrão do torchvision: convoluções e `fc` aleatórias, enquanto parâmetros e
estatísticas de BatchNorm começam nos valores padrão, não aleatórios.
Apesar da inicialização sem pré-treino, preserva-se a normalização ImageNet
das imagens para comparação com os outros experimentos.

Cada fold recebe um novo modelo e um novo Adam; a avaliação ocorre somente
após a última época. Os padrões são os mesmos do baseline: 10 folds, 10
épocas, batch 32, taxa 0,0001 e semente 42. Os argumentos `--data-dir`,
`--seed`, `--epochs`, `--batch-size`, `--learning-rate`, `--num-workers` e
`--device` permitem alterar a execução; mantenha-os iguais ao baseline para
comparar. As métricas são gravadas em `results/resnet_scratch.json` e os
pesos finais em `saved_models/resnet_scratch_fold_<fold>.pth`; `--output` e
`--model-dir` alteram os caminhos. Arquivos com os mesmos nomes são
sobrescritos; o JSON é escrito apenas depois de concluir todos os folds.

## Inicialização estrutural por texturas

`integrated_resnet.py` cria a mesma ResNet-18 pré-treinada do baseline,
substitui `fc` e injeta pesos determinísticos diretamente no `state_dict` da
`conv1`, mantendo-os treináveis:

```python
from integrated_resnet import create_integrated_model
import torch

model = create_integrated_model(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```

Crie uma nova instância e um novo otimizador **dentro de cada fold**, antes do
treinamento. O script de demonstração `python integrated_resnet.py` apenas
cria e inspeciona o modelo; ele não executa a validação cruzada nem altera o
baseline existente.

Os pesos têm formato `(64, 3, 7, 7)`:

- 32 filtros Gabor reais de fase zero, com orientações `kπ/8`, `k=0,...,7`,
  comprimentos de onda 2, 3, 4 e 6 pixels, sigma 2 e razão espacial 0,5.
  Cada filtro tem média espacial removida e norma L2 espacial unitária.
- 32 filtros de diferença vizinho-centro inspirados no LBP: oito vizinhos,
  deslocamentos de 1 e 3 pixels e duas polaridades, com norma L2 espacial
  unitária. Eles não calculam o LBP completo: não há limiar binário, codificação
  nem histograma. A segunda polaridade preserva diferenças de ambos os sinais.

Cada filtro espacial é repetido nos três canais RGB com peso dividido por 3.
Isto combina os canais de entrada igualmente, **após** a normalização ImageNet,
e não equivale à conversão convencional para tons de cinza. A escala e a
normalização dos pesos são escolhas de inicialização para experimentação, não
garantias de equivalência à distribuição de ativações da `conv1` pré-treinada.
Os demais pesos e estatísticas pré-treinadas não são alterados; a camada `fc`
é nova. Após a injeção, a retropropagação pode modificar livremente os filtros.

## Experimento integrado em 10 folds

```bash
python train_integrated_cv.py --num-workers 2
```

O experimento reutiliza o loop, as transformações e as partições estratificadas
do baseline ResNet. A cada fold, cria uma nova rede integrada com pesos
ImageNet e filtros Gabor/LBP na `conv1`, além de um novo Adam sem estado
anterior. Todos os parâmetros são treináveis. Após um número fixo de épocas,
o fold externo é avaliado **uma única vez**; acurácia e F1 macro vêm da mesma
avaliação. Não há seleção de época pelos dados de validação.

O padrão de 10 épocas, batch de 32, taxa 0,0001 e semente 42 corresponde ao
baseline ResNet. É possível mudar esses valores com `--epochs`, `--batch-size`,
`--learning-rate` e `--seed`, mas para comparação mantenha-os iguais entre
experimentos. `--data-dir`, `--num-workers` e `--device` também são aceitos.
O número de folds é fixado em 10.

Os dez `state_dict`s finais são salvos em
`saved_models/integrated_resnet_fold_<fold>.pth`; as métricas por fold, médias,
desvios padrão populacionais e configurações são salvos em
`results/integrated_resnet.json`. Use `--model-dir` e `--output` para escolher
outros caminhos; arquivos existentes com os mesmos nomes são sobrescritos. Se
o processo for interrompido, checkpoints de folds concluídos permanecem, mas
o JSON só é gravado após os dez folds. Esses arquivos não são versionados.
Para carregar um checkpoint, crie a arquitetura com
`create_integrated_model(num_classes=10)` e use `model.load_state_dict(
torch.load(caminho, map_location="cpu", weights_only=True))`; os filtros
aprendidos sobrescrevem os valores iniciais.

O JSON também inclui `history` por fold, com perda média e acurácia **de
treino** por época. A acurácia é calculada a partir dos logits de cada batch
antes da respetiva atualização, em modo de treino; não é uma avaliação única
do modelo final da época. As curvas dos dez folds e a respetiva média são
gravadas em PNG junto ao JSON, com sufixo `_curves.png`; use `--plot-output`
para alterar o caminho. A validação externa continua a ocorrer apenas no fim
do fold. Registos anteriores sem histórico não permitem reconstruir a
acurácia por época; é necessário executar novamente.

Para recriar um gráfico a partir de um JSON com histórico:

```bash
python plot_learning_curves.py results/integrated_resnet.json results/curves.png
```

### ResNet integrada from scratch

```bash
python train_integrated_scratch_cv.py --num-workers 2
```

Esta variante usa `resnet18(weights=None)` e injeta os mesmos 64 filtros na
`conv1` treinável. Os demais parâmetros usam a inicialização padrão do
torchvision: convoluções Kaiming, `fc` inicializada por `nn.Linear` e
BatchNorm com valores padrão. Nenhum peso ImageNet é carregado; a
normalização ImageNet das imagens é mantida para comparação com os baselines.

Ela usa o mesmo loop do experimento integrado: nova rede e novo Adam por fold,
avaliação externa somente após a última época, e histórico de perda e acurácia
de treino por época. Os padrões são 10 folds, 10 épocas, batch 32, taxa
0,0001 e semente 42. Os argumentos de configuração são os mesmos do
`train_integrated_cv.py`. Os resultados ficam em
`results/integrated_scratch_run.json`, os checkpoints em
`saved_models/integrated_scratch/`, e o gráfico em
`results/integrated_scratch_curves.png`. É possível alterar esses caminhos
com `--output`, `--model-dir` e `--plot-output`; arquivos existentes são
sobrescritos.

### Visualização dos filtros e matriz de confusão

```bash
python plot_filters.py
python plot_confusion.py
```

Os scripts escolhem o fold de maior acurácia no JSON da execução integrada
from scratch; em caso de empate, escolhem o fold de menor número. A figura
deste fold é uma análise **descritiva do melhor fold**, não uma estimativa
imparcial do desempenho geral: para isso, use as métricas dos 10 folds.

O primeiro script cria `results/filters_initial.png` e
`results/filters_trained.png`. As imagens preservam os três canais RGB e usam
a mesma escala simétrica para todos os filtros de ambos os estados: zero é
representado por 0,5 e os extremos por 0 e 1. Assim, cores indicam pesos por
canal, não cores reais das imagens. A primeira metade do grid corresponde a
Gabor e a segunda às diferenças inspiradas no LBP.

O segundo script reconstrói o split usando a semente e o tamanho de batch do
JSON, verifica o mapeamento e o número de imagens e compara a acurácia
recalculada com a registrada. A matriz `results/confusion_matrix.png` contém
contagens absolutas, com classe verdadeira nas linhas e predita nas colunas.
Os arquivos originais do dataset devem permanecer inalterados; contagem e
mapeamento iguais não comprovam que os arquivos sejam idênticos.

Ambos aceitam `--results` e `--model-dir`; o segundo aceita ainda `--data-dir`,
`--device`, `--num-workers` e `--output`. O primeiro aceita
`--initial-output` e `--trained-output`. As figuras são salvas a 300 dpi.

Uma diminuição da perda de treino, por si só, não identifica um mínimo local
nem demonstra que mais épocas melhorariam a generalização. Qualquer ajuste de
épocas ou BatchNorm deve ser decidido com validação interna, sem usar os folds
externos para seleção.

Para executar os testes:

```bash
python -m unittest discover -s tests
```
