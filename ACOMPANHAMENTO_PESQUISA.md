# Acompanhamento da pesquisa

## Resumo em poucas palavras

**Já foram feitos testes preliminares de métodos de referência na base FMD. O melhor resultado registrado foi 71,30% com ResNet-18 pré-treinada e SVM. Ainda falta implementar e testar o método híbrido, que é o foco da pesquisa.**

Este documento compara os objetivos formais da qualificação com o que está disponível no projeto. As ideias extras anotadas no texto da qualificação não são tratadas como requisitos obrigatórios aqui.

## O que já foi feito

- O código carrega imagens da base **FMD**, organizada em 10 classes.
- Foram implementados extratores de características para **ResNet-18 pré-treinada**, **ResNet-18 com pesos aleatórios**, **LBP**, **GLCM** e uma **rede rasa de três camadas com pesos aleatórios**.
- As características são classificadas por um **SVM linear**. O código padroniza as características antes da classificação.
- A avaliação usa **validação cruzada estratificada com 10 folds**, isto é, divide os dados em 10 partes e repete a avaliação de modo que cada parte seja usada para teste. A configuração fixa `random_state=42`.
- Foram salvos vetores de características e rótulos na pasta `cache_features/`, o que evita recalcular parte da extração.
- As médias e os desvios-padrão foram registrados em `resultados_experimentos.csv`.

### Resultados registrados na FMD

| Método | Acurácia média | Desvio-padrão |
|---|---:|---:|
| ResNet-18 pré-treinada + SVM | **71,30%** | 3,41% |
| Rede rasa aleatória + SVM | 38,70% | 3,32% |
| LBP + SVM | 37,10% | 3,65% |
| GLCM + SVM | 33,50% | 4,78% |
| ResNet-18 com pesos aleatórios + SVM | 27,20% | 2,04% |

O CSV tem 14 registros de experimento; alguns métodos aparecem repetidos. A tabela mostra os cinco métodos distintos, não uma média calculada a partir dessas repetições.

### Como interpretar esses números

- **A ResNet-18 pré-treinada é o melhor baseline registrado até agora.** “Baseline” significa um método de referência usado para comparação.
- A rede rasa foi usada para extrair características com pesos aleatórios. **Ela não foi treinada para aprender os filtros de textura da proposta.**
- Portanto, esses números ainda **não medem o desempenho do método híbrido proposto**, nem demonstram se a hipótese da pesquisa está correta.

## O que falta fazer

| Objetivo formal | Situação atual |
|---|---|
| Implementar convoluções híbridas que incorporem descritores locais e possam aprender durante o treinamento | **Pendente.** O código tem extratores LBP/GLCM separados, mas não uma camada híbrida aprendível. |
| Adaptar arquiteturas de referência, como ResNet ou VGG, com a abordagem híbrida | **Pendente.** A ResNet existente é usada como extrator de características; não foi adaptada com camadas híbridas. |
| Treinar e testar o método nas bases DTD, FMD e KTH-TIPS2b | **Parcial.** Há resultados salvos para FMD. DTD e KTH-TIPS2b ainda não aparecem nos artefatos verificados. |
| Comparar o método híbrido com baselines e resultados da literatura usando acurácia | **Parcial.** Há comparação preliminar entre baselines na FMD, mas falta avaliar o método híbrido e completar a comparação prevista. |
| Comparar custo computacional e quantidade de parâmetros treináveis | **Pendente.** Não foram encontrados registros dessas medições. |

## Checkpoints para acompanhar o trabalho

Os checkpoints abaixo estão em ordem prática. Marque cada um como concluído quando o critério de saída estiver atendido, e não apenas quando o código começar a ser escrito.

### Checkpoint 1 — Fechar e documentar o baseline FMD

- [ ] Confirmar que todos os métodos comparados usam a mesma divisão de folds e o mesmo classificador, ou documentar claramente qualquer exceção.
- [ ] Registrar uma tabela final com média, desvio-padrão e configuração de cada baseline.
- [ ] Guardar as configurações de pré-processamento e extração necessárias para reproduzir cada resultado.
- **Concluído quando:** os resultados da FMD puderem ser reproduzidos a partir de configurações registradas e comparados de forma consistente.

### Checkpoint 2 — Implementar o método híbrido

- [ ] Definir precisamente como o descritor local será incorporado à convolução e quais partes serão aprendíveis.
- [ ] Implementar a camada híbrida e testá-la isoladamente, verificando dimensões de entrada e saída e a passagem do gradiente.
- [ ] Integrar a camada a uma arquitetura escolhida, começando por uma alternativa pequena e controlável.
- **Concluído quando:** a rede executar uma etapa de treinamento e atualização dos parâmetros sem erros, e a implementação corresponder à descrição do método.

### Checkpoint 3 — Treinar e avaliar na FMD

- [ ] Treinar a arquitetura híbrida usando uma divisão de dados definida; evitar usar imagens de teste para ajustar o modelo.
- [ ] Avaliar com o protocolo de validação escolhido e registrar média, desvio-padrão e configurações.
- [ ] Comparar com os baselines FMD do Checkpoint 1 sob condições equivalentes.
- **Concluído quando:** houver resultado reproduzível do método híbrido na FMD e comparação justa com os baselines.

### Checkpoint 4 — Avaliar nas três bases da pesquisa

- [ ] Preparar e documentar o protocolo para FMD, DTD e KTH-TIPS2b, respeitando as divisões oficiais quando aplicável.
- [ ] Executar o método proposto e os baselines selecionados em cada base.
- [ ] Registrar resultados por base e esclarecer diferenças de protocolo que impeçam comparação direta.
- **Concluído quando:** as três bases previstas tiverem resultados documentados, ou qualquer impossibilidade estiver justificada e acordada.

### Checkpoint 5 — Medir eficiência e concluir a comparação

- [ ] Contabilizar parâmetros treináveis das arquiteturas comparadas.
- [ ] Medir custo computacional com um protocolo consistente, informando hardware e condições da medição.
- [ ] Comparar acurácia e eficiência com os baselines e com trabalhos da literatura cujos protocolos sejam comparáveis.
- **Concluído quando:** houver uma análise que responda não só “qual acerta mais?”, mas também “qual exige menos recursos?”.

### Checkpoint 6 — Consolidar a evidência e atualizar a qualificação

- [ ] Organizar tabelas, configurações e resultados para que outra pessoa consiga entender como foram obtidos.
- [ ] Atualizar as seções de método, bases de dados, resultados, discussão e conclusão com o que foi realmente executado.
- [ ] Separar resultados medidos, interpretações e limitações.
- **Concluído quando:** cada afirmação experimental do texto puder ser rastreada até um procedimento e um resultado registrados.

## Prioridade recomendada

**Próximo passo principal: definir e implementar a camada híbrida (Checkpoint 2).** Antes de executar mais variações de baselines, é importante tornar claro o método que será avaliado. Em paralelo, vale organizar as configurações e os resultados da FMD para que a comparação futura seja consistente.

## Limites desta avaliação

- Os resultados acima foram lidos do CSV do projeto; não foram recalculados durante esta revisão.
- A pasta `dataset_texture/` não pôde ser inspecionada por falta de permissão. Seu conteúdo não foi considerado como evidência de bases ou experimentos concluídos.
- O histórico Git disponível tem apenas o commit inicial observado, então não permite reconstituir as etapas anteriores do desenvolvimento.

## Arquivos de referência

- Código dos experimentos: [main.py](./main.py)
- Resultados registrados: [resultados_experimentos.csv](./resultados_experimentos.csv)
- Dependências declaradas: [requirements.txt](./requirements.txt)
