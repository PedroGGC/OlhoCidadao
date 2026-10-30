# Dados, metodologia e avaliação da IA

Protocolo 3, implementado em 27/10/2026 sobre a coleta de 22/10/2026. Todo o treino, seleção, calibração, classificação e avaliação são automáticos, sem revisão humana. A MLP e o Adam são implementados do zero em NumPy, sem frameworks de aprendizagem e sem nenhum modelo ou API de IA externo: tudo roda localmente em CPU.

## O que a exatidão significa

O sistema aprende um gabarito estatístico: valor da nota acima da média de sua categoria em 2025 mais 1,2 desvios padrão de 2025. Os arquivos públicos não contêm rótulos de fraude comprovada. Portanto, 99,54% de precisão significa concordância dos alertas com esse critério histórico, não 99,54% de certeza de corrupção.

A fórmula do alvo e seu limiar não entram na rede. Entram `log1p(valor)`, o log do valor padronizado com média/desvio logarítmicos históricos da categoria e categorias one-hot. Partido, UF, mês e identidade do parlamentar não entram na decisão. Existem 25 categorias conhecidas, uma posição desconhecida e duas entradas numéricas: **28 → 16 tanh → 2 Softmax**.

Como o gabarito é uma fronteira por categoria, a tarefa da rede é aprender 25 "degraus" a partir de exemplos, sem conhecer a fórmula. A regra aplicada diretamente acertaria 100% por definição; ela é o teto da tarefa, não um concorrente. Os erros restantes ficam concentrados perto da fronteira (tabela abaixo), que é exatamente onde uma função contínua aproxima um degrau.

## Separação dos dados

| Papel | Competência | Registros |
| --- | --- | ---: |
| Referências históricas | jan–dez/2025 | 193.477 |
| Fit | Subconjunto de 2025 | 25.278: todos os 12.639 positivos e o mesmo número de normais |
| Parada antecipada | jan–mar/2026 | 5.000 de 44.729, distribuição natural |
| Escolha de ativação/arquitetura | jan–mar/2026 | 44.729, distribuição natural |
| Calibração sigmoidal | abr–mai/2026 | 35.417, distribuição natural |
| Escolha de limiar | jun/2026 | 16.239, distribuição natural |
| Teste reservado | jul–out/2026 | 35.922, distribuição natural |

As referências e os alvos de todos os períodos são definidos usando somente 2025. A normalização numérica é ajustada nas notas de treino. A amostragem garante representação dos estratos existentes de classe, UF, Casa e categoria e usa semente 42.

## Treino e escolha da ativação

Adam com taxa 0,01, sem L2, até 6.000 épocas e parada por perda de validação com paciência 100; os melhores pesos são recuperados. Os ajustes do protocolo 3 foram decididos só com jan–jun/2026:

| Mudança | Efeito na validação jan–mar |
| --- | --- |
| Fit de 5.000 → 25.278 notas | erros caíram de 60 para 32 |
| L2 de 0,00001 → 0 | erros caíram de 32 para 24 (a penalidade suavizava o degrau) |

Quatro candidatas são treinadas com os mesmos dados e escolhidas pelo F1 em jan–mar no melhor limiar desse período. A sigmoide, ativação original da rede de referência da disciplina, é treinada só como comparação, pois a atividade exige trocar a ativação:

| Ativação | Neurônios | Melhor época | Erros jan–mar | F1 jan–mar |
| --- | ---: | ---: | ---: | ---: |
| **tanh (escolhida)** | 16 | 1.778 | 21 | 99,681% |
| sigmoide (referência) | 16 | 2.297 | 20 | 99,696% |
| leaky ReLU | 16 | 1.169 | 21 | 99,681% |
| tanh | 32 | 629 | 21 | 99,681% |

As ativações empatam em qualidade (diferença de um erro em 44.729 notas). A tanh converge em 23% menos épocas que a sigmoide e é centrada em zero, combinando com as entradas Z-Score; com 16 neurônios empata com 32, então a rede menor é mantida.

A calibração usa somente abr–mai, corrigindo a saída da rede balanceada para a população natural. Junho escolhe o limiar que maximiza F1 com recall mínimo de 90% nesse período. Limiar final: aproximadamente **38,24%**.

## Resultado no teste reservado

A referência é a arquitetura anterior de cinco entradas, retreinada nos mesmos registros históricos, com categorias ordenadas pela mediana histórica, mesma validação temporal e taxa efetiva original de gradiente. Sua saída segue o limiar original de 50%, sem calibração. A tabela também traz o modelo do protocolo 2 (fit de 5.000 notas, com L2). Todos enfrentam exatamente as mesmas 35.922 notas e o mesmo gabarito congelado.

| Métrica | Arquitetura anterior | Protocolo 2 | **Protocolo 3** |
| --- | ---: | ---: | ---: |
| Acurácia | 96,3810% | 99,8775% | **99,9638%** |
| Precisão | 75,9524% | 98,6343% | **99,5418%** |
| Recall | 73,3896% | 99,6933% | **99,9617%** |
| F1 | 74,6490% | 99,1609% | **99,7513%** |
| Average precision | 84,9909% | 99,8889% | **99,8897%** |
| Falsos positivos | 606 | 36 | **12** |
| Falsos negativos | 694 | 8 | **1** |
| Brier | 0,026903 | 0,001237 | **0,000634** |
| ECE, dez faixas | 2,9195 p.p. | 0,1248 p.p. | **0,1396 p.p.** |

Os erros no teste caíram de 44 para 13. A regra de sempre prever normal teria acurácia de aproximadamente 92,74%, com recall zero. O teste contém 2.608 positivos estatísticos. Os intervalos Wilson de 95% são aproximadamente 99,20–99,74% para precisão e 99,78–99,99% para recall; não ajustam dependência entre notas do mesmo parlamentar.

Onde estão os erros, pela distância relativa entre o valor da nota e o limiar da sua categoria:

| Distância ao limiar | Notas | Erros |
| --- | ---: | ---: |
| até 1% | 62 | 6 |
| 1–5% | 369 | 1 |
| 5–20% | 1.158 | 4 |
| acima de 20% | 34.333 | 2 |

O teste artificial aumentou 2.000 notas inicialmente normais para duas vezes o limiar histórico da respectiva categoria: 100% foram detectadas. Esse teste avalia sensibilidade a aumento de valor; seus exemplos não são notas reais de fraude e não entram no resultado principal.

Relatório reproduzível com métricas por UF, Casa e categoria, comparação de ativações, erros por distância ao limiar e faixas de calibração: `outputs/model_evaluation.json` (restaurado pelo snapshot).

**Transparência sobre o teste:** jul–out/2026 já tinha sido avaliado no protocolo 2. As mudanças do protocolo 3 foram escolhidas só com jan–jun e o teste foi lido uma vez, ao final, mas ele não é mais inédito para o projeto. Um teste totalmente independente exige meses posteriores a out/2026.

## Limites da avaliação automática

O teste mede generalização temporal do critério escolhido, cuja regra é simples e baseada em valor/categoria. Bom desempenho nesse teste não demonstra descoberta de irregularidades independentes desse critério. O sistema não resolve fracionamento de notas, preços locais, adequação do serviço, comprovação de entrega ou padrões relacionais entre fornecedores. Parlamentares podem aparecer em mais de um período; não é avaliação de pessoas inéditas.

Categorias inéditas usam a referência global histórica e uma entrada desconhecida explícita. Não há garantia de calibração para categorias ausentes do treino. As métricas detalhadas devem ser interpretadas com a contagem de positivos de cada grupo, especialmente grupos pequenos.

O limite histórico de 2025 permanece congelado. Repetir a mesma execução verifica reprodutibilidade, não cria um novo teste independente.

## Notas publicadas depois do treino (coleta de 28/10/2026)

O modelo do protocolo 3 foi congelado sobre a coleta de 22/10/2026. Em 28/10 as fontes oficiais foram baixadas de novo e o modelo foi aplicado **sem retreinar**: pesos, normalização, calibração, limiar e gabarito (limiares de 2025) ficaram iguais. `python scripts/evaluate_new_notes.py` reproduz o processo; com `--offline`, reaplica sobre os arquivos locais.

Cada nota recebe a coluna `coletaPosterior`: 0 = idêntica à da coleta do treino, 1 = nova, 2 = mesma nota republicada pela fonte com outra categoria ou valor.

| Grupo | Notas | Observação |
| --- | ---: | --- |
| Novas em categorias conhecidas | 2.115 | 2.008 da Câmara e 107 do Senado; 1.631 de ago–out/2026, lançamentos atrasados |
| Nova categoria da Câmara: `PASSAGEM AÉREA - SIGEPA` | 29.952 | Toda em jan–jul/2025; categoria que a rede nunca viu |
| Revisadas pela fonte | 2.018 | 2.017 notas do Senado de um mesmo fornecedor movidas de "consultorias" para "passagens aéreas"; 1 valor corrigido |
| Removidas | 1 | |

**Resultado nas 2.115 notas novas de categorias conhecidas** — dados que nenhuma etapa de treino, seleção, calibração ou limiar conheceu:

| Métrica | Valor |
| --- | ---: |
| Atípicas pelo gabarito | 202 |
| Verdadeiros positivos / negativos | 202 / 1.913 |
| Falsos positivos / negativos | **0 / 0** |
| Precisão, recall e F1 | 100% |

Zero erros em 2.115 notas é compatível com a taxa do teste (13 erros em 35.922, cerca de 0,04%): a expectativa seria menos de um erro. O intervalo de Wilson de 95% para o recall nos 202 positivos vai de cerca de 98,1% a 100%, então o resultado confirma o teste temporal, mas não prova exatidão perfeita.

**Categoria inédita (SIGEPA).** A rede recebe a posição "categoria desconhecida" e a referência global. Para comparar, o gabarito usa a mesma regra do projeto calculada só com essa categoria (limiar de R$ 2.138,11). Resultado: recall de 100% (3.505 de 3.505), mas **1.098 falsos positivos** (precisão de 76,15%). É o limite esperado de um modelo que aprendeu degraus por categoria: sem exemplos da categoria, ele não sabe onde fica o degrau dela. Essas notas ficam fora das métricas principais e aparecem como alertas no painel. Incluir a SIGEPA no treino exige um novo protocolo com período de teste novo.

**Teste original.** As 35.683 notas de jul–out que continuam idênticas mantêm o resultado do protocolo 3 (12 falsos positivos e 1 falso negativo). As outras 239 notas do teste foram revisadas pela fonte.

## Reproduzir e conferir

```sh
pnpm train
python -m unittest discover -s tests
python scripts/validate_model.py
pnpm test
python scripts/validate_dataset.py --check
pnpm build
```

`validate_model.py` reconcilia todas as 357.850 inferências, o teste original nas notas não revisadas e as notas novas com os pesos exportados. Os testes JavaScript comparam o mesmo contrato em notas, extremos e valores próximos à fronteira, com tolerância de 1e-10. Os nomes dos CSVs permanecem iguais. A base integral continua disponível no dashboard; apenas o fit usa a amostra balanceada de 25.278 notas de 2025.

Referências metodológicas: [separação temporal](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) e [calibração de probabilidades](https://scikit-learn.org/stable/modules/calibration.html). As implementações deste projeto usam NumPy, não scikit-learn.


## Fontes e cobertura dos dados

A coleta de despesas utilizada neste snapshot é de 28/10/2026; o modelo foi treinado com a de 22/10/2026 (325.784 notas). Inclui 318.211 lançamentos da Câmara e 39.639 do Senado: **357.850 despesas positivas**, **R$ 455.818.617,49** e 27 UFs. O ano de 2026 é parcial. Esses valores não são o saldo líquido após estornos; não há multiplicação ou reponderação de notas.

- Câmara: dumps oficiais [2025](https://www.camara.leg.br/cotas/Ano-2025.csv.zip) e [2026](https://www.camara.leg.br/cotas/Ano-2026.csv.zip), abertos em memória, UTF-8 e separador `;`.
- Senado: consolidados CEAPS [2025](https://www.senado.leg.br/transparencia/LAI/verba/despesa_ceaps_2025.csv) e [2026](https://www.senado.leg.br/transparencia/LAI/verba/despesa_ceaps_2026.csv), Latin-1 e preâmbulo de atualização ignorado.
- UF dos senadores: [cadastro da 57ª legislatura](https://legis.senado.leg.br/dadosabertos/senador/lista/legislatura/57), incluindo suplentes. O período inclui 90 pessoas nas 81 cadeiras. Não se atribui DF a nomes desconhecidos nem se deduz partido histórico pelo cadastro atual; o CSV CEAPS não fornece esse partido.
- Entram valores positivos identificados com UF válida. Lideranças da Câmara sem UF e estornos negativos ficam fora. Um documento pode ter lançamentos distintos, portanto o número da nota isolado não é usado para eliminar registros.
- CNPJ/CPF e número documental são texto. Datas inválidas publicadas são preservadas; a competência oficial de ano/mês determina as partições.

Maranhão: 4.745 notas/R$ 10.923.528,61 em 2025 e 2.208 notas/R$ 5.785.332,25 em 2026. O relatório `outputs/data_quality_report.json`, fornecido pelo snapshot, concilia valores em centavos e hashes das fontes, CSVs e payload.

## Obras e municípios

A [API ObrasGov](https://api-publica.obrasgov.gestao.gov.br/obras) fornece projetos e geometrias, cruzados por `id_projeto_investimento`. O cadastro de 22/10/2026 contém 154.737 projetos com UF válida, 154.306 com município identificado e 5.569 códigos municipais. Há 14.716 divergências sinalizadas entre a geometria e a menção explícita a município/UF no nome oficial.

Uma obra pode abranger várias cidades, mas seu valor não é multiplicado. Município ausente não é inventado. O filtro municipal seleciona obras, não determina o município beneficiário de uma nota parlamentar. Ano de cadastro e situação da obra são independentes da competência da despesa. Investimento previsto não é repasse ou pagamento realizado. Valores simbólicos até R$ 1.000 continuam visíveis, mas não entram na comparação orçamentária.

`python scripts/update_public_works.py --refresh` atualiza o cadastro preservando as inferências. Para visualizar uma geração de pesquisa use `pnpm dev:local`; para distribuí-la, publique um novo snapshot com `pnpm data:publish`.

## Custos comparativos e limitações financeiras

As referências ficam centralizadas em `src/public_costs.py`: UBS R$ 250 mil, climatização escolar R$ 130 mil, ambulância R$ 280 mil, pavimentação R$ 450 mil, aluno/mês R$ 480, consulta especializada R$ 120 e cesta assistencial/diária operacional R$ 350.

São cenários aproximados, não contratos ou tarifas nacionais uniformes. Referências: [FUNDEB/FNDE](https://www.gov.br/fnde/pt-br/acesso-a-informacao/acoes-e-programas/financiamento/fundeb/2025), [credenciamento CISCOPAR](https://portal.ciscopar.com.br/credenciamento/tabelas?tp_tabela=3), [ambulâncias de Curitiba a R$ 289 mil por veículo](https://www.curitiba.pr.gov.br/noticias/prefeito-eduardo-pimentel-entrega-20-ambulancias-para-renovacao-da-frota-do-samu/78124) e [pavimentação em São Cristóvão](https://www.saocristovao.se.gov.br/noticia/prefeitura-de-sao-cristovao-inaugura-pavimentacao-de-tres-ruas-no-povoado-tinhare). A cesta assistencial não é a cesta nacional DIEESE e não houve importação de contratos do Transferegov.

`excessoMensal` é o nome legado da diferença positiva entre uma nota sinalizada e sua referência histórica; não é gasto mensal recorrente. `impacto5Anos` repete esse excedente por 60 meses como hipótese comparativa, sem demonstrar prejuízo ou economia realizável.

## Contexto acadêmico

Atividade prática do 2º bimestre de Redes Neurais/IA, individual ou em dupla, com apresentação prevista para 30/10/2026. Os requisitos são rede do zero, dados brasileiros e adaptação de arquitetura/ativação. O projeto mantém álgebra NumPy, Softmax, entropia cruzada, `forward`, `loss`, `backpropagation` e `fit`, com ativação `tanh` e sua derivada analítica, comparada experimentalmente com sigmoide e leaky ReLU.

Materiais indicados: [conceitos básicos](https://www.youtube.com/watch?v=ebToEXQFCo4&list=PLQH6T1jnIb5J7vugBAauJsFU8Qgvuf-4X) e [rede neural do zero em Python](https://www.youtube.com/watch?v=OGcOH9damt0&list=PLQH6T1jnIb5I1s7chxtL24ZuX4g-PA7uK).
