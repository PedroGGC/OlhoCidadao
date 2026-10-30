# Dados, metodologia e avaliação da IA

Implementado em 26/10/2026. A coleta usada nesta execução é de 29/10/2026. Todo o treino, seleção, calibração, classificação e avaliação são automáticos, sem revisão humana. A MLP e o Adam são implementados do zero em NumPy, sem frameworks de aprendizagem.

## O que a exatidão significa

O sistema aprende um gabarito estatístico: valor da nota acima da média de sua categoria em 2025 mais 1,2 desvios padrão de 2025. Os arquivos públicos não contêm rótulos de fraude comprovada. Portanto, 99,06% de precisão significa concordância dos alertas com esse critério histórico, não 99,06% de certeza de corrupção.

A fórmula do alvo e seu limiar não entram na rede. Entram `log1p(valor)`, o log do valor padronizado com média/desvio logarítmicos históricos da categoria e categorias one-hot. Partido, UF, mês e identidade do parlamentar não entram na decisão. Existem 25 categorias conhecidas, uma posição desconhecida e duas entradas numéricas: **28 → 16 tanh → 2 Softmax**.

## Separação dos dados

| Papel | Competência | Registros |
| --- | --- | ---: |
| Referências históricas | jan–dez/2025 | 193.825 |
| Fit | Subconjunto de 2025 | 5.000, sem reposição, classes balanceadas |
| Parada e escolha de arquitetura | jan–mar/2026 | 5.000 de 44.783, distribuição natural |
| Calibração sigmoidal | abr–mai/2026 | 35.442, distribuição natural |
| Escolha de limiar | jun/2026 | 16.249, distribuição natural |
| Teste reservado | jul–out/2026 | 38.407, distribuição natural |

As referências e os alvos de todos os períodos são definidos usando somente 2025. A normalização numérica é ajustada nas 5.000 notas de treino. A amostragem garante representação dos estratos existentes de classe, UF, Casa e categoria e usa semente 42.

As candidatas têm 16 e 32 neurônios, Adam com taxa 0,01, L2 de 0,00001, até 1.200 épocas e parada por perda de validação com paciência 100. Os melhores pesos são recuperados. A escolha da arquitetura maximiza average precision em jan–mar; escolheu 16 neurônios, época 682. O teste final não é exibido durante o fit nem usado para escolher parâmetros.

A calibração usa somente abr–mai, corrigindo a saída da rede balanceada para a população natural. Junho escolhe o limiar que maximiza F1 com recall mínimo de 90% nesse período. Limiar final: aproximadamente **32,1843%**. A meta de recall na validação não garante o mesmo resultado em dados futuros.

## Resultado no teste reservado

A referência é a arquitetura anterior de cinco entradas, retreinada nos mesmos 5.000 registros históricos, com categorias ordenadas pela mediana histórica, mesma validação temporal e taxa efetiva original de gradiente. Sua saída segue o limiar original de 50%, sem calibração. As duas abordagens enfrentam exatamente as mesmas 38.407 notas e o mesmo gabarito congelado.

| Métrica | Arquitetura anterior | Modelo refinado |
| --- | ---: | ---: |
| Acurácia | 96,1908% | 99,8568% |
| Precisão | 74,6853% | 99,0605% |
| Recall | 74,2957% | 99,0261% |
| F1 | 74,4900% | 99,0433% |
| Average precision | 83,7815% | 99,8349% |
| Verdadeiros positivos | 2.136 | 2.847 |
| Verdadeiros negativos | 34.808 | 35.505 |
| Falsos positivos | 724 | 27 |
| Falsos negativos | 739 | 28 |
| Brier | 0,029045 | 0,001564 |
| ECE, dez faixas | 2,8572 p.p. | 0,1563 p.p. |

A redução de falsos positivos foi de 96,27%. A regra de sempre prever normal teria acurácia de aproximadamente 92,51%, com recall zero. O teste contém 2.875 positivos estatísticos. Os intervalos Wilson de 95% são aproximadamente 98,64–99,35% para precisão e 98,60–99,33% para recall; não ajustam dependência entre notas do mesmo parlamentar.

O teste artificial aumentou 2.000 notas inicialmente normais para duas vezes o limiar histórico da respectiva categoria: 100% foram detectadas. Esse teste avalia sensibilidade a aumento de valor; seus exemplos não são notas reais de fraude e não entram no resultado principal.

Relatório reproduzível com métricas por UF, Casa e categoria, contagens e faixas de calibração: `outputs/model_evaluation.json` (restaurado pelo snapshot). Nenhum resultado do teste foi usado para retreinar ou ajustar o modelo após sua avaliação.

## Limites da avaliação automática

O teste mede generalização temporal do critério escolhido, cuja regra é simples e baseada em valor/categoria. Bom desempenho nesse teste não demonstra descoberta de irregularidades independentes desse critério. O sistema não resolve fracionamento de notas, preços locais, adequação do serviço, comprovação de entrega ou padrões relacionais entre fornecedores. Parlamentares podem aparecer em mais de um período; não é avaliação de pessoas inéditas.

Categorias inéditas usam a referência global histórica e uma entrada desconhecida explícita. Não há garantia de calibração para categorias ausentes do treino. As métricas detalhadas devem ser interpretadas com a contagem de positivos de cada grupo, especialmente grupos pequenos.

O limite histórico de 2025 permanece congelado para esta experiência. Usar os resultados de julho–outubro para criar novas versões exige um novo período reservado para manter a avaliação futura independente. Repetir a mesma execução verifica reprodutibilidade, não cria um novo teste independente.

## Reproduzir e conferir

```sh
pnpm train
python -m unittest discover -s tests
python scripts/validate_model.py
pnpm test
python scripts/validate_dataset.py --check
pnpm build
```

`validate_model.py` reconcilia todas as 328.706 inferências e as métricas do teste com os pesos exportados. Os testes JavaScript comparam o mesmo contrato em notas, extremos e valores próximos à fronteira, com tolerância de 1e-10. Os nomes dos CSVs permanecem iguais. A base integral continua disponível no dashboard; apenas o fit é limitado a 5.000 notas.

Referências metodológicas: [separação temporal](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) e [calibração de probabilidades](https://scikit-learn.org/stable/modules/calibration.html). As implementações deste projeto usam NumPy, não scikit-learn.


## Fontes e cobertura dos dados

A coleta de despesas utilizada neste snapshot é de 29/10/2026. Inclui 289.067 lançamentos da Câmara e 39.639 do Senado: **328.706 despesas positivas**, **R$ 422.670.871,03** e 27 UFs. O ano de 2026 é parcial. Esses valores não são o saldo líquido após estornos; não há multiplicação ou reponderação de notas.

- Câmara: dumps oficiais [2025](https://www.camara.leg.br/cotas/Ano-2025.csv.zip) e [2026](https://www.camara.leg.br/cotas/Ano-2026.csv.zip), abertos em memória, UTF-8 e separador `;`.
- Senado: consolidados CEAPS [2025](https://www.senado.leg.br/transparencia/LAI/verba/despesa_ceaps_2025.csv) e [2026](https://www.senado.leg.br/transparencia/LAI/verba/despesa_ceaps_2026.csv), Latin-1 e preâmbulo de atualização ignorado.
- UF dos senadores: [cadastro da 57ª legislatura](https://legis.senado.leg.br/dadosabertos/senador/lista/legislatura/57), incluindo suplentes. O período inclui 90 pessoas nas 81 cadeiras. Não se atribui DF a nomes desconhecidos nem se deduz partido histórico pelo cadastro atual; o CSV CEAPS não fornece esse partido.
- Entram valores positivos identificados com UF válida. Lideranças da Câmara sem UF e estornos negativos ficam fora. Um documento pode ter lançamentos distintos, portanto o número da nota isolado não é usado para eliminar registros.
- CNPJ/CPF e número documental são texto. Datas inválidas publicadas são preservadas; a competência oficial de ano/mês determina as partições.

Maranhão: 3.785 notas/R$ 9.682.875,86 em 2025 e 2.220 notas/R$ 5.807.414,53 em 2026. O relatório `outputs/data_quality_report.json`, fornecido pelo snapshot, concilia valores em centavos e hashes das fontes, CSVs e payload.

## Obras e municípios

A [API ObrasGov](https://api-publica.obrasgov.gestao.gov.br/obras) fornece projetos e geometrias, cruzados por `id_projeto_investimento`. O cadastro de 22/10/2026 contém 154.737 projetos com UF válida, 154.306 com município identificado e 5.569 códigos municipais. Há 14.716 divergências sinalizadas entre a geometria e a menção explícita a município/UF no nome oficial.

Uma obra pode abranger várias cidades, mas seu valor não é multiplicado. Município ausente não é inventado. O filtro municipal seleciona obras, não determina o município beneficiário de uma nota parlamentar. Ano de cadastro e situação da obra são independentes da competência da despesa. Investimento previsto não é repasse ou pagamento realizado. Valores simbólicos até R$ 1.000 continuam visíveis, mas não entram na comparação orçamentária.

`python scripts/update_public_works.py --refresh` atualiza o cadastro preservando as inferências. Para visualizar uma geração de pesquisa use `pnpm dev:local`; para distribuí-la, publique um novo snapshot com `pnpm data:publish`.

## Custos comparativos e limitações financeiras

As referências ficam centralizadas em `src/public_costs.py`: UBS R$ 250 mil, climatização escolar R$ 130 mil, ambulância R$ 280 mil, pavimentação R$ 450 mil, aluno/mês R$ 480, consulta especializada R$ 120 e cesta assistencial/diária operacional R$ 350.

São cenários aproximados, não contratos ou tarifas nacionais uniformes. Referências: [FUNDEB/FNDE](https://www.gov.br/fnde/pt-br/acesso-a-informacao/acoes-e-programas/financiamento/fundeb/2025), [credenciamento CISCOPAR](https://portal.ciscopar.com.br/credenciamento/tabelas?tp_tabela=3), [ambulâncias de Curitiba a R$ 289 mil por veículo](https://www.curitiba.pr.gov.br/noticias/prefeito-eduardo-pimentel-entrega-20-ambulancias-para-renovacao-da-frota-do-samu/78124) e [pavimentação em São Cristóvão](https://www.saocristovao.se.gov.br/noticia/prefeitura-de-sao-cristovao-inaugura-pavimentacao-de-tres-ruas-no-povoado-tinhare). A cesta assistencial não é a cesta nacional DIEESE e não houve importação de contratos do Transferegov.

`excessoMensal` é o nome legado da diferença positiva entre uma nota sinalizada e sua referência histórica; não é gasto mensal recorrente. `impacto5Anos` repete esse excedente por 60 meses como hipótese comparativa, sem demonstrar prejuízo ou economia realizável.

## Contexto acadêmico

Atividade prática do 2º bimestre de Redes Neurais/IA, individual ou em dupla, com apresentação prevista para 30/10/2026. Os requisitos são rede do zero, dados brasileiros e adaptação de arquitetura/ativação. O projeto mantém álgebra NumPy, Softmax, entropia cruzada, `forward`, `loss`, `backpropagation` e `fit`, com ativação `tanh` e sua derivada analítica.

Materiais indicados: [conceitos básicos](https://www.youtube.com/watch?v=ebToEXQFCo4&list=PLQH6T1jnIb5J7vugBAauJsFU8Qgvuf-4X) e [rede neural do zero em Python](https://www.youtube.com/watch?v=OGcOH9damt0&list=PLQH6T1jnIb5I1s7chxtL24ZuX4g-PA7uK).
