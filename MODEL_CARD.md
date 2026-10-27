# Modelo autônomo de atipicidade — protocolo 2

Implementado em 26/10/2026. A coleta usada nesta execução é de 22/10/2026. Todo o treino, seleção, calibração, classificação e avaliação são automáticos, sem revisão humana. A MLP e o Adam são implementados do zero em NumPy, sem frameworks de aprendizagem.

## O que a exatidão significa

O sistema aprende um gabarito estatístico: valor da nota acima da média de sua categoria em 2025 mais 1,2 desvios padrão de 2025. Os arquivos públicos não contêm rótulos de fraude comprovada. Portanto, 98,63% de precisão significa concordância dos alertas com esse critério histórico, não 98,63% de certeza de corrupção.

A fórmula do alvo e seu limiar não entram na rede. Entram `log1p(valor)`, o log do valor padronizado com média/desvio logarítmicos históricos da categoria e categorias one-hot. Partido, UF, mês e identidade do parlamentar não entram na decisão. Existem 25 categorias conhecidas, uma posição desconhecida e duas entradas numéricas: **28 → 16 tanh → 2 Softmax**.

## Separação dos dados

| Papel | Competência | Registros |
| --- | --- | ---: |
| Referências históricas | jan–dez/2025 | 193.477 |
| Fit | Subconjunto de 2025 | 5.000, sem reposição, classes balanceadas |
| Parada e escolha de arquitetura | jan–mar/2026 | 5.000 de 44.729, distribuição natural |
| Calibração sigmoidal | abr–mai/2026 | 35.417, distribuição natural |
| Escolha de limiar | jun/2026 | 16.239, distribuição natural |
| Teste reservado | jul–out/2026 | 35.922, distribuição natural |

As referências e os alvos de todos os períodos são definidos usando somente 2025. A normalização numérica é ajustada nas 5.000 notas de treino. A amostragem garante representação dos estratos existentes de classe, UF, Casa e categoria e usa semente 42.

As candidatas têm 16 e 32 neurônios, Adam com taxa 0,01, L2 de 0,00001, até 1.200 épocas e parada por perda de validação com paciência 100. Os melhores pesos são recuperados. A escolha da arquitetura maximiza average precision em jan–mar; escolheu 16 neurônios, época 1.199. O teste final não é exibido durante o fit nem usado para escolher parâmetros.

A calibração usa somente abr–mai, corrigindo a saída da rede balanceada para a população natural. Junho escolhe o limiar que maximiza F1 com recall mínimo de 90% nesse período. Limiar final: aproximadamente **29,9789%**. A meta de recall na validação não garante o mesmo resultado em dados futuros.

## Resultado no teste reservado

A referência é a arquitetura anterior de cinco entradas, retreinada nos mesmos 5.000 registros históricos, com categorias ordenadas pela mediana histórica, mesma validação temporal e taxa efetiva original de gradiente. Sua saída segue o limiar original de 50%, sem calibração. As duas abordagens enfrentam exatamente as mesmas 35.922 notas e o mesmo gabarito congelado.

| Métrica | Arquitetura anterior | Modelo refinado |
| --- | ---: | ---: |
| Acurácia | 96,3783% | 99,8775% |
| Precisão | 76,4468% | 98,6343% |
| Recall | 72,4310% | 99,6933% |
| F1 | 74,3847% | 99,1609% |
| Average precision | 85,1652% | 99,8889% |
| Verdadeiros positivos | 1.889 | 2.600 |
| Verdadeiros negativos | 32.732 | 33.278 |
| Falsos positivos | 582 | 36 |
| Falsos negativos | 719 | 8 |
| Brier | 0,026495 | 0,001237 |
| ECE, dez faixas | 2,7509 p.p. | 0,1248 p.p. |

A redução de falsos positivos foi de 93,81%. A regra de sempre prever normal teria acurácia de aproximadamente 92,74%, com recall zero. O teste contém 2.608 positivos estatísticos. Os intervalos Wilson de 95% são aproximadamente 98,12–99,01% para precisão e 99,40–99,84% para recall; não ajustam dependência entre notas do mesmo parlamentar.

O teste artificial aumentou 2.000 notas inicialmente normais para duas vezes o limiar histórico da respectiva categoria: 100% foram detectadas. Esse teste avalia sensibilidade a aumento de valor; seus exemplos não são notas reais de fraude e não entram no resultado principal.

Relatório reproduzível com métricas por UF, Casa e categoria, contagens e faixas de calibração: [outputs/model_evaluation.json](outputs/model_evaluation.json). Nenhum resultado do teste foi usado para retreinar ou ajustar o modelo após sua avaliação.

## Limites da avaliação automática

O teste mede generalização temporal do critério escolhido, cuja regra é simples e baseada em valor/categoria. Bom desempenho nesse teste não demonstra descoberta de irregularidades independentes desse critério. O sistema não resolve fracionamento de notas, preços locais, adequação do serviço, comprovação de entrega ou padrões relacionais entre fornecedores. Parlamentares podem aparecer em mais de um período; não é avaliação de pessoas inéditas.

Categorias inéditas usam a referência global histórica e uma entrada desconhecida explícita. Não há garantia de calibração para categorias ausentes do treino. As métricas detalhadas devem ser interpretadas com a contagem de positivos de cada grupo, especialmente grupos pequenos.

O limite histórico de 2025 permanece congelado para esta experiência. Usar os resultados de julho–outubro para criar novas versões exige um novo período reservado para manter a avaliação futura independente. Repetir a mesma execução verifica reprodutibilidade, não cria um novo teste independente.

## Reproduzir e conferir

```sh
python main.py --offline
python -m unittest discover -s tests
python scripts/validate_model.py
node --test tests/*.test.mjs
python scripts/validate_dataset.py
pnpm build
```

`validate_model.py` reconcilia todas as 325.784 inferências e as métricas do teste com os pesos exportados. Os testes JavaScript comparam o mesmo contrato em notas, extremos e valores próximos à fronteira, com tolerância de 1e-10. Os nomes dos CSVs permanecem iguais. A base integral continua disponível no dashboard; apenas o fit é limitado a 5.000 notas.

Referências metodológicas: [separação temporal](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) e [calibração de probabilidades](https://scikit-learn.org/stable/modules/calibration.html). As implementações deste projeto usam NumPy, não scikit-learn.
