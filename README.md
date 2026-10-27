# OlhoCidadão — Auditoria Inteligente de Gastos Parlamentares (CEAP)

Sistema híbrido de **Transparência Pública Brasileira** que combina uma **Rede Neural Multilayer Perceptron (MLP) construída 100% do zero em NumPy** com um **Analisador de Impacto Social** (integrado à API do Google Gemini e com motor analítico offline integrado). O projeto roda **100% em CPU local**, sem necessidade de GPU dedicada ou frameworks pesados de Deep Learning.

---

## 1. Contexto Acadêmico & Diretrizes Oficiais da Atividade

> **Disciplina:** Redes Neurais / Inteligência Artificial  
> **Composição da Nota:** Atividade Prática do 2º Bimestre (composição de nota com a Prova Oficial)  
> **Formato de Trabalho:** Individual ou em Duplas  
> **Prazo de Entrega e Apresentação:** 30/10/2026
>
> **Materiais de Apoio Fundamentais Indicados pelo Professor:**
>
> - 📺 [Redes Neurais — Conceitos Básicos (Playlist)](https://www.youtube.com/watch?v=ebToEXQFCo4&list=PLQH6T1jnIb5J7vugBAauJsFU8Qgvuf-4X)
> - 📺 [Rede neural do zero em Python — Referência do Projeto (Playlist)](https://www.youtube.com/watch?v=OGcOH9damt0&list=PLQH6T1jnIb5I1s7chxtL24ZuX4g-PA7uK)

### 📋 Matriz de Conformidade com o Escopo da Atividade

| Item do Escopo Oficial          | Diretriz Exigida pelo Professor                                                                                                                                                                   | Como o Projeto OlhoCidadão Cumpre                                                                                                                                                                                                                                                                                                                                                                                                                    | Arquivo de Referência                                           |      Status       |
| :------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------- | :---------------: |
| **1. Desenvolvimento Base**     | _"Desenvolver a estrutura de rede neural criada e explicada na playlist de referência 'Rede neural do zero em Python'."_                                                                          | Implementação da classe `NnModel` construída **100% em NumPy do zero**, sem scikit-learn, pytorch ou tensorflow. Contém pesos Xavier, `forward`, `loss` analítica por Entropia Cruzada, `backpropagation` por Regra da Cadeia e loop `fit` com exportação de curva de perda.                                                                                                                                                                         | [`src/nn_model.py`](src/nn_model.py)                            | **100% Conforme** |
| **2. Escolha do Dataset**       | _"Escolher um banco de dados de qualquer área de interesse, com a condição obrigatória de que sejam dados de origem do Brasil."_                                                                  | Consumo de dados públicos oficiais brasileiros da **Câmara dos Deputados (CEAP)**, **Senado Federal (CEAPS)**, **IBGE** e **Tesouro Nacional/SICONFI**. Inclui 325.784 notas reais e treino balanceado de até 5.000 notas, com inferência na base integral.                                                                                                                                                                                            | [`src/data_fetcher.py`](src/data_fetcher.py) e [`data/`](data/) | **100% Conforme** |
| **3. Customização & Aplicação** | _"Aplicar o banco de dados escolhido na rede neural desenvolvida. Vocês devem alterar a função de ativação e adequar a arquitetura da rede de acordo com a problemática dos dados selecionados."_ | • **Adequação da Arquitetura:** Entradas dinâmicas com valor logarítmico, contexto histórico da categoria e codificação one-hot: **28 entradas nesta execução**, sem partido e UF na decisão.<br>• **Alteração da Ativação:** Substituição da sigmoide clássica pela **Tangente Hiperbólica ($\tanh$)**, com derivada analítica exata $f'(a) = 1 - a^2$ no backpropagation, operando em harmonia com normalização Z-Score. | [`src/nn_model.py`](src/nn_model.py)                   | **100% Conforme** |

---

## 2. Executar

```sh
pip install -r requirements.txt
pnpm install
python main.py --offline
pnpm dev
```

`--offline` reutiliza os dumps oficiais locais e não chama Gemini. `python main.py` atualiza as fontes e, se `GEMINI_API_KEY` estiver configurada, permite texto complementar via Gemini. A rede neural, a calibração e sua avaliação funcionam inteiramente em CPU, sem API de IA e sem revisão humana.

Para visualizar os artefatos já gerados, basta `pnpm dev`. `pnpm start` atualiza/treina antes de abrir o Vite; `pnpm build` gera `dist/`.

## 3. Modelo autônomo e avaliação

A implementação mantém `NnModel` feita do zero em NumPy: inicialização Xavier, `forward`, Softmax, entropia cruzada, derivada analítica de `tanh`, `backpropagation` e `fit`. O gradiente corresponde à perda média, com opção Adam implementada em NumPy, regularização L2 e parada antecipada com recuperação dos melhores pesos.

O modelo recebe o logaritmo do valor, esse valor padronizado pelo histórico da categoria e a categoria em one-hot, incluindo uma posição para categoria desconhecida. Não recebe o limiar, o alvo, partido, UF ou identidade do parlamentar. Nesta base: **28 → 16 tanh → 2 Softmax**. As duas primeiras entradas são normalizadas com parâmetros do treino; one-hot permanece binário.

O gabarito automático continua sendo `valor > média da categoria + 1,2 × desvio`, agora calculado **somente em 2025** e congelado antes da avaliação. Não existe ajuste de percentil para forçar quantidade de alertas. As métricas medem aprendizado desse critério de atipicidade; não são métricas de detecção de corrupção.

| Papel | Período | Registros disponíveis | Uso |
| --- | --- | ---: | --- |
| Referências e treino | 2025 | 193.477 | Estatísticas históricas; fit em 5.000 notas, metade de cada classe |
| Validação | jan–mar/2026 | 44.729 | Amostra natural de 5.000 para parada e escolha entre 16 e 32 neurônios |
| Calibração | abr–mai/2026 | 35.417 | Ajuste sigmoidal de scores na prevalência natural |
| Limiar | jun/2026 | 16.239 | Maximiza F1 com recall mínimo de 90% |
| Teste final | jul–out/2026 | 35.922 | Avaliação após congelar arquitetura, pesos, calibração e limiar |

Partições usam ano/mês de competência dos arquivos oficiais. Todos os registros de teste são posteriores aos de ajuste. A avaliação não é um teste de parlamentares inéditos: a mesma pessoa pode aparecer em períodos diferentes.

Comparação com a arquitetura anterior **retreinada nas mesmas 5.000 notas**, no mesmo teste temporal:

| Métrica | Arquitetura anterior | Modelo refinado |
| --- | ---: | ---: |
| Acurácia | 96,38% | 99,88% |
| Precisão dos alertas | 76,45% | 98,63% |
| Recall | 72,43% | 99,69% |
| F1 | 74,38% | 99,16% |
| Falsos positivos | 582 | 36 |
| Falsos negativos | 719 | 8 |
| Brier (menor é melhor) | 0,02649 | 0,00124 |

Esses resultados não devem ser comparados diretamente às métricas do antigo teste aleatório de 2.000 notas. O relatório completo, com métricas por UF, Casa e categoria, intervalos Wilson, bins de calibração e teste artificial de sensibilidade, está em [outputs/model_evaluation.json](outputs/model_evaluation.json). Consulte [MODEL_CARD.md](MODEL_CARD.md) para limites e protocolo.

## 4. Dados e dashboard

Os nomes dos CSVs em `data/` são preservados. Os dumps da Câmara e do Senado contêm 286.252 e 39.532 despesas positivas, respectivamente; `raw_despesas_camara.csv` e `processed_dataset.csv` contêm os **325.784 registros integrais**. O total observado é **R$ 418.329.396,80**, nas 27 UFs. O treino amostra dados; o dashboard recebe inferência em todos os registros.

O JSON usa colunas, dicionários e gzip; o navegador filtra a base integral em Web Worker. As obras municipais vêm do cadastro ObrasGov, com origem de localização e divergências visíveis, e referências estimadas aparecem identificadas. Detalhes, fontes e custos comparativos: [DATA_PIPELINE.md](DATA_PIPELINE.md).

O simulador na Ficha Técnica executa os mesmos pesos, transformação, calibração e limiar do Python. A ficha mostra o protocolo temporal e a comparação de desempenho. O resumo do dashboard é calculado a partir dos dados filtrados; Gemini é opcional apenas no relatório escrito.

`excessoMensal` é o nome legado da diferença positiva entre cada nota sinalizada e a média histórica de referência; não significa despesa recorrente. `impacto5Anos` multiplica essa diferença por 60 como cenário hipotético, sem demonstrar perda ou economia pública realizável.

## 5. Estrutura e verificação

- [src/nn_model.py](src/nn_model.py): MLP, gradientes, Adam e curva de aprendizado.
- [src/model_evaluation.py](src/model_evaluation.py): partições temporais, referências, entradas, calibração, limiar e métricas.
- [src/model_training.py](src/model_training.py): treino e escolha automática sem consultar o teste final.
- [src/model_export.py](src/model_export.py) e [web/model.js](web/model.js): contrato dos pesos e inferência compatível no navegador.
- [outputs/loss_curve.png](outputs/loss_curve.png): curva de treino/validação; teste final fora do fit.
- [outputs/model_parity.json](outputs/model_parity.json): casos gerados por NumPy para verificar a inferência JavaScript.
- [outputs/relatorio_auditoria.md](outputs/relatorio_auditoria.md): análise descritiva e comparações de custo.

```sh
python -m unittest discover -s tests
python scripts/validate_model.py
node --test tests/*.test.mjs
python scripts/validate_dataset.py
pnpm build
```

Os testes verificam gradientes por diferenças finitas, ausência de influência de valores futuros nas referências, separação temporal, recuperação dos melhores pesos, calibração, limiares e paridade NumPy/JavaScript. Os validadores conciliam todas as inferências com CSV/payload e os valores monetários com as fontes oficiais.
