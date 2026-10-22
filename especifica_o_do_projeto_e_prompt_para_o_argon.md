# Projeto: Auditoria Inteligente de Gastos Parlamentares (CEAP)
**Disciplina:** Redes Neurais / Inteligência Artificial (Nota do 2º Bimestre)  
**Prazo de Entrega:** 30/10/2026  
**Execução:** 100% CPU Local (Sem necessidade de GPU dedicada)

---

## 1. Diretrizes Oficiais da Atividade Prática (Avaliação do 2º Bimestre)

> *"A diretriz para a nossa atividade prática que irá compor a nota do 2º bimestre, juntamente com a prova oficial."*

### Escopo Oficial da Atividade:
O objetivo é praticar e expandir os conhecimentos adquiridos ao longo das aulas através dos seguintes passos obrigatórios:

1. **Desenvolvimento Base:** Desenvolver a estrutura de rede neural criada e explicada na playlist de referência *"Rede neural do zero em Python"*.
2. **Escolha do Dataset:** Escolher um banco de dados de qualquer área de interesse, com a **condição obrigatória de que sejam dados de origem do Brasil**.
3. **Customização & Aplicação:** Aplicar o banco de dados escolhido na rede neural desenvolvida. Vocês devem **alterar a função de ativação** e **adequar a arquitetura da rede** de acordo com a problemática dos dados selecionados.

### Formato de Trabalho, Prazo e Materiais de Apoio:
- **Formato:** Individual ou em duplas (no máximo).
- **Prazo de Entrega e Apresentação:** Data limite: **30/10/2026**.
- **Materiais de Apoio Fundamentais Indicados pelo Professor:**
  - 📺 **Redes Neurais - Conceitos Básicos:** [Playlist no YouTube](https://www.youtube.com/watch?v=ebToEXQFCo4&list=PLQH6T1jnIb5J7vugBAauJsFU8Qgvuf-4X)
  - 📺 **Rede neural do zero em Python (Referência do Projeto):** [Playlist no YouTube](https://www.youtube.com/watch?v=OGcOH9damt0&list=PLQH6T1jnIb5I1s7chxtL24ZuX4g-PA7uK)

---

### Matriz de Conformidade do Projeto OlhoCidadão com a Atividade

| Requisito do Professor | Exigência da Atividade | Implementação no Projeto OlhoCidadão | Arquivo de Código | Conformidade |
| :--- | :--- | :--- | :--- | :---: |
| **Passo 1: Desenvolvimento Base** | Implementar a estrutura de rede neural da playlist de referência. | Classe `NnModel` construída **100% em NumPy do zero** (inicialização de pesos Xavier, forward pass, categorical cross-entropy loss analítica, backpropagation por regra da cadeia e fit loop). | [`src/nn_model.py`](src/nn_model.py) | **100% Conforme** |
| **Passo 2: Escolha do Dataset** | Banco de dados de qualquer área com **origem obrigatória do Brasil**. | Utilização de bases públicas oficiais do Brasil: **Câmara dos Deputados (CEAP)**, **Senado Federal (CEAPS)**, **IBGE** e **Tesouro Nacional/SICONFI**. Mais de 326 mil notas reais integradas. | [`src/data_fetcher.py`](src/data_fetcher.py) e [`data/`](data/) | **100% Conforme** |
| **Passo 3: Customização & Aplicação** | Adequar a arquitetura da rede e alterar a função de ativação. | • **Arquitetura:** Camada de entrada expandida para 5 neurônios reais (`valorLiquido`, `codTipoDespesa`, `mes`, `codPartido`, `codUF`).<br>• **Ativação:** Substituição da sigmoide clássica pela **Tangente Hiperbólica ($\tanh$)** com derivada analítica exata ($1 - a^2$). | [`src/nn_model.py`](src/nn_model.py#L38-L102) | **100% Conforme** |

---

## 2. Arquitetura do Sistema (Pipeline em 2 Etapas)

Para manter a fidelidade ao trabalho acadêmico e adicionar a camada de análise social com IA Generativa, o sistema funciona em duas etapas conectadas:

### Etapa 1: O Motor de Detecção (Rede Neural do Zero - `NnModel`)
* Recebe a matriz de dados de despesas parlamentares normalizada via *Z-Score*:
  $$X_{\text{norm}} = \frac{X - \mu}{\sigma + 10^{-8}}$$
* Processa os dados através da camada oculta utilizando a função de ativação customizada $\tanh(z)$:
  $$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$
  Com derivada aplicada no *backpropagation*:
  $$f'(a) = 1 - a^2$$
* Classifica cada nota fiscal/despesa em duas classes na saída:
  * **Classe $0$:** Gasto dentro do padrão histórico da categoria.
  * **Classe $1$:** Gasto Atípico / Suspeito de Superfaturamento.

### Etapa 2: O Tradutor de Impacto Social (Módulo LLM / IA Generativa)
* Sempre que a Rede Neural do Zero classifica um gasto como **Classe $1$ (Atípico)**, os dados dessa despesa são enviados para o módulo de Inteligência Artificial Generativa (via API do Gemini/OpenAI ou modo simulado offline).
* O módulo projeta o excesso daquele gasto ao longo de um mandato ($4$ a $5$ anos) e calcula o que poderia ser financiado na sociedade brasileira com esse montante (ex: quantidades de merendas escolares no PNAE, cestas básicas, computadores para escolas públicas ou insumos hospitalares do SUS).

---

## 3. Estrutura de Diretórios e Bases de Dados Massivas (`data/`)

```text
OlhoCidadão/
│
├── data/
│   ├── cota_parlamentar_camara_completa.csv    # DUMP MASSIVO: 326.304 notas fiscais reais (513 Deputados, 70.4 MB)
│   ├── despesas_senadores_ceaps_completo.csv   # DUMP MASSIVO: 39.652 notas fiscais reais (81 Senadores, 9.9 MB)
│   ├── deputados_historico_completo_camara.csv # Histórico completo de 7.889 Deputados da história do Brasil (1.3 MB)
│   ├── deputados_federais_camara.csv           # 513 Deputados Federais em exercício (122 KB)
│   ├── senadores_federais_completo.csv         # 81 Senadores da República em exercício (18 KB)
│   ├── prefeituras_municipios_tesouro_ibge.csv # 5.570 Prefeituras Municipais com CNPJ e População (326 KB)
│   ├── raw_despesas_camara.csv                 # Amostra estratificada de 10.000 notas fiscais reais (2.8 MB)
│   ├── processed_dataset.csv                   # Dataset de 10.000 amostras codificadas e rotuladas para MLP (3.2 MB)
│   ├── partidos_politicos_camara.csv           # Mapeamento oficial de legendas partidárias (2 KB)
│   ├── estados_ibge_oficial.csv                # Tabela de 27 UFs e 5 Macrorregiões do IBGE (2.5 KB)
│   └── indicadores_custos_publicos_br.csv      # Custos de referência social PNAE/SUS/DIEESE/SENASP (1.5 KB)
│
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py               # Coleta, consolidação massiva e normalização Z-Score
│   ├── nn_model.py                   # Rede Neural MLP construída do zero em NumPy (tanh + softmax)
│   └── impact_analyzer.py            # Analisador de impacto social em 5 anos e parecer cidadão
│
├── outputs/
│   ├── loss_curve.png                # Gráfico da queda de loss e acurácia da rede neural
│   └── relatorio_auditoria.md        # Relatório de auditoria e impacto social
│
├── public/
│   └── data/
│       ├── dashboard_data.json       # Pesos treinados e notas auditadas para a aplicação web
│       └── loss_curve.png            # Curva de treinamento servida no navegador
│
├── web/
│   ├── app.js                        # Dashboard interativo + inferência da MLP no navegador
│   └── style.css                     # Design responsivo e painel cidadão
│
├── index.html                        # Aplicação Web (Vite + pnpm)
├── main.py                           # Orquestrador ponta a ponta do pipeline
├── requirements.txt                  # Dependências Python (numpy, pandas, matplotlib, requests, google-genai)
├── package.json                      # Gerenciador pnpm e scripts frontend
└── README.md                         # Documentação acadêmica detalhada
```

---

