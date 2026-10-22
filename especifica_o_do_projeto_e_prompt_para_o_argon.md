# Projeto: Auditoria Inteligente de Gastos Parlamentares (CEAP)
**Disciplina:** Redes Neurais / Inteligência Artificial (Nota do 2º Bimestre)  
**Prazo de Entrega:** 30/10/2026  
**Execução:** 100% CPU Local (Sem necessidade de GPU dedicada)

---

## 1. Visão Geral e Alinhamento com as Diretrizes

Este projeto une o rigor acadêmico exigido na atividade prática (construção de uma rede neural *Multilayer Perceptron* do zero em Python utilizando apenas `numpy`) com uma aplicação prática de forte impacto social no Brasil: **a auditoria de gastos públicos de deputados federais e a tradução de gastos atípicos em custo de oportunidade social** (educação, saúde e segurança).

### Checklist dos Requisitos do Professor:
1. **Desenvolvimento Base (Playlist de Referência):** Implementação da classe `NnModel` contendo os métodos `__init__`, `forward`, `loss`, `backpropagation` e `fit`, operando exclusivamente com álgebra linear via `numpy`.
2. **Escolha do Dataset (Origem Brasil):** Consumo de dados reais da **Cota para o Exercício da Atividade Parlamentar (CEAP)** via API Oficial de Dados Abertos da Câmara dos Deputados (`dadosabertos.camara.leg.br`).
3. **Customização e Aplicação:**
   * **Adequação da Arquitetura:** A camada de entrada deixa de ter apenas $2$ neurônios (como no exemplo sintético `make_moons` da playlist) e passa a receber $N$ variáveis reais do gasto parlamentar (Valor Líquido, Categoria da Despesa, Mês, UF/Região, Partido).
   * **Alteração da Função de Ativação:** Substituição da função de ativação padrão da camada oculta pela **Tangente Hiperbólica ($\tanh$)** ou **Leaky ReLU**, ajustando sua respectiva derivada no *backpropagation*.

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

## 4. Prompt Mestre para o Argon (Copiar e Colar)

Copie todo o bloco abaixo e envie diretamente para o **Argon** para que ele crie os arquivos, baixe os dados da Câmara dos Deputados, implemente a rede neural do zero e deixe o projeto pronto para rodar na sua máquina:

```text
Atue como um Engenheiro de Machine Learning Sênior. Preciso que você crie a base completa e funcional de um projeto em Python que roda 100% em CPU local. 

O projeto tem fins acadêmicos e deve seguir estritamente as regras abaixo:

### OBJETIVO DO SISTEMA
Criar um sistema híbrido de Transparência Pública Brasileira em duas etapas:
1. Uma Rede Neural Multilayer Perceptron (MLP) construída DO ZERO (usando APENAS `numpy`, sem scikit-learn ou pytorch para o modelo) que classifica gastos de deputados federais brasileiros em 0 (Normal) ou 1 (Atípico/Suspeito).
2. Um analisador de impacto social que pega os gastos classificados como 1 (Atípicos) pela rede neural, projeta o desperdício acumulado em 5 anos e gera um parecer detalhando como esse dinheiro poderia resolver vulnerabilidades brasileiras (educação, merenda escolar, saúde básica, segurança), usando a API do Google Gemini (com fallback de template inteligente caso a chave de API `GEMINI_API_KEY` não esteja configurada no ambiente).

### ESTRUTURA DE ARQUIVOS QUE VOCÊ DEVE CRIAR

1. `requirements.txt`:
   - Deve conter apenas: `numpy`, `pandas`, `matplotlib`, `requests`, `google-genai`, `python-dotenv`.

2. `src/data_fetcher.py`:
   - Deve conectar na API de Dados Abertos da Câmara dos Deputados (`https://dadosabertos.camara.leg.br/api/v2/deputados`).
   - Buscar uma lista de deputados em exercício e coletar suas despesas (`/deputados/{id}/despesas`) até formar um dataset de aproximadamente 800 a 1.000 registros reais.
   - IMPORTANTE: Implemente um mecanismo de segurança (try/except com timeout curto). Se a API da Câmara estiver fora do ar ou lenta, gere automaticamente um dataset realista baseado nos padrões reais da CEAP (com deputados, partidos, estados, tipos de despesas reais como "DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR", "LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES", "COMBUSTÍVEIS E LUBRIFICANTES", "FORNECIMENTO DE ALIMENTAÇÃO DO PARLAMENTAR" e valores em R$) e salve em `data/raw_despesas_camara.csv`.
   - Pré-processamento:
     - Extrair 5 features numéricas para a matriz `X`: 
       1. `valorLiquido` (float)
       2. `codTipoDespesa` (int mapeado da categoria)
       3. `mes` (int de 1 a 12)
       4. `codPartido` (int mapeado da sigla do partido)
       5. `codUF` (int mapeado do estado do deputado)
     - Criar a variável alvo `y` (0 ou 1): Calcule a média e o desvio padrão do gasto por categoria (`codTipoDespesa`). Se o `valorLiquido` ultrapassar `media + 1.2 * desvio_padrao` (ou for um valor excessivamente alto para a categoria), defina `y = 1` (Atípico). Caso contrário, `y = 0` (Normal). Garanta que o dataset final fique razoavelmente balanceado.
     - Crie uma função de normalização Z-Score em NumPy (`(X - mean) / (std + 1e-8)`) para que a rede neural não sofra overflow.

3. `src/nn_model.py`:
   - Implemente a classe `NnModel` construída 100% do zero com `numpy` seguindo a estrutura clássica:
     - `__init__(self, X, y, n_hidden=16, lr=0.05)`: Inicializa pesos `W1`, `b1`, `W2`, `b2` dinamicamente de acordo com `X.shape[1]` (5 features de entrada) e 2 neurônios de saída (para classificação binária via Softmax + Cross-Entropy, conforme a playlist de referência "Rede neural do zero em Python").
     - CUSTOMIZAÇÃO OBRIGATÓRIA: Na camada oculta, utilize a função de ativação **Tangente Hiperbólica (`np.tanh`)** ou **Leaky ReLU** (e não a sigmoide padrão), documentando claramente nos comentários onde a função de ativação e sua derivada foram customizadas.
     - `forward(self, X)`: Calcula a ativação da camada oculta e a saída com `softmax`.
     - `loss(self, y_pred)`: Calcula a perda de entropia cruzada (Cross-Entropy Loss).
     - `backpropagation(self, X, y)`: Calcula os gradientes `dW2`, `db2`, `dW1`, `db1` aplicando a derivada exata da função de ativação escolhida e atualiza os pesos.
     - `fit(self, epochs=2000)`: Executa o loop de treinamento, armazena o histórico de loss e acurácia, e salva o gráfico da curva de aprendizado em `outputs/loss_curve.png` usando `matplotlib`.
     - `predict(self, X)`: Retorna a classe predita (0 ou 1) e a probabilidade de ser um gasto atípico.

4. `src/impact_analyzer.py`:
   - Recebe os registros originais (não normalizados) que a `NnModel` classificou como `1` (Gasto Atípico).
   - Calcula a projeção: considerando um valor base aceitável para a categoria, calcula o excesso multiplicado por 60 meses (5 anos de impacto acumulado).
   - Se houver `GEMINI_API_KEY` no ambiente, chama o modelo `gemini-2.5-flash` pedindo uma análise direta e cidadã mostrando o que esse valor acumulado compraria em Educação (merendas escolares a R$ 1,50, kits escolares), Saúde (consultas, medicamentos básicos) e Segurança no estado (UF) daquele parlamentar.
   - Se não houver chave de API configurada, gera um relatório analítico automático muito bem formatado realizando os cálculos exatos com base nos custos médios públicos brasileiros (ex: custo médio de merenda escolar, salário mínimo, cestas básicas e equipamentos escolares) para que o sistema funcione perfeitamente offline.
   - Salva o resultado completo em `outputs/relatorio_auditoria.md`.

5. `main.py`:
   - Executa todo o fluxo mostrando logs claros no terminal:
     1. Coleta e preparação dos dados brasileiros (CEAP).
     2. Divisão em treino e teste (80%/20%) feita apenas com `numpy`.
     3. Instanciação e treinamento da `NnModel` mostrando a queda da `loss` e a acurácia final no conjunto de teste.
     4. Execução da auditoria nos casos suspeitos detectados pela rede neural e geração do relatório de impacto social em `outputs/relatorio_auditoria.md`.

6. `README.md`:
   - Documentação clara explicando:
     - Como o código cumpre os 3 requisitos da avaliação (Desenvolvimento Base, Dataset do Brasil e Customização da Arquitetura + Função de Ativação).
     - Como rodar o projeto (`pip install -r requirements.txt` e `python main.py`).

Por favor, crie todos esses arquivos agora, execute o `main.py` para validar que o treinamento da rede neural converge sem erros de dimensão matricial e verifique se os arquivos em `outputs/` foram gerados corretamente.
```