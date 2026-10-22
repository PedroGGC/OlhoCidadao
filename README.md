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
| **2. Escolha do Dataset**       | _"Escolher um banco de dados de qualquer área de interesse, com a condição obrigatória de que sejam dados de origem do Brasil."_                                                                  | Consumo de dados públicos oficiais brasileiros da **Câmara dos Deputados (CEAP)**, **Senado Federal (CEAPS)**, **IBGE** e **Tesouro Nacional/SICONFI**. Inclui dump integral de 326 mil notas fiscais e amostra balanceada de 10.000 notas fiscais reais.                                                                                                                                                                                            | [`src/data_fetcher.py`](src/data_fetcher.py) e [`data/`](data/) | **100% Conforme** |
| **3. Customização & Aplicação** | _"Aplicar o banco de dados escolhido na rede neural desenvolvida. Vocês devem alterar a função de ativação e adequar a arquitetura da rede de acordo com a problemática dos dados selecionados."_ | • **Adequação da Arquitetura:** Camada de entrada expandida de 2 entradas sintéticas (`make_moons`) para **5 features reais** de gastos parlamentares (`valorLiquido`, `codTipoDespesa`, `mes`, `codPartido`, `codUF`).<br>• **Alteração da Ativação:** Substituição da sigmoide clássica pela **Tangente Hiperbólica ($\tanh$)**, com derivada analítica exata $f'(a) = 1 - a^2$ no backpropagation, operando em harmonia com normalização Z-Score. | [`src/nn_model.py`](src/nn_model.py#L38-L102)                   | **100% Conforme** |

---

## 2. Detalhamento Técnico dos 3 Requisitos da Atividade

### Requisito 1: Desenvolvimento Base (Rede Neural do Zero em Python/NumPy)

O arquivo [`src/nn_model.py`](src/nn_model.py) implementa a classe `NnModel` utilizando exclusivamente álgebra linear vetorial em `numpy` (sem `scikit-learn`, `pytorch` ou `tensorflow`), contemplando toda a estrutura clássica da playlist de referência (_"Rede neural do zero em Python"_):

- `__init__(self, X, y, n_hidden=16, lr=0.05)`: Inicialização dinâmica das matrizes de pesos `W1`, `b1`, `W2` e `b2` via método de Xavier/Glorot.
- `forward(self, X)`: Propagação direta da camada de entrada para a camada oculta e cálculo de probabilidades na saída via função `softmax` numericamente estável.
- `loss(self, y_pred)`: Cálculo exato da perda por **Entropia Cruzada Categórica (_Cross-Entropy Loss_)**:
  $$\mathcal{L}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} \log\left(\hat{y}_{i, y_i}\right)$$
- `backpropagation(self, X, y)`: Cálculo analítico dos gradientes `dW2`, `db2`, `dW1`, `db1` aplicando a Regra da Cadeia e atualização por Gradiente Descendente.
- `fit(self, epochs=2000)` e `predict(self, X)`: Loop de treinamento completo com exportação automática da curva de aprendizado em `outputs/loss_curve.png`.

### Requisito 2: Escolha do Dataset (Origem Brasil — Câmara dos Deputados & Senado Federal)

O módulo [`src/data_fetcher.py`](src/data_fetcher.py) consome e consolida dados públicos oficiais massivos da **Cota para o Exercício da Atividade Parlamentar (CEAP)** da Câmara dos Deputados e do **CEAPS** do Senado Federal, além de cadastros do IBGE e Tesouro Nacional:

- **Base Completa & Dumps Massivos Integrados:** O repositório já inclui a base completa histórica de notas fiscais da Câmara dos Deputados (`cota_parlamentar_camara_completa.csv` — **326.304 notas fiscais**, ~70.4 MB) e a base integral dos senadores (`despesas_senadores_ceaps_completo.csv` — **39.652 notas fiscais**, ~9.9 MB), cobrindo todos os 513 deputados federais e 81 senadores da República.
- **Amostra Estratificada para Treinamento (10.000 notas fiscais):** Para garantir treinamento ágil em qualquer CPU convencional sem lentidão de I/O em disco, o pipeline estrutura uma amostra balanceada e representativa de **10.000 notas fiscais reais** salvas em `data/raw_despesas_camara.csv` e `data/processed_dataset.csv`.
- **Mecanismo de Resiliência Offline/Timeout:** Caso os portais governamentais apresentem indisponibilidade ou limite de taxa (_rate limiting_), o pipeline carrega diretamente os dados locais já baixados ou aciona o gerador estatístico fundamentado nas métricas reais da CEAP.
- **Rotulagem Estatística (`y`) e Normalização Z-Score:** Calcula a média ($\mu_c$) e o desvio padrão ($\sigma_c$) por categoria de despesa (`codTipoDespesa`). Notas cujo `valorLiquido` ultrapassam $\mu_c + 1.2\sigma_c$ recebem `y = 1` (**Gasto Atípico/Suspeito**); as demais recebem `y = 0` (**Gasto Normal**). Para evitar _overflow_ exponencial na rede, aplica-se a normalização Z-Score em NumPy puro:
  $$X_{\text{norm}} = \frac{X - \mu}{\sigma + 10^{-8}}$$

### Requisito 3: Customização da Arquitetura e da Função de Ativação

Em relação ao modelo didático clássico de 2 entradas sintéticas (`make_moons`) com ativação sigmoide, foram realizadas duas customizações fundamentais em [`src/nn_model.py`](src/nn_model.py):

1. **Adequação da Camada de Entrada (5 Features Reais):** A matriz `X` alimenta dinamicamente `n_input = X.shape[1] = 5` neurônios de entrada correspondentes a:
   1. `valorLiquido` (float em R$)
   2. `codTipoDespesa` (int mapeado da rubrica da CEAP)
   3. `mes` (int de 1 a 12)
   4. `codPartido` (int mapeado da sigla partidária)
   5. `codUF` (int mapeado da Unidade Federativa do parlamentar)
2. **Customização da Função de Ativação na Camada Oculta (`np.tanh`):** Substituiu-se a sigmoide padrão pela **Tangente Hiperbólica ($\tanh$)** (com suporte adicional a _Leaky ReLU_):
   $$a^{(1)} = \tanh\left(z^{(1)}\right) = \frac{e^{z^{(1)}} - e^{-z^{(1)}}}{e^{z^{(1)}} + e^{-z^{(1)}}}$$
   No método `backpropagation`, a retropropagação do erro aplica a **derivada analítica exata da Tangente Hiperbólica**:
   $$\frac{\partial \tanh\left(z^{(1)}\right)}{\partial z^{(1)}} = 1 - \left(a^{(1)}\right)^2$$
   Como a $\tanh$ é centrada em zero no intervalo $(-1, 1)$, ela opera em sinergia direta com a normalização Z-Score, eliminando o problema de _zig-zag_ nos gradientes e acelerando a convergência.

---

## 3. Analisador de Impacto Social (`src/impact_analyzer.py`)

Todos os gastos do conjunto de teste classificados como **Classe 1 (Atípicos)** pela Rede Neural são auditados pelo módulo [`src/impact_analyzer.py`](src/impact_analyzer.py):

- Calcula o excesso mensal frente à média aceitável da categoria e projeta o desperdício acumulado ao longo de **60 meses (5 anos)**.
- Converte o desperdício projetado em custo de oportunidade social com base em indicadores públicos brasileiros:
  - **Educação Básica:** Merendas escolares do PNAE (R$ 1,50/refeição), kits escolares completos (R$ 120,00) e computadores para laboratórios públicos (R$ 2.800,00).
  - **Saúde Pública (SUS):** Consultas médicas de atenção primária (R$ 65,00) e kits de medicamentos da Farmácia Básica (R$ 45,00).
  - **Segurança e Assistência Social:** Diárias operacionais de patrulhamento estadual (R$ 420,00) e cestas básicas familiares (R$ 750,00).
- Se a variável de ambiente `GEMINI_API_KEY` estiver configurada, utiliza o SDK `google-genai` para enriquecer o parecer técnico e cidadão; caso contrário, utiliza automaticamente o motor analítico offline, gerando o documento completo em `outputs/relatorio_auditoria.md`.

---

## 4. Estrutura de Arquivos do Projeto

```text
OlhoCidadão/
├── data/
│   ├── cota_parlamentar_camara_completa.csv    # DUMP MASSIVO: 326.304 notas fiscais reais (TODOS os 513 Deputados, 70.4 MB)
│   ├── despesas_senadores_ceaps_completo.csv   # DUMP MASSIVO: 39.652 notas fiscais reais (TODOS os 81 Senadores, 9.9 MB)
│   ├── deputados_historico_completo_camara.csv # TODOS os 7.889 Deputados Federais da história do Brasil (1.3 MB)
│   ├── deputados_federais_camara.csv           # TODOS os 513 Deputados Federais em exercício (API Câmara, 122 KB)
│   ├── senadores_federais_completo.csv         # TODOS os 81 Senadores da República em exercício (API Senado, 18 KB)
│   ├── prefeituras_municipios_tesouro_ibge.csv # TODAS as 5.570 Prefeituras Municipais + Estados com CNPJ e população (Tesouro/IBGE, 326 KB)
│   ├── raw_despesas_camara.csv                 # Amostra estratificada de 10.000 notas reais de Deputados e Senadores (2.8 MB)
│   ├── processed_dataset.csv                   # Dataset de 10.000 linhas codificado e rotulado (y = 0 ou 1, 3.2 MB)
│   ├── partidos_politicos_camara.csv           # Tabela oficial de Partidos Políticos (2 KB)
│   ├── estados_ibge_oficial.csv                # Tabela oficial das 27 UFs e Regiões do Brasil (IBGE, 2.5 KB)
│   └── indicadores_custos_publicos_br.csv      # Custos unitários de referência FNDE/SUS/DIEESE/SENASP (1.5 KB)
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py               # Coleta e consolidação de dados massivos, amostragem e Z-Score
│   ├── nn_model.py                   # Rede Neural MLP construída 100% do zero em NumPy (tanh)
│   └── impact_analyzer.py            # Projeção de 5 anos e geração do parecer de impacto social
├── outputs/
│   ├── loss_curve.png                     # Curva de aprendizado (Queda da Loss e Acurácia)
│   └── relatorio_auditoria.md             # Relatório final de auditoria e custo de oportunidade social
├── public/
│   └── data/
│       ├── dashboard_data.json            # Pesos treinados (W1, b1, W2, b2), Z-Score e notas auditadas para a Web
│       └── loss_curve.png                 # Gráfico da curva de aprendizado servido na interface Web
├── web/
│   ├── app.js                             # Lógica do Dashboard + Inferência MLP (tanh + softmax) no navegador
│   └── style.css                          # Estilização responsiva com Container Queries e Sticky Headers
├── index.html                             # Página principal da Aplicação Web (Vite + pnpm)
├── package.json                           # Configuração do projeto Web gerenciado via pnpm
├── main.py                                # Orquestrador ponta a ponta do pipeline Python/NumPy
├── requirements.txt                       # Dependências leves para execução 100% em CPU
└── README.md                              # Documentação técnica e acadêmica
```

### 4.1 Detalhamento dos Datasets Públicos Massivos e Completos (`data/`)

O projeto não se restringe a pequenos recortes sintetizados. Ele consolida uma infraestrutura completa de dados abertos governamentais brasileiros:

| Arquivo em `data/`                            | Fonte Oficial                                                                                             | Qtd. Registros |   Tamanho    | Conteúdo e Finalidade                                                                                                                                     |
| :-------------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :------------: | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`cota_parlamentar_camara_completa.csv`**    | [Câmara dos Deputados](https://dadosabertos.camara.leg.br/)                                               |  **326.304**   | **~70.4 MB** | **DUMP COMPLETO:** Todas as despesas e notas fiscais declaradas por todos os 513 deputados federais (com CNPJ, fornecedor, valor líquido e link da NF-e). |
| **`despesas_senadores_ceaps_completo.csv`**   | [Senado Federal (CEAPS)](https://dadosabertos.senado.leg.br/)                                             |   **39.652**   | **~9.9 MB**  | **DUMP COMPLETO:** Todas as notas fiscais dos 81 senadores da República em exercício nos exercícios recentes de 2025 e 2026.                              |
| **`deputados_historico_completo_camara.csv`** | [API Dados Abertos](https://dadosabertos.camara.leg.br/api/v2/deputados)                                  |   **7.889**    | **~1.3 MB**  | Base histórica com todos os parlamentares federais que já exerceram mandato na história do Brasil.                                                        |
| **`processed_dataset.csv`**                   | Amostragem Estratificada + Engenharia de Features                                                         |   **10.000**   | **~3.2 MB**  | Dataset pré-processado com as 5 features normalizadas via Z-Score e rótulo supervisionado `y \in \{0, 1\}` para treino e teste da MLP.                    |
| **`raw_despesas_camara.csv`**                 | Extração Cruzada (Deputados + Senadores)                                                                  |   **10.000**   | **~2.8 MB**  | Amostra estratificada preservando nomes de parlamentares, partidos, categorias originais e URLs das notas fiscais.                                        |
| **`prefeituras_municipios_tesouro_ibge.csv`** | [IBGE](https://servicodados.ibge.gov.br/) & [Tesouro Nacional / SICONFI](https://siconfi.tesouro.gov.br/) |   **5.570**    | **~326 KB**  | Relação de todos os 5.570 municípios e capitais brasileiras, incluindo CNPJs de prefeituras, estados e estimativas populacionais.                         |
| **`deputados_federais_camara.csv`**           | [API da Câmara](https://dadosabertos.camara.leg.br/api/v2/deputados)                                      |    **513**     | **~122 KB**  | Cadastro completo dos 513 deputados da 57ª legislatura em exercício (foto oficial, partido, UF, e-mail).                                                  |
| **`senadores_federais_completo.csv`**         | [API do Senado](https://adm.senado.gov.br/)                                                               |     **81**     |  **~18 KB**  | Cadastro dos 81 senadores federais com links oficiais e gabinetes.                                                                                        |
| **`estados_ibge_oficial.csv`**                | [API Localidades IBGE](https://servicodados.ibge.gov.br/)                                                 |     **27**     | **~2.5 KB**  | Tabela de referência das 27 Unidades Federativas e 5 Macrorregiões do Brasil.                                                                             |
| **`partidos_politicos_camara.csv`**           | [API Câmara](https://dadosabertos.camara.leg.br/)                                                         |  **Legendas**  | **~2.0 KB**  | Mapeamento oficial de legendas partidárias brasileiras.                                                                                                   |
| **`indicadores_custos_publicos_br.csv`**      | FNDE/PNAE, DataSUS/SIGTAP, DIEESE, SENASP                                                                 |   **Tabela**   | **~1.5 KB**  | Custos unitários públicos auditados para o cálculo de custo de oportunidade social.                                                                       |

> [!NOTE]
> **Por que esses dados massivos estão versionados no repositório?**  
> Para garantir **100% de autonomia e reprodutibilidade imediata**. Ao clonar o projeto, o avaliador ou usuário tem à disposição o ecossistema integral de dados sem depender de conexões externas lentas ou de instabilidades momentâneas nas APIs governamentais.

> [!WARNING]
> **Aviso de Arquivos Grandes no GitHub (> 50 MB):**  
> O arquivo `cota_parlamentar_camara_completa.csv` possui **~70.4 MB**, o que dispara um aviso automático do GitHub recomendando o uso do _Git LFS (Large File Storage)_ para arquivos acima de 50 MB. O GitHub aceita arquivos de até 100 MB em commits padrão. Caso essa base venha a ser expandida no futuro além de 100 MB, utilize o Git LFS ou ignore a base bruta completa no `.gitignore`.

---

## 5. Como Executar o Projeto (Backend Python + Site Web com `pnpm`)

### Passo 1: Instalar as dependências do Python (Motor da Rede Neural)

```bash
pip install -r requirements.txt
```

### Passo 2: Instalar as dependências Web via `pnpm`

```bash
pnpm install
```

### Passo 3: (Opcional) Configurar chave da API Gemini

Caso deseje utilizar a geração complementar via IA Generativa do Google Gemini, defina a variável de ambiente `GEMINI_API_KEY` ou crie um arquivo `.env` na raiz:

```env
GEMINI_API_KEY=sua_chave_aqui
```

_(Se não configurar a chave, o projeto funcionará 100% offline usando o motor analítico matemático embutido)._

### Passo 4: Treinar a Rede Neural e Iniciar o Site Web (`pnpm`)

Você pode rodar o pipeline de treinamento e subir o servidor Web interativo com um único comando `pnpm`:

```bash
pnpm start
```

Ou executar cada etapa separadamente via `pnpm`:

```bash
# 1. Coleta dados da Câmara/IBGE, treina a Rede Neural em NumPy e exporta os artefatos Web
pnpm run train

# 2. Inicia o servidor de desenvolvimento Web (Vite) em http://localhost:5173
pnpm run dev

# 3. Gera a build de produção otimizada do site na pasta dist/
pnpm run build

# 4. Visualiza a build de produção localmente
pnpm run preview
```

_(Também é possível rodar apenas o pipeline Python diretamente no terminal com `python main.py`)._

### O que você encontrará na Aplicação Web (`pnpm dev`) e em `outputs/`:

1. **Dashboard Web Interativo (`http://localhost:5173`):**
   - **Simulador da Rede Neural em Tempo Real:** Executa o _Forward Pass_ com os pesos reais treinados (`W1`, `b1`, `W2`, `b2`), normalização Z-Score e ativação `tanh(z)` diretamente no navegador, exibindo as ativações dos 16 neurônios ocultos ao vivo.
   - **Painel de Custo de Oportunidade Social (5 Anos):** Conversão automática em merendas escolares (PNAE), kits escolares, computadores, consultas no SUS, medicamentos básicos e diárias de segurança.
   - **Tabela de Notas Fiscais Reais Auditadas:** Filtros por classificação (`Atípico` vs `Normal`), UF e busca textual por deputado, fornecedor ou CNPJ, com link direto para o PDF oficial da nota fiscal no portal da Câmara dos Deputados.
2. `outputs/loss_curve.png`: Gráfico comprovando a convergência da Rede Neural (`NnModel`).
3. `outputs/relatorio_auditoria.md`: Relatório detalhado dos gastos atípicos detectados e seu impacto social em 5 anos.

---

## 6. Como a IA Funciona no Projeto? (Processamento, Comportamento e Tempo Real vs. Pré-processamento)

O projeto possui **duas camadas de inteligência** que trabalham juntas. Veja exatamente o que acontece nos bastidores:

### A. A Rede Neural (`NnModel`) — O "Detector de Gastos Fora do Padrão"

Implementada do zero em Python usando apenas `numpy` em [`src/nn_model.py`](src/nn_model.py):

1. **O que ela recebe de entrada (`X`):** Para cada nota fiscal de deputado ou senador, [`src/data_fetcher.py`](src/data_fetcher.py) extrai **5 números (features)**:
   - `valorLiquido`: Valor da nota em R$
   - `codTipoDespesa`: Código da categoria (ex.: Combustível = 1, Divulgação = 2, Locação de veículos = 3...)
   - `mes`: Mês da nota (1 a 12)
   - `codPartido`: Código numérico do partido
   - `codUF`: Código numérico do estado (UF)
2. **Como ela aprende (Treinamento Offline — quando você roda `python main.py` ou `pnpm run train`):**
   - Primeiro, `normalize_zscore` coloca as 5 variáveis na mesma escala matemática para que valores em milhares de reais não "esmaguem" o código do mês (1 a 12).
   - Das **355.546 notas reais consolidadas** (Câmara e Senado), **284.436 (80%) vão para treino** e **71.110 (20%) ficam escondidas para teste** em `train_test_split_numpy`.
   - Durante **2.000 épocas** (`NnModel.fit`), a rede tenta adivinhar se cada gasto é **Classe 0 (Normal)** ou **Classe 1 (Acima do padrão daquela categoria)**, mede o erro via `NnModel.loss` e corrige seus pesos internos (`W1`, `b1`, `W2`, `b2`) via `NnModel.backpropagation`.
3. **Ela analisa em tempo real no site ou antes?**
   - **Nos registros da tabela principal do site:** A classificação (`Classe 0` ou `Classe 1` e a `% de probabilidade`) já vem **pré-calculada pelo Python** e salva em [`public/data/dashboard_data.json`](public/data/dashboard_data.json) quando `main.py` roda. Por isso, quando você filtra por partido, estado, mês ou parlamentar no site, os gráficos e alertas respondem **instantaneamente (em milissegundos)**.
   - **No Simulador da aba "Ficha Técnica":** Ali a rede neural roda **100% em tempo real no seu navegador**! O arquivo `main.py` exporta as matrizes de pesos reais treinadas (`W1`, `b1`, `W2`, `b2`) para o JSON, e a função `predictMlpBrowser` em [`web/app.js`](web/app.js) executa a multiplicação matricial `Z-Score → tanh → Softmax` ao vivo a cada vez que você digita um valor ou troca a categoria.

### B. O Resumo Inteligente e Impacto Social — O "Tradutor Cidadão"

- **No site (em tempo real a cada filtro):** A função `renderKpisAndInsight` em [`web/app.js`](web/app.js) lê todas as notas do filtro ativo, cruza com as notas que a Rede Neural marcou como `Classe 1` (alerta), calcula a diferença em relação à média geral, identifica a maior categoria/fornecedor/partido/UF daquele recorte e escreve na hora o parágrafo explicativo e a conversão em merendas escolares e consultas no SUS.
- **No script Python (`ImpactAnalyzer`):** Gera o documento [`outputs/relatorio_auditoria.md`](outputs/relatorio_auditoria.md). Se houver uma chave `GEMINI_API_KEY` no arquivo `.env`, ele consulta o modelo Gemini para redigir um parecer adicional; se não houver chave (modo padrão 100% local), ele usa o motor analítico determinístico em Python sem precisar de internet.

---

## 7. Como a IA Determina o que é um Valor "Normal" (`0`) ou "Com Alerta" (`1`)? (Metodologia e Vieses)

Como a Câmara e o Senado **não publicam uma coluna dizendo "esta nota é suspeita"**, o sistema primeiro cria uma **régua estatística por categoria** para ensinar a Rede Neural, e depois a Rede Neural aprende a **fronteira não-linear** entre as variáveis dentro de `preprocess_dataset` ([`src/data_fetcher.py`](src/data_fetcher.py)) e `NnModel` ([`src/nn_model.py`](src/nn_model.py)):

### Passo 1: A régua por categoria (quem cria o gabarito `y` de treino)

Não faria sentido comparar uma nota de **combustível** (geralmente R$ 150 a R$ 500) com uma nota de **divulgação parlamentar** ou **locação de veículos** (que frequentemente passa de R$ 5.000 ou R$ 10.000). Por isso, o código separa todas as notas por **tipo de despesa** (`codTipoDespesa`) e calcula para cada categoria:

1. A **média ($\mu$)** e o **desvio padrão ($\sigma$)** daquela categoria específica.
2. O **limiar estatístico**:
   $$\text{Limiar} = \mu_{\text{categoria}} + 1{,}2 \cdot \sigma_{\text{categoria}}$$
   _(Se uma categoria tiver poucos picos extremos que puxam o desvio padrão muito para cima, o código aplica um teto no percentil 74 em `src/data_fetcher.py` para garantir que ~25% das notas fiquem na Classe 1 e a rede tenha exemplos suficientes para aprender)._
3. Se o `valorLiquido` da nota passa desse limiar da sua própria categoria, ela recebe **`y = 1` (Alerta / Acima do padrão)**; caso contrário, recebe **`y = 0` (Dentro da média)**.

### Passo 2: Como a Rede Neural (`NnModel`) aprende e decide

Durante o treino (`NnModel.fit`), a rede neural **não recebe a fórmula do limiar**. Ela recebe apenas os **5 números crus normalizados** (`valorLiquido`, `codTipoDespesa`, `mes`, `codPartido`, `codUF`) e precisa descobrir sozinha, ajustando os pesos dos 16 neurônios com `tanh`, que:

- R$ 2.500 no código `1` (_Combustíveis_) é muito alto $\rightarrow$ **Classe 1 (Alerta)**;
- R$ 2.500 no código `2` (_Divulgação_) ou `3` (_Locação de Veículos_) é normal $\rightarrow$ **Classe 0 (Normal)**.

### Passo 3: Vieses e Limitações Estatísticas (Importante para Apresentação)

Para fins acadêmicos e de triagem estatística inicial, o método é robusto (**95,80% de acurácia no teste**), mas no mundo real possui **4 vieses importantes**:

1. **"Anomalia Estatística" $\neq$ "Ilegalidade ou Corrupção":** A IA avalia quão caro um recibo único foi em relação à média da categoria. Por exemplo, um parlamentar do **Acre (`AC`)**, **Roraima (`RR`)** ou **Amazonas (`AM`)** paga passagens aéreas muito mais caras para Brasília do que um parlamentar de **Goiás (`GO`)** ou **Minas Gerais (`MG`)**. Uma passagem de R$ 4.500 saindo do Norte pode receber **Alerta (`Classe 1`)** mesmo sendo o preço regular da companhia aérea.
2. **Fracionamento de Notas (Falsos Negativos):** Como a rede avalia **nota por nota individualmente**, ela detecta notas únicas de valor alto, mas não soma dezenas de notas pequenas no mesmo mês (ex.: abastecer R$ 250 de combustível 40 vezes no mesmo mês parece `Classe 0` em cada nota isolada).
3. **Teto de Balanceamento (~25% de alertas):** Em `src/data_fetcher.py`, garante-se que cerca de **25,7% do dataset** seja rotulado como `Classe 1` (`percentil 74`) para que a rede neural treinada do zero não sofra com desbalanceamento extremo. Como consequência, o quarto mais caro de cada categoria sempre será sinalizado para conferência.
4. **Codificação Numérica Simples (_Ordinal Encoding_):** Para manter a arquitetura enxuta com **5 entradas** (`5 → 16 → 2`), partidos, estados e categorias são mapeados para números inteiros (`1, 2, 3...`). A camada oculta com `16 neurônios (tanh)` compensa boa parte disso criando curvas não-lineares, mas em sistemas de grande porte poderiam ser usadas colunas binárias (_One-Hot Encoding_).

---

## 8. O que são os Campos da Guia "Ficha Técnica" e o que a Atividade Pede?

A guia **"Ficha Técnica (Rede Neural)"** do site funciona como um **painel visual de apresentação** para demonstrar ao professor o cumprimento de cada item da diretriz da atividade:

| Campo na guia "Ficha Técnica" | O que significa de forma simples? | É pedido na atividade? |
| :--- | :--- | :--- |
| **Arquitetura `5 → 16 (tanh) → 2`** | Mostra o formato da rede neural: **5 entradas** (valor, tipo de despesa, mês, partido, UF), **16 neurônios na camada oculta** usando a função **Tangente Hiperbólica (`tanh`)**, e **2 saídas** (`0 = Dentro da média`, `1 = Com alerta`). | **SIM (Obrigatório — Itens 1 e 3 da atividade).** A playlist de referência ensina uma rede pequena com função Sigmoide; a atividade exige _adequar a arquitetura à problemática dos dados_ (5 entradas → 16 ocultos → 2 saídas) e _alterar a função de ativação_ (`tanh`). |
| **Queda do erro (Cross-Entropy Loss) `0.5716 → 0.1859`** | Mede o quanto a rede neural estava "errando" na Época 1 (`0.5716`) até a Época 2000 (`0.1859`, quando aprendeu os padrões). Quanto menor a _Loss_, mais treinada está a rede. | **SIM (Obrigatório — Item 1 da atividade).** Na playlist _"Rede neural do zero em Python"_, a função `loss` (Entropia Cruzada com Softmax) é parte central da estrutura base que prova que a rede está aprendendo. |
| **Acurácia no teste (`91,21%`) + Precisão (`96,44%`) e Recall** | Mostra a nota da rede neural fazendo uma "prova" com **71.110 notas fiscais que ela nunca viu durante o treino** (20% das 355.546 notas). | **Recomendado para apresentação (Item 3).** É como se comprova para o professor que a aplicação do banco de dados brasileiro na rede funcionou de verdade. |
| **Matriz de confusão (`VP`, `VN`, `FP`, `FN`)** | Detalha os acertos e erros nos **71.110 testes**:<br>• **VP (12.441):** Gastos acima do padrão que a IA acertou.<br>• **VN (52.418):** Gastos normais que a IA acertou.<br>• **FP (459):** Era normal, mas a IA deu alerta.<br>• **FN (5.792):** Era alto, mas passou despercebido. | **Complemento de avaliação.** Não é exigido nominalmente no texto curto da atividade, mas ajuda a explicar a acurácia e a precisão de 96,4% caso o professor pergunte onde a rede erra ou acerta. |
| **Simulador de inferência (`W1, b1, W2, b2`) + Ativações `h1..h16`** | Permite digitar um gasto fictício ao vivo e ver os **16 neurônios da camada oculta (`h1` a `h16`)** mudando de valor entre `-1` e `+1` (que é o intervalo matemático da função `tanh`). | **Complemento visual para apresentação (Item 3).** Serve para demonstrar ao vivo na entrega (30/10/2026) a rede funcionando na prática. |
| **Curva de aprendizado (`loss_curve.png`)** | O gráfico gerado pelo `matplotlib` mostrando a linha vermelha (erro caindo) e a linha azul (acerto subindo) ao longo das 2.000 épocas. | **SIM (Item 1 da atividade).** Na playlist de referência do professor, plotar a curva de erro ao longo das épocas é o fechamento clássico do código para provar a convergência do `backpropagation`. |
| **As 3 caixas de código no rodapé (`Z-Score`, `np.tanh`, `1 - a1²`)** | Mostram exatamente as 3 fórmulas matemáticas implementadas em [`src/nn_model.py`](src/nn_model.py):<br>1. Normalização dos dados brasileiros;<br>2. A nova função de ativação no `forward`;<br>3. A derivada `1 - a1**2` no `backpropagation`. | **SIM (Coração do Item 3 da atividade).** Quando o professor perguntar _"Onde vocês alteraram a função de ativação?"_, a resposta está exatamente nessas duas linhas (`np.tanh(z1)` na ida e `1.0 - np.power(a1, 2)` na volta do erro). |
