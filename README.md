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
> - 📺 [Redes Neurais — Conceitos Básicos (Playlist)](https://www.youtube.com/watch?v=ebToEXQFCo4&list=PLQH6T1jnIb5J7vugBAauJsFU8Qgvuf-4X)
> - 📺 [Rede neural do zero em Python — Referência do Projeto (Playlist)](https://www.youtube.com/watch?v=OGcOH9damt0&list=PLQH6T1jnIb5I1s7chxtL24ZuX4g-PA7uK)

### 📋 Matriz de Conformidade com o Escopo da Atividade

| Item do Escopo Oficial | Diretriz Exigida pelo Professor | Como o Projeto OlhoCidadão Cumpre | Arquivo de Referência | Status |
| :--- | :--- | :--- | :--- | :---: |
| **1. Desenvolvimento Base** | *"Desenvolver a estrutura de rede neural criada e explicada na playlist de referência 'Rede neural do zero em Python'."* | Implementação da classe `NnModel` construída **100% em NumPy do zero**, sem scikit-learn, pytorch ou tensorflow. Contém pesos Xavier, `forward`, `loss` analítica por Entropia Cruzada, `backpropagation` por Regra da Cadeia e loop `fit` com exportação de curva de perda. | [`src/nn_model.py`](src/nn_model.py) | **100% Conforme** |
| **2. Escolha do Dataset** | *"Escolher um banco de dados de qualquer área de interesse, com a condição obrigatória de que sejam dados de origem do Brasil."* | Consumo de dados públicos oficiais brasileiros da **Câmara dos Deputados (CEAP)**, **Senado Federal (CEAPS)**, **IBGE** e **Tesouro Nacional/SICONFI**. Inclui dump integral de 326 mil notas fiscais e amostra balanceada de 10.000 notas fiscais reais. | [`src/data_fetcher.py`](src/data_fetcher.py) e [`data/`](data/) | **100% Conforme** |
| **3. Customização & Aplicação** | *"Aplicar o banco de dados escolhido na rede neural desenvolvida. Vocês devem alterar a função de ativação e adequar a arquitetura da rede de acordo com a problemática dos dados selecionados."* | • **Adequação da Arquitetura:** Camada de entrada expandida de 2 entradas sintéticas (`make_moons`) para **5 features reais** de gastos parlamentares (`valorLiquido`, `codTipoDespesa`, `mes`, `codPartido`, `codUF`).<br>• **Alteração da Ativação:** Substituição da sigmoide clássica pela **Tangente Hiperbólica ($\tanh$)**, com derivada analítica exata $f'(a) = 1 - a^2$ no backpropagation, operando em harmonia com normalização Z-Score. | [`src/nn_model.py`](src/nn_model.py#L38-L102) | **100% Conforme** |

---

## 2. Detalhamento Técnico dos 3 Requisitos da Atividade

### Requisito 1: Desenvolvimento Base (Rede Neural do Zero em Python/NumPy)
O arquivo [`src/nn_model.py`](src/nn_model.py) implementa a classe `NnModel` utilizando exclusivamente álgebra linear vetorial em `numpy` (sem `scikit-learn`, `pytorch` ou `tensorflow`), contemplando toda a estrutura clássica da playlist de referência (*"Rede neural do zero em Python"*):
- `__init__(self, X, y, n_hidden=16, lr=0.05)`: Inicialização dinâmica das matrizes de pesos `W1`, `b1`, `W2` e `b2` via método de Xavier/Glorot.
- `forward(self, X)`: Propagação direta da camada de entrada para a camada oculta e cálculo de probabilidades na saída via função `softmax` numericamente estável.
- `loss(self, y_pred)`: Cálculo exato da perda por **Entropia Cruzada Categórica (*Cross-Entropy Loss*)**:
  $$\mathcal{L}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} \log\left(\hat{y}_{i, y_i}\right)$$
- `backpropagation(self, X, y)`: Cálculo analítico dos gradientes `dW2`, `db2`, `dW1`, `db1` aplicando a Regra da Cadeia e atualização por Gradiente Descendente.
- `fit(self, epochs=2000)` e `predict(self, X)`: Loop de treinamento completo com exportação automática da curva de aprendizado em `outputs/loss_curve.png`.

### Requisito 2: Escolha do Dataset (Origem Brasil — Câmara dos Deputados & Senado Federal)
O módulo [`src/data_fetcher.py`](src/data_fetcher.py) consome e consolida dados públicos oficiais massivos da **Cota para o Exercício da Atividade Parlamentar (CEAP)** da Câmara dos Deputados e do **CEAPS** do Senado Federal, além de cadastros do IBGE e Tesouro Nacional:
- **Base Completa & Dumps Massivos Integrados:** O repositório já inclui a base completa histórica de notas fiscais da Câmara dos Deputados (`cota_parlamentar_camara_completa.csv` — **326.304 notas fiscais**, ~70.4 MB) e a base integral dos senadores (`despesas_senadores_ceaps_completo.csv` — **39.652 notas fiscais**, ~9.9 MB), cobrindo todos os 513 deputados federais e 81 senadores da República.
- **Amostra Estratificada para Treinamento (10.000 notas fiscais):** Para garantir treinamento ágil em qualquer CPU convencional sem lentidão de I/O em disco, o pipeline estrutura uma amostra balanceada e representativa de **10.000 notas fiscais reais** salvas em `data/raw_despesas_camara.csv` e `data/processed_dataset.csv`.
- **Mecanismo de Resiliência Offline/Timeout:** Caso os portais governamentais apresentem indisponibilidade ou limite de taxa (*rate limiting*), o pipeline carrega diretamente os dados locais já baixados ou aciona o gerador estatístico fundamentado nas métricas reais da CEAP.
- **Rotulagem Estatística (`y`) e Normalização Z-Score:** Calcula a média ($\mu_c$) e o desvio padrão ($\sigma_c$) por categoria de despesa (`codTipoDespesa`). Notas cujo `valorLiquido` ultrapassam $\mu_c + 1.2\sigma_c$ recebem `y = 1` (**Gasto Atípico/Suspeito**); as demais recebem `y = 0` (**Gasto Normal**). Para evitar *overflow* exponencial na rede, aplica-se a normalização Z-Score em NumPy puro:
  $$X_{\text{norm}} = \frac{X - \mu}{\sigma + 10^{-8}}$$

### Requisito 3: Customização da Arquitetura e da Função de Ativação
Em relação ao modelo didático clássico de 2 entradas sintéticas (`make_moons`) com ativação sigmoide, foram realizadas duas customizações fundamentais em [`src/nn_model.py`](src/nn_model.py):
1. **Adequação da Camada de Entrada (5 Features Reais):** A matriz `X` alimenta dinamicamente `n_input = X.shape[1] = 5` neurônios de entrada correspondentes a:
   1. `valorLiquido` (float em R$)
   2. `codTipoDespesa` (int mapeado da rubrica da CEAP)
   3. `mes` (int de 1 a 12)
   4. `codPartido` (int mapeado da sigla partidária)
   5. `codUF` (int mapeado da Unidade Federativa do parlamentar)
2. **Customização da Função de Ativação na Camada Oculta (`np.tanh`):** Substituiu-se a sigmoide padrão pela **Tangente Hiperbólica ($\tanh$)** (com suporte adicional a *Leaky ReLU*):
   $$a^{(1)} = \tanh\left(z^{(1)}\right) = \frac{e^{z^{(1)}} - e^{-z^{(1)}}}{e^{z^{(1)}} + e^{-z^{(1)}}}$$
   No método `backpropagation`, a retropropagação do erro aplica a **derivada analítica exata da Tangente Hiperbólica**:
   $$\frac{\partial \tanh\left(z^{(1)}\right)}{\partial z^{(1)}} = 1 - \left(a^{(1)}\right)^2$$
   Como a $\tanh$ é centrada em zero no intervalo $(-1, 1)$, ela opera em sinergia direta com a normalização Z-Score, eliminando o problema de *zig-zag* nos gradientes e acelerando a convergência.

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

| Arquivo em `data/` | Fonte Oficial | Qtd. Registros | Tamanho | Conteúdo e Finalidade |
| :--- | :--- | :---: | :---: | :--- |
| **`cota_parlamentar_camara_completa.csv`** | [Câmara dos Deputados](https://dadosabertos.camara.leg.br/) | **326.304** | **~70.4 MB** | **DUMP COMPLETO:** Todas as despesas e notas fiscais declaradas por todos os 513 deputados federais (com CNPJ, fornecedor, valor líquido e link da NF-e). |
| **`despesas_senadores_ceaps_completo.csv`** | [Senado Federal (CEAPS)](https://dadosabertos.senado.leg.br/) | **39.652** | **~9.9 MB** | **DUMP COMPLETO:** Todas as notas fiscais dos 81 senadores da República em exercício nos exercícios recentes de 2025 e 2026. |
| **`deputados_historico_completo_camara.csv`** | [API Dados Abertos](https://dadosabertos.camara.leg.br/api/v2/deputados) | **7.889** | **~1.3 MB** | Base histórica com todos os parlamentares federais que já exerceram mandato na história do Brasil. |
| **`processed_dataset.csv`** | Amostragem Estratificada + Engenharia de Features | **10.000** | **~3.2 MB** | Dataset pré-processado com as 5 features normalizadas via Z-Score e rótulo supervisionado `y \in \{0, 1\}` para treino e teste da MLP. |
| **`raw_despesas_camara.csv`** | Extração Cruzada (Deputados + Senadores) | **10.000** | **~2.8 MB** | Amostra estratificada preservando nomes de parlamentares, partidos, categorias originais e URLs das notas fiscais. |
| **`prefeituras_municipios_tesouro_ibge.csv`** | [IBGE](https://servicodados.ibge.gov.br/) & [Tesouro Nacional / SICONFI](https://siconfi.tesouro.gov.br/) | **5.570** | **~326 KB** | Relação de todos os 5.570 municípios e capitais brasileiras, incluindo CNPJs de prefeituras, estados e estimativas populacionais. |
| **`deputados_federais_camara.csv`** | [API da Câmara](https://dadosabertos.camara.leg.br/api/v2/deputados) | **513** | **~122 KB** | Cadastro completo dos 513 deputados da 57ª legislatura em exercício (foto oficial, partido, UF, e-mail). |
| **`senadores_federais_completo.csv`** | [API do Senado](https://adm.senado.gov.br/) | **81** | **~18 KB** | Cadastro dos 81 senadores federais com links oficiais e gabinetes. |
| **`estados_ibge_oficial.csv`** | [API Localidades IBGE](https://servicodados.ibge.gov.br/) | **27** | **~2.5 KB** | Tabela de referência das 27 Unidades Federativas e 5 Macrorregiões do Brasil. |
| **`partidos_politicos_camara.csv`** | [API Câmara](https://dadosabertos.camara.leg.br/) | **Legendas** | **~2.0 KB** | Mapeamento oficial de legendas partidárias brasileiras. |
| **`indicadores_custos_publicos_br.csv`** | FNDE/PNAE, DataSUS/SIGTAP, DIEESE, SENASP | **Tabela** | **~1.5 KB** | Custos unitários públicos auditados para o cálculo de custo de oportunidade social. |

> [!NOTE]
> **Por que esses dados massivos estão versionados no repositório?**  
> Para garantir **100% de autonomia e reprodutibilidade imediata**. Ao clonar o projeto, o avaliador ou usuário tem à disposição o ecossistema integral de dados sem depender de conexões externas lentas ou de instabilidades momentâneas nas APIs governamentais.

> [!WARNING]
> **Aviso de Arquivos Grandes no GitHub (> 50 MB):**  
> O arquivo `cota_parlamentar_camara_completa.csv` possui **~70.4 MB**, o que dispara um aviso automático do GitHub recomendando o uso do *Git LFS (Large File Storage)* para arquivos acima de 50 MB. O GitHub aceita arquivos de até 100 MB em commits padrão. Caso essa base venha a ser expandida no futuro além de 100 MB, utilize o Git LFS ou ignore a base bruta completa no `.gitignore`.

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
*(Se não configurar a chave, o projeto funcionará 100% offline usando o motor analítico matemático embutido).*

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

*(Também é possível rodar apenas o pipeline Python diretamente no terminal com `python main.py`).*

### O que você encontrará na Aplicação Web (`pnpm dev`) e em `outputs/`:
1. **Dashboard Web Interativo (`http://localhost:5173`):**
   - **Simulador da Rede Neural em Tempo Real:** Executa o *Forward Pass* com os pesos reais treinados (`W1`, `b1`, `W2`, `b2`), normalização Z-Score e ativação `tanh(z)` diretamente no navegador, exibindo as ativações dos 16 neurônios ocultos ao vivo.
   - **Painel de Custo de Oportunidade Social (5 Anos):** Conversão automática em merendas escolares (PNAE), kits escolares, computadores, consultas no SUS, medicamentos básicos e diárias de segurança.
   - **Tabela de Notas Fiscais Reais Auditadas:** Filtros por classificação (`Atípico` vs `Normal`), UF e busca textual por deputado, fornecedor ou CNPJ, com link direto para o PDF oficial da nota fiscal no portal da Câmara dos Deputados.
2. `outputs/loss_curve.png`: Gráfico comprovando a convergência da Rede Neural (`NnModel`).
3. `outputs/relatorio_auditoria.md`: Relatório detalhado dos gastos atípicos detectados e seu impacto social em 5 anos.
