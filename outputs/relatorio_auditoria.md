# Relatório de Auditoria Inteligente e Impacto Social (CEAP)

**Data da Auditoria:** 22/10/2026 às 13:44  
**Motor de Detecção:** Rede Neural Multilayer Perceptron (`NnModel` 100% NumPy com ativação `tanh`)  
**Motor de Parecer Social:** Motor Analítico Cidadão Offline (Fallback Inteligente Baseado em Custos Públicos Oficiais)  

---

## 1. Resumo Executivo da Auditoria no Conjunto de Teste

- **Gastos Classificados como Atípicos/Suspeitos (Classe 1):** `12900` notas fiscais auditadas
- **Volume Mensal Auditado nesses Casos:** `R$ 56.380.959,97`
- **Excesso Mensal Acima da Média Aceitável por Categoria:** `R$ 45.721.781,23`
- **Projeção Acumulada de Desperdício em 5 Anos (60 meses):** **R$ 2.743.306.873,81**

---

## 2. Tradução em Custo de Oportunidade Social no Brasil (Projeção 5 Anos)

Caso o montante excedente identificado pela Rede Neural fosse realocado integralmente em políticas públicas essenciais, seria possível financiar:

| Área Prioritária | Indicador / Política Pública de Referência | Custo Unitário Médio | Quantidade Financiável (5 Anos) | Fonte Oficial |
| :--- | :--- | :---: | :---: | :--- |
| **Educação Básica** | Merendas Escolares (PNAE/FNDE) | R$ 1,50 | **1.828.871.249 refeições** | FNDE / MEC |
| **Educação Básica** | Kits Completos de Material Escolar | R$ 120,00 | **22.860.890 alunos atendidos** | FNDE / MEC |
| **Inclusão Digital** | Computadores para Escolas Públicas | R$ 2.800,00 | **979.752 equipamentos** | PNCP / Gov.br |
| **Saúde Pública (SUS)** | Consultas Médicas de Atenção Primária | R$ 65,00 | **42.204.721 consultas** | DataSUS / SIGTAP |
| **Saúde Pública (SUS)** | Kits Mensais da Farmácia Básica | R$ 45,00 | **60.962.374 tratamentos mensais** | RENAME / MS |
| **Segurança Pública** | Diárias Operacionais de Patrulhamento/EPIs | R$ 420,00 | **6.531.683 diárias táticas** | SENASP / MJSP |
| **Assistência Social** | Cestas Básicas Familiares (Ref. DIEESE) | R$ 750,00 | **3.657.742 famílias assistidas** | DIEESE |
| **Renda do Trabalho** | Salários Mínimos Equivalentes | R$ 1.518,00 | **1.807.185 salários mínimos** | Portal da Transparência |

---

## 3. Detalhamento dos 8 Casos Mais Críticos Detectados pela Rede Neural

### Caso #01 — Aluisio Mendes (Dep. Federal) (REPUBLICANOS-MA)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Adriano L Silva Engenharia Serviços e Comércio` (CNPJ/CPF: `352.650.610/0016-5` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/2999/2025/8028441.pdf)
- **Valor Líquido da Nota:** **R$ 140.000,00** *(Valor Base Aceitável na Categoria: R$ 2.271,73)*
- **Excesso Mensal Detectado:** `R$ 137.728,27` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 8.263.696,21**
- **Parecer de Retorno Social para o Estado (MA):**
  - **Educação em MA:** Financiaria **5.509.130 merendas escolares** (a R$ 1,50) ou **68.864 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em MA:** Garantiria **127.133 consultas médicas no SUS** ou **183.637 kits de medicamentos essenciais**.
  - **Segurança Pública em MA:** Custearia **19.675 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #02 — Marcos Soares (Dep. Federal) (PSDB-RJ)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Supraset Grafica e Editora Ltda` (CNPJ/CPF: `012.096.040/0010-5` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/nota-fiscal-eletronica?ideDocumentoFiscal=8037519)
- **Valor Líquido da Nota:** **R$ 110.000,00** *(Valor Base Aceitável na Categoria: R$ 2.271,73)*
- **Excesso Mensal Detectado:** `R$ 107.728,27` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 6.463.696,21**
- **Parecer de Retorno Social para o Estado (RJ):**
  - **Educação em RJ:** Financiaria **4.309.130 merendas escolares** (a R$ 1,50) ou **53.864 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em RJ:** Garantiria **99.441 consultas médicas no SUS** ou **143.637 kits de medicamentos essenciais**.
  - **Segurança Pública em RJ:** Custearia **15.389 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #03 — Antônia Lúcia (Dep. Federal) (MDB-AC)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `E.O.PEREIRA` (CNPJ/CPF: `236.578.190/0012-1` | Competência: `10/2026`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/2258/2026/8162426.pdf)
- **Valor Líquido da Nota:** **R$ 103.000,00** *(Valor Base Aceitável na Categoria: R$ 2.271,73)*
- **Excesso Mensal Detectado:** `R$ 100.728,27` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 6.043.696,21**
- **Parecer de Retorno Social para o Estado (AC):**
  - **Educação em AC:** Financiaria **4.029.130 merendas escolares** (a R$ 1,50) ou **50.364 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em AC:** Garantiria **92.979 consultas médicas no SUS** ou **134.304 kits de medicamentos essenciais**.
  - **Segurança Pública em AC:** Custearia **14.389 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #04 — PROFESSORA DORINHA SEABRA (Senador) (UNIÃO-TO)
- **Categoria da Despesa (CEAP):** `CONTRATAÇÃO DE CONSULTORIAS, ASSESSORIAS, PESQUISAS, TRABALHOS TÉCNICOS E OUTROS SERVIÇOS DE APOIO AO EXERCÍCIO DO MANDATO PARLAMENTAR`
- **Fornecedor Declarado:** `GRÁFICA E EDITORA SANTO EXPEDITO LTDA` (CNPJ/CPF: `09.015.122/0001-26` | Competência: `09/2025`) | [Nota Fiscal Oficial na Câmara](https://www6g.senado.leg.br/transparencia/sen/)
- **Valor Líquido da Nota:** **R$ 100.200,00** *(Valor Base Aceitável na Categoria: R$ 1.106,70)*
- **Excesso Mensal Detectado:** `R$ 99.093,30` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 5.945.597,80**
- **Parecer de Retorno Social para o Estado (TO):**
  - **Educação em TO:** Financiaria **3.963.731 merendas escolares** (a R$ 1,50) ou **49.546 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em TO:** Garantiria **91.470 consultas médicas no SUS** ou **132.124 kits de medicamentos essenciais**.
  - **Segurança Pública em TO:** Custearia **14.156 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #05 — Amanda Gentil (Dep. Federal) (PP-MA)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `INDUSTRIA GRAFICA E EDITORA SAO PAULO LTDA` (CNPJ/CPF: `197.686.560/0018-7` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/nota-fiscal-eletronica?ideDocumentoFiscal=8037448)
- **Valor Líquido da Nota:** **R$ 82.000,10** *(Valor Base Aceitável na Categoria: R$ 2.271,73)*
- **Excesso Mensal Detectado:** `R$ 79.728,37` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 4.783.702,21**
- **Parecer de Retorno Social para o Estado (MA):**
  - **Educação em MA:** Financiaria **3.189.134 merendas escolares** (a R$ 1,50) ou **39.864 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em MA:** Garantiria **73.595 consultas médicas no SUS** ou **106.304 kits de medicamentos essenciais**.
  - **Segurança Pública em MA:** Custearia **11.389 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #06 — Silvye Alves (Dep. Federal) (UNIÃO-GO)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Mais Brasil Publicidade e Propaganda LTDA` (CNPJ/CPF: `140.531.780/0010-5` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/3530/2025/8032568.pdf)
- **Valor Líquido da Nota:** **R$ 78.360,00** *(Valor Base Aceitável na Categoria: R$ 2.271,73)*
- **Excesso Mensal Detectado:** `R$ 76.088,27` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 4.565.296,21**
- **Parecer de Retorno Social para o Estado (GO):**
  - **Educação em GO:** Financiaria **3.043.530 merendas escolares** (a R$ 1,50) ou **38.044 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em GO:** Garantiria **70.235 consultas médicas no SUS** ou **101.451 kits de medicamentos essenciais**.
  - **Segurança Pública em GO:** Custearia **10.869 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #07 — Luciano Amaral (Dep. Federal) (PSD-AL)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `INDÚSTRIA GRÁFICA JARAGUÁ LTDA` (CNPJ/CPF: `108.030.130/0015-2` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/3621/2025/8032637.pdf)
- **Valor Líquido da Nota:** **R$ 71.400,00** *(Valor Base Aceitável na Categoria: R$ 2.271,73)*
- **Excesso Mensal Detectado:** `R$ 69.128,27` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 4.147.696,21**
- **Parecer de Retorno Social para o Estado (AL):**
  - **Educação em AL:** Financiaria **2.765.130 merendas escolares** (a R$ 1,50) ou **34.564 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em AL:** Garantiria **63.810 consultas médicas no SUS** ou **92.171 kits de medicamentos essenciais**.
  - **Segurança Pública em AL:** Custearia **9.875 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #08 — Paulo Lemos (Dep. Federal) (PT-AP)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Agir LTDA` (CNPJ/CPF: `468.799.140/0017-5` | Competência: `05/2026`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/3795/2026/8110301.pdf)
- **Valor Líquido da Nota:** **R$ 70.000,00** *(Valor Base Aceitável na Categoria: R$ 2.271,73)*
- **Excesso Mensal Detectado:** `R$ 67.728,27` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 4.063.696,21**
- **Parecer de Retorno Social para o Estado (AP):**
  - **Educação em AP:** Financiaria **2.709.130 merendas escolares** (a R$ 1,50) ou **33.864 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em AP:** Garantiria **62.518 consultas médicas no SUS** ou **90.304 kits de medicamentos essenciais**.
  - **Segurança Pública em AP:** Custearia **9.675 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

---

## 4. Parecer Analítico Consolidado

### 4.1 Diagnóstico Técnico por Categoria de Gasto (CEAP)

A inspeção automatizada pela Rede Neural Multilayer Perceptron (`NnModel`) evidencia que os maiores desvios financeiros concentram-se em rubricas de difícil mensuração física imediata (como divulgação de atividade parlamentar, consultorias técnicas, locação de veículos e passagens aéreas). Confira a distribuição do impacto acumulado em 5 anos por categoria entre os casos suspeitos:

| Categoria de Despesa (CEAP) | Ocorrências Atípicas | Impacto Projetado em 5 Anos |
| :--- | :---: | :---: |
| `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR` | 1543 | **R$ 1.158.654.832,83** |
| `MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE PARLAMENTAR` | 2145 | **R$ 527.945.961,02** |
| `LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES` | 569 | **R$ 229.633.584,69** |
| `COMBUSTÍVEIS E LUBRIFICANTES` | 3467 | **R$ 163.169.465,77** |
| `LOCOMOÇÃO, HOSPEDAGEM, ALIMENTAÇÃO, COMBUSTÍVEIS E LUBRIFICANTES` | 749 | **R$ 142.231.234,35** |
| `CONTRATAÇÃO DE CONSULTORIAS, ASSESSORIAS, PESQUISAS, TRABALHOS TÉCNICOS E OUTROS SERVIÇOS DE APOIO AO EXERCÍCIO DO MANDATO PARLAMENTAR` | 304 | **R$ 126.248.994,13** |
| `PASSAGEM AÉREA - SIGEPA` | 1271 | **R$ 115.594.724,70** |
| `ALUGUEL DE IMÓVEIS PARA ESCRITÓRIO POLÍTICO, COMPREENDENDO DESPESAS CONCERNENTES A ELES` | 398 | **R$ 85.390.833,88** |
| `HOSPEDAGEM ,EXCETO DO PARLAMENTAR NO DISTRITO FEDERAL` | 702 | **R$ 36.485.898,85** |
| `TELEFONIA` | 933 | **R$ 32.297.911,46** |
| `LOCAÇÃO OU FRETAMENTO DE AERONAVES` | 34 | **R$ 32.200.419,36** |
| `SERVIÇO DE SEGURANÇA PRESTADO POR EMPRESA ESPECIALIZADA` | 78 | **R$ 28.757.761,24** |
| `AQUISIÇÃO DE MATERIAL DE CONSUMO PARA USO NO ESCRITÓRIO POLÍTICO, INCLUSIVE AQUISIÇÃO OU LOCAÇÃO DE SOFTWARE, DESPESAS POSTAIS, AQUISIÇÃO DE PUBLICAÇÕES, LOCAÇÃO DE MÓVEIS E DE EQUIPAMENTOS` | 101 | **R$ 17.733.464,24** |
| `PASSAGENS AÉREAS, AQUÁTICAS E TERRESTRES NACIONAIS` | 88 | **R$ 16.031.496,94** |
| `SERVIÇO DE TÁXI, PEDÁGIO E ESTACIONAMENTO` | 352 | **R$ 12.586.598,28** |
| `PASSAGEM AÉREA - REEMBOLSO` | 8 | **R$ 7.532.435,12** |
| `PASSAGEM AÉREA - RPA` | 56 | **R$ 7.058.820,71** |
| `ASSINATURA DE PUBLICAÇÕES` | 22 | **R$ 1.539.923,87** |
| `SERVIÇOS POSTAIS` | 14 | **R$ 680.319,24** |
| `PASSAGENS TERRESTRES, MARÍTIMAS OU FLUVIAIS` | 34 | **R$ 631.661,29** |
| `FORNECIMENTO DE ALIMENTAÇÃO DO PARLAMENTAR` | 30 | **R$ 520.489,77** |
| `PARTICIPAÇÃO EM CURSO, PALESTRA OU EVENTO SIMILAR` | 1 | **R$ 366.755,91** |
| `AQUISIÇÃO DE TOKENS E CERTIFICADOS DIGITAIS` | 1 | **R$ 13.286,13** |

### 4.2 Bases de Dados Públicos Oficiais Incluídas no Projeto (`data/`)

| Arquivo CSV Local | Fonte Pública Oficial | Conteúdo |
| :--- | :--- | :--- |
| `data/raw_despesas_camara.csv` | [Dados Abertos da Câmara dos Deputados](https://dadosabertos.camara.leg.br/) | Notas fiscais reais da CEAP (fornecedor, CNPJ, valor e link do documento fiscal) |
| `data/deputados_federais_camara.csv` | [API Câmara — Deputados](https://dadosabertos.camara.leg.br/api/v2/deputados) | Cadastro oficial dos deputados federais em exercício |
| `data/partidos_politicos_camara.csv` | [API Câmara — Partidos](https://dadosabertos.camara.leg.br/api/v2/partidos) | Lista oficial das legendas partidárias na Câmara |
| `data/estados_ibge_oficial.csv` | [API de Localidades do IBGE](https://servicodados.ibge.gov.br/api/v1/localidades/estados) | Tabela oficial das 27 UFs e macrorregiões brasileiras |
| `data/indicadores_custos_publicos_br.csv` | FNDE/MEC, DataSUS/SIGTAP, DIEESE e SENASP | Custos públicos unitários para conversão de impacto social |

### 4.3 Conclusão e Recomendações de Controle Social

1. **Teto Paramétrico por Rubrica:** A aplicação de limites estatísticos dinâmicos ($\mu + 1.2\sigma$) combinada à classificação não-linear via Rede Neural permite sinalizar preventivamente notas fiscais com forte descolamento do padrão histórico da Câmara dos Deputados.
2. **Auditoria de Fornecedores Recorrentes:** Recomenda-se cruzamento automático de CNPJ para despesas de *Divulgação Parlamentar* e *Consultorias Técnicas* que ultrapassem 2 desvios padrão da média mensal.
3. **Transparência Cidadã Ativa:** Traduzir cifras orçamentárias em indicadores tangíveis — como merendas escolares do PNAE, consultas de atenção primária do SUS e patrulhamento comunitário estadual — aproxima o controle da administração pública da realidade cotidiana do cidadão brasileiro.
