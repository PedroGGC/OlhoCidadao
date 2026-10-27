# Relatório de Auditoria Inteligente e Impacto Social (CEAP)

**Data da Auditoria:** 26/10/2026 às 11:53
**Motor de Detecção:** Rede Neural Multilayer Perceptron (`NnModel` 100% NumPy com ativação `tanh`)  
**Motor de Parecer Social:** Motor Analítico Cidadão Offline (Fallback Inteligente Baseado em Custos Públicos Oficiais)  

---

As classes são rótulos estatísticos, não comprovação de irregularidade. Custos são referências comparativas; a projeção de 60 meses é hipotética e não representa gasto observado.

## 1. Resumo Executivo da Auditoria na Base Integral

- **Gastos Classificados como Atípicos/Suspeitos (Classe 1):** `22370` notas fiscais auditadas
- **Volume Auditado nesses Casos (2025–2026):** `R$ 174.516.661,18`
- **Excedente nas Notas Acima da Média Histórica da Categoria:** `R$ 144.479.894,19`
- **Cenário hipotético: repetição do excedente por 60 meses:** **R$ 8.668.793.651,24**

---

## 2. Tradução em Custo de Oportunidade Social no Brasil (Projeção 5 Anos)

Caso o montante excedente identificado pela Rede Neural fosse realocado integralmente em políticas públicas essenciais, seria possível financiar:

| Área Prioritária | Indicador / Política Pública de Referência | Custo Unitário Médio | Quantidade Financiável (5 Anos) | Fonte Oficial |
| :--- | :--- | :---: | :---: | :--- |
| **Educação Básica** | Merendas Escolares (PNAE/FNDE) | R$ 1,50 | **5.779.195.767 refeições** | FNDE / MEC |
| **Educação Básica** | Kits Completos de Material Escolar | R$ 120,00 | **72.239.947 alunos atendidos** | FNDE / MEC |
| **Inclusão Digital** | Computadores para Escolas Públicas | R$ 2.800,00 | **3.095.997 equipamentos** | PNCP / Gov.br |
| **Saúde Pública (SUS)** | Consultas Médicas de Atenção Primária | R$ 120,00 | **72.239.947 consultas** | CISCOPAR (referência de credenciamento) |
| **Saúde Pública (SUS)** | Kits Mensais da Farmácia Básica | R$ 45,00 | **192.639.858 tratamentos mensais** | RENAME / MS |
| **Segurança Pública** | Diárias Operacionais de Patrulhamento/EPIs | R$ 350,00 | **24.767.981 diárias táticas** | Estimativa do projeto |
| **Assistência Social** | Cestas Básicas Familiares (estimativa assistencial) | R$ 350,00 | **24.767.981 famílias assistidas** | Estimativa do projeto |
| **Renda do Trabalho** | Salários Mínimos Equivalentes | R$ 1.518,00 | **5.710.667 salários mínimos** | Portal da Transparência |

---

## 3. Detalhamento dos 8 Casos Mais Críticos Detectados pela Rede Neural

### Caso #01 — STYVENSON VALENTIM (Senador) (N/I-RN)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `PLANALTO MARKETING & PUBLICIDADE LTDA` (CNPJ/CPF: `26.551.610/0001-77` | Competência: `05/2025`) | [Nota Fiscal Oficial na Câmara](https://www6g.senado.leg.br/transparencia/sen/)
- **Valor Líquido da Nota:** **R$ 175.000,00** *(Valor Base Aceitável na Categoria: R$ 3.505,93)*
- **Excesso Mensal Detectado:** `R$ 171.494,07` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 10.289.644,00**
- **Parecer de Retorno Social para o Estado (RN):**
  - **Educação em RN:** Financiaria **6.859.762 merendas escolares** (a R$ 1,50) ou **85.747 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em RN:** Garantiria **85.747 consultas médicas no SUS** ou **228.658 kits de medicamentos essenciais**.
  - **Segurança Pública em RN:** Custearia **29.398 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #02 — Aluisio Mendes (Dep. Federal) (REPUBLICANOS-MA)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Adriano L Silva Engenharia Serviços e Comércio` (CNPJ/CPF: `352.650.610/0016-5` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/2999/2025/8028441.pdf)
- **Valor Líquido da Nota:** **R$ 140.000,00** *(Valor Base Aceitável na Categoria: R$ 3.505,93)*
- **Excesso Mensal Detectado:** `R$ 136.494,07` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 8.189.644,00**
- **Parecer de Retorno Social para o Estado (MA):**
  - **Educação em MA:** Financiaria **5.459.762 merendas escolares** (a R$ 1,50) ou **68.247 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em MA:** Garantiria **68.247 consultas médicas no SUS** ou **181.992 kits de medicamentos essenciais**.
  - **Segurança Pública em MA:** Custearia **23.398 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #03 — Silas Câmara (Dep. Federal) (REPUBLICANOS-AM)
- **Categoria da Despesa (CEAP):** `LOCAÇÃO OU FRETAMENTO DE AERONAVES`
- **Fornecedor Declarado:** `AMAZONAVES TAXI AEREO LTDA` (CNPJ/CPF: `030.907.560/0016-7` | Competência: `02/2026`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/1286/2026/8065824.pdf)
- **Valor Líquido da Nota:** **R$ 149.500,00** *(Valor Base Aceitável na Categoria: R$ 19.434,51)*
- **Excesso Mensal Detectado:** `R$ 130.065,49` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 7.803.929,63**
- **Parecer de Retorno Social para o Estado (AM):**
  - **Educação em AM:** Financiaria **5.202.619 merendas escolares** (a R$ 1,50) ou **65.032 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em AM:** Garantiria **65.032 consultas médicas no SUS** ou **173.420 kits de medicamentos essenciais**.
  - **Segurança Pública em AM:** Custearia **22.296 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #04 — Sargento Gonçalves (Dep. Federal) (PL-RN)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Maricleydson Costa da Silva` (CNPJ/CPF: `173.404.470/0010-2` | Competência: `10/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/nota-fiscal-eletronica?ideDocumentoFiscal=7995308)
- **Valor Líquido da Nota:** **R$ 128.000,00** *(Valor Base Aceitável na Categoria: R$ 3.505,93)*
- **Excesso Mensal Detectado:** `R$ 124.494,07` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 7.469.644,00**
- **Parecer de Retorno Social para o Estado (RN):**
  - **Educação em RN:** Financiaria **4.979.762 merendas escolares** (a R$ 1,50) ou **62.247 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em RN:** Garantiria **62.247 consultas médicas no SUS** ou **165.992 kits de medicamentos essenciais**.
  - **Segurança Pública em RN:** Custearia **21.341 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #05 — Sargento Gonçalves (Dep. Federal) (PL-RN)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Maricleydson Costa da Silva` (CNPJ/CPF: `173.404.470/0010-2` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/nota-fiscal-eletronica?ideDocumentoFiscal=8035404)
- **Valor Líquido da Nota:** **R$ 126.400,00** *(Valor Base Aceitável na Categoria: R$ 3.505,93)*
- **Excesso Mensal Detectado:** `R$ 122.894,07` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 7.373.644,00**
- **Parecer de Retorno Social para o Estado (RN):**
  - **Educação em RN:** Financiaria **4.915.762 merendas escolares** (a R$ 1,50) ou **61.447 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em RN:** Garantiria **61.447 consultas médicas no SUS** ou **163.858 kits de medicamentos essenciais**.
  - **Segurança Pública em RN:** Custearia **21.067 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #06 — PROFESSORA DORINHA SEABRA (Senador) (N/I-TO)
- **Categoria da Despesa (CEAP):** `CONTRATAÇÃO DE CONSULTORIAS, ASSESSORIAS, PESQUISAS, TRABALHOS TÉCNICOS E OUTROS SERVIÇOS DE APOIO AO EXERCÍCIO DO MANDATO PARLAMENTAR`
- **Fornecedor Declarado:** `GRÁFICA E EDITORA SANTO EXPEDITO LTDA` (CNPJ/CPF: `09.015.122/0001-26` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www6g.senado.leg.br/transparencia/sen/)
- **Valor Líquido da Nota:** **R$ 110.000,00** *(Valor Base Aceitável na Categoria: R$ 1.954,13)*
- **Excesso Mensal Detectado:** `R$ 108.045,87` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 6.482.752,36**
- **Parecer de Retorno Social para o Estado (TO):**
  - **Educação em TO:** Financiaria **4.321.834 merendas escolares** (a R$ 1,50) ou **54.022 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em TO:** Garantiria **54.022 consultas médicas no SUS** ou **144.061 kits de medicamentos essenciais**.
  - **Segurança Pública em TO:** Custearia **18.522 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #07 — Marcos Soares (Dep. Federal) (PSDB-RJ)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Supraset Grafica e Editora Ltda` (CNPJ/CPF: `012.096.040/0010-5` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/nota-fiscal-eletronica?ideDocumentoFiscal=8037519)
- **Valor Líquido da Nota:** **R$ 110.000,00** *(Valor Base Aceitável na Categoria: R$ 3.505,93)*
- **Excesso Mensal Detectado:** `R$ 106.494,07` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 6.389.644,00**
- **Parecer de Retorno Social para o Estado (RJ):**
  - **Educação em RJ:** Financiaria **4.259.762 merendas escolares** (a R$ 1,50) ou **53.247 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em RJ:** Garantiria **53.247 consultas médicas no SUS** ou **141.992 kits de medicamentos essenciais**.
  - **Segurança Pública em RJ:** Custearia **18.256 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #08 — Antônia Lúcia (Dep. Federal) (MDB-AC)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `E.O.PEREIRA` (CNPJ/CPF: `236.578.190/0012-1` | Competência: `10/2026`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/2258/2026/8162426.pdf)
- **Valor Líquido da Nota:** **R$ 103.000,00** *(Valor Base Aceitável na Categoria: R$ 3.505,93)*
- **Excesso Mensal Detectado:** `R$ 99.494,07` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 5.969.644,00**
- **Parecer de Retorno Social para o Estado (AC):**
  - **Educação em AC:** Financiaria **3.979.762 merendas escolares** (a R$ 1,50) ou **49.747 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em AC:** Garantiria **49.747 consultas médicas no SUS** ou **132.658 kits de medicamentos essenciais**.
  - **Segurança Pública em AC:** Custearia **17.056 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

---

## 4. Parecer Analítico Consolidado

### 4.1 Diagnóstico Técnico por Categoria de Gasto (CEAP)

A inspeção automatizada pela Rede Neural Multilayer Perceptron (`NnModel`) evidencia que os maiores desvios financeiros concentram-se em rubricas de difícil mensuração física imediata (como divulgação de atividade parlamentar, consultorias técnicas, locação de veículos e passagens aéreas). Confira a distribuição do impacto acumulado em 5 anos por categoria entre os casos suspeitos:

| Categoria de Despesa (CEAP) | Ocorrências Atípicas | Impacto Projetado em 5 Anos |
| :--- | :---: | :---: |
| `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR` | 3107 | **R$ 3.590.296.584,01** |
| `MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE PARLAMENTAR` | 4164 | **R$ 1.660.958.746,23** |
| `LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES` | 2097 | **R$ 934.449.424,85** |
| `COMBUSTÍVEIS E LUBRIFICANTES` | 3174 | **R$ 595.882.782,14** |
| `LOCOMOÇÃO, HOSPEDAGEM, ALIMENTAÇÃO, COMBUSTÍVEIS E LUBRIFICANTES` | 1213 | **R$ 558.089.650,41** |
| `CONTRATAÇÃO DE CONSULTORIAS, ASSESSORIAS, PESQUISAS, TRABALHOS TÉCNICOS E OUTROS SERVIÇOS DE APOIO AO EXERCÍCIO DO MANDATO PARLAMENTAR` | 297 | **R$ 364.637.075,01** |
| `ALUGUEL DE IMÓVEIS PARA ESCRITÓRIO POLÍTICO, COMPREENDENDO DESPESAS CONCERNENTES A ELES` | 873 | **R$ 287.802.164,94** |
| `PASSAGENS AÉREAS, AQUÁTICAS E TERRESTRES NACIONAIS` | 733 | **R$ 121.148.772,77** |
| `SERVIÇO DE SEGURANÇA PRESTADO POR EMPRESA ESPECIALIZADA` | 221 | **R$ 101.783.290,28** |
| `HOSPEDAGEM ,EXCETO DO PARLAMENTAR NO DISTRITO FEDERAL` | 893 | **R$ 99.657.096,01** |
| `TELEFONIA` | 1770 | **R$ 94.375.891,88** |
| `LOCAÇÃO OU FRETAMENTO DE AERONAVES` | 33 | **R$ 75.250.577,78** |
| `AQUISIÇÃO DE MATERIAL DE CONSUMO PARA USO NO ESCRITÓRIO POLÍTICO, INCLUSIVE AQUISIÇÃO OU LOCAÇÃO DE SOFTWARE, DESPESAS POSTAIS, AQUISIÇÃO DE PUBLICAÇÕES, LOCAÇÃO DE MÓVEIS E DE EQUIPAMENTOS` | 248 | **R$ 74.461.921,68** |
| `SERVIÇO DE TÁXI, PEDÁGIO E ESTACIONAMENTO` | 1481 | **R$ 56.548.970,29** |
| `PASSAGEM AÉREA - REEMBOLSO` | 23 | **R$ 22.452.336,36** |
| `FORNECIMENTO DE ALIMENTAÇÃO DO PARLAMENTAR` | 1719 | **R$ 11.997.289,91** |
| `SERVIÇOS POSTAIS` | 24 | **R$ 5.890.548,78** |
| `PASSAGEM AÉREA - RPA` | 31 | **R$ 4.851.200,43** |
| `PASSAGENS TERRESTRES, MARÍTIMAS OU FLUVIAIS` | 235 | **R$ 3.896.203,56** |
| `ASSINATURA DE PUBLICAÇÕES` | 28 | **R$ 3.037.722,36** |
| `PARTICIPAÇÃO EM CURSO, PALESTRA OU EVENTO SIMILAR` | 2 | **R$ 724.775,23** |
| `CONSULTORIAS, PESQUISAS E TRABALHOS TÉCNICOS` | 1 | **R$ 384.546,04** |
| `LOCAÇÃO OU FRETAMENTO DE EMBARCAÇÕES` | 1 | **R$ 158.361,82** |
| `AQUISIÇÃO DE TOKENS E CERTIFICADOS DIGITAIS` | 2 | **R$ 57.718,48** |

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
