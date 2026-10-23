# Relatório de Auditoria Inteligente e Impacto Social (CEAP)

**Data da Auditoria:** 22/10/2026 às 15:15
**Motor de Detecção:** Rede Neural Multilayer Perceptron (`NnModel` 100% NumPy com ativação `tanh`)  
**Motor de Parecer Social:** Motor Analítico Cidadão Offline (Fallback Inteligente Baseado em Custos Públicos Oficiais)  

---

As classes são rótulos estatísticos, não comprovação de irregularidade. Custos são referências comparativas; a projeção de 60 meses é hipotética e não representa gasto observado.

## 1. Resumo Executivo da Auditoria na Base Integral

- **Gastos Classificados como Atípicos/Suspeitos (Classe 1):** `47758` notas fiscais auditadas
- **Volume Auditado nesses Casos (2025–2026):** `R$ 225.395.363,45`
- **Excesso Mensal Acima da Média Aceitável por Categoria:** `R$ 176.658.034,69`
- **Cenário hipotético: repetição do excedente por 60 meses:** **R$ 10.599.482.081,13**

---

## 2. Tradução em Custo de Oportunidade Social no Brasil (Projeção 5 Anos)

Caso o montante excedente identificado pela Rede Neural fosse realocado integralmente em políticas públicas essenciais, seria possível financiar:

| Área Prioritária | Indicador / Política Pública de Referência | Custo Unitário Médio | Quantidade Financiável (5 Anos) | Fonte Oficial |
| :--- | :--- | :---: | :---: | :--- |
| **Educação Básica** | Merendas Escolares (PNAE/FNDE) | R$ 1,50 | **7.066.321.387 refeições** | FNDE / MEC |
| **Educação Básica** | Kits Completos de Material Escolar | R$ 120,00 | **88.329.017 alunos atendidos** | FNDE / MEC |
| **Inclusão Digital** | Computadores para Escolas Públicas | R$ 2.800,00 | **3.785.529 equipamentos** | PNCP / Gov.br |
| **Saúde Pública (SUS)** | Consultas Médicas de Atenção Primária | R$ 120,00 | **88.329.017 consultas** | CISCOPAR (referência de credenciamento) |
| **Saúde Pública (SUS)** | Kits Mensais da Farmácia Básica | R$ 45,00 | **235.544.046 tratamentos mensais** | RENAME / MS |
| **Segurança Pública** | Diárias Operacionais de Patrulhamento/EPIs | R$ 350,00 | **30.284.234 diárias táticas** | Estimativa do projeto |
| **Assistência Social** | Cestas Básicas Familiares (estimativa assistencial) | R$ 350,00 | **30.284.234 famílias assistidas** | Estimativa do projeto |
| **Renda do Trabalho** | Salários Mínimos Equivalentes | R$ 1.518,00 | **6.982.531 salários mínimos** | Portal da Transparência |

---

## 3. Detalhamento dos 8 Casos Mais Críticos Detectados pela Rede Neural

### Caso #01 — STYVENSON VALENTIM (Senador) (N/I-RN)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `PLANALTO MARKETING & PUBLICIDADE LTDA` (CNPJ/CPF: `26.551.610/0001-77` | Competência: `05/2025`) | [Nota Fiscal Oficial na Câmara](https://www6g.senado.leg.br/transparencia/sen/)
- **Valor Líquido da Nota:** **R$ 175.000,00** *(Valor Base Aceitável na Categoria: R$ 3.557,11)*
- **Excesso Mensal Detectado:** `R$ 171.442,89` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 10.286.573,30**
- **Parecer de Retorno Social para o Estado (RN):**
  - **Educação em RN:** Financiaria **6.857.715 merendas escolares** (a R$ 1,50) ou **85.721 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em RN:** Garantiria **85.721 consultas médicas no SUS** ou **228.590 kits de medicamentos essenciais**.
  - **Segurança Pública em RN:** Custearia **29.390 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #02 — Aluisio Mendes (Dep. Federal) (REPUBLICANOS-MA)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Adriano L Silva Engenharia Serviços e Comércio` (CNPJ/CPF: `352.650.610/0016-5` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/2999/2025/8028441.pdf)
- **Valor Líquido da Nota:** **R$ 140.000,00** *(Valor Base Aceitável na Categoria: R$ 3.557,11)*
- **Excesso Mensal Detectado:** `R$ 136.442,89` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 8.186.573,30**
- **Parecer de Retorno Social para o Estado (MA):**
  - **Educação em MA:** Financiaria **5.457.715 merendas escolares** (a R$ 1,50) ou **68.221 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em MA:** Garantiria **68.221 consultas médicas no SUS** ou **181.923 kits de medicamentos essenciais**.
  - **Segurança Pública em MA:** Custearia **23.390 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #03 — Silas Câmara (Dep. Federal) (REPUBLICANOS-AM)
- **Categoria da Despesa (CEAP):** `LOCAÇÃO OU FRETAMENTO DE AERONAVES`
- **Fornecedor Declarado:** `AMAZONAVES TAXI AEREO LTDA` (CNPJ/CPF: `030.907.560/0016-7` | Competência: `02/2026`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/1286/2026/8065824.pdf)
- **Valor Líquido da Nota:** **R$ 149.500,00** *(Valor Base Aceitável na Categoria: R$ 21.705,79)*
- **Excesso Mensal Detectado:** `R$ 127.794,21` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 7.667.652,89**
- **Parecer de Retorno Social para o Estado (AM):**
  - **Educação em AM:** Financiaria **5.111.768 merendas escolares** (a R$ 1,50) ou **63.897 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em AM:** Garantiria **63.897 consultas médicas no SUS** ou **170.392 kits de medicamentos essenciais**.
  - **Segurança Pública em AM:** Custearia **21.907 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #04 — Sargento Gonçalves (Dep. Federal) (PL-RN)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Maricleydson Costa da Silva` (CNPJ/CPF: `173.404.470/0010-2` | Competência: `10/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/nota-fiscal-eletronica?ideDocumentoFiscal=7995308)
- **Valor Líquido da Nota:** **R$ 128.000,00** *(Valor Base Aceitável na Categoria: R$ 3.557,11)*
- **Excesso Mensal Detectado:** `R$ 124.442,89` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 7.466.573,30**
- **Parecer de Retorno Social para o Estado (RN):**
  - **Educação em RN:** Financiaria **4.977.715 merendas escolares** (a R$ 1,50) ou **62.221 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em RN:** Garantiria **62.221 consultas médicas no SUS** ou **165.923 kits de medicamentos essenciais**.
  - **Segurança Pública em RN:** Custearia **21.333 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #05 — Sargento Gonçalves (Dep. Federal) (PL-RN)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Maricleydson Costa da Silva` (CNPJ/CPF: `173.404.470/0010-2` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/nota-fiscal-eletronica?ideDocumentoFiscal=8035404)
- **Valor Líquido da Nota:** **R$ 126.400,00** *(Valor Base Aceitável na Categoria: R$ 3.557,11)*
- **Excesso Mensal Detectado:** `R$ 122.842,89` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 7.370.573,30**
- **Parecer de Retorno Social para o Estado (RN):**
  - **Educação em RN:** Financiaria **4.913.715 merendas escolares** (a R$ 1,50) ou **61.421 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em RN:** Garantiria **61.421 consultas médicas no SUS** ou **163.790 kits de medicamentos essenciais**.
  - **Segurança Pública em RN:** Custearia **21.058 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #06 — PROFESSORA DORINHA SEABRA (Senador) (N/I-TO)
- **Categoria da Despesa (CEAP):** `CONTRATAÇÃO DE CONSULTORIAS, ASSESSORIAS, PESQUISAS, TRABALHOS TÉCNICOS E OUTROS SERVIÇOS DE APOIO AO EXERCÍCIO DO MANDATO PARLAMENTAR`
- **Fornecedor Declarado:** `GRÁFICA E EDITORA SANTO EXPEDITO LTDA` (CNPJ/CPF: `09.015.122/0001-26` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www6g.senado.leg.br/transparencia/sen/)
- **Valor Líquido da Nota:** **R$ 110.000,00** *(Valor Base Aceitável na Categoria: R$ 1.945,21)*
- **Excesso Mensal Detectado:** `R$ 108.054,79` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 6.483.287,15**
- **Parecer de Retorno Social para o Estado (TO):**
  - **Educação em TO:** Financiaria **4.322.191 merendas escolares** (a R$ 1,50) ou **54.027 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em TO:** Garantiria **54.027 consultas médicas no SUS** ou **144.073 kits de medicamentos essenciais**.
  - **Segurança Pública em TO:** Custearia **18.523 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #07 — Marcos Soares (Dep. Federal) (PSDB-RJ)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Supraset Grafica e Editora Ltda` (CNPJ/CPF: `012.096.040/0010-5` | Competência: `12/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/nota-fiscal-eletronica?ideDocumentoFiscal=8037519)
- **Valor Líquido da Nota:** **R$ 110.000,00** *(Valor Base Aceitável na Categoria: R$ 3.557,11)*
- **Excesso Mensal Detectado:** `R$ 106.442,89` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 6.386.573,30**
- **Parecer de Retorno Social para o Estado (RJ):**
  - **Educação em RJ:** Financiaria **4.257.715 merendas escolares** (a R$ 1,50) ou **53.221 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em RJ:** Garantiria **53.221 consultas médicas no SUS** ou **141.923 kits de medicamentos essenciais**.
  - **Segurança Pública em RJ:** Custearia **18.247 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #08 — Antônia Lúcia (Dep. Federal) (MDB-AC)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `E.O.PEREIRA` (CNPJ/CPF: `236.578.190/0012-1` | Competência: `10/2026`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/2258/2026/8162426.pdf)
- **Valor Líquido da Nota:** **R$ 103.000,00** *(Valor Base Aceitável na Categoria: R$ 3.557,11)*
- **Excesso Mensal Detectado:** `R$ 99.442,89` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 5.966.573,30**
- **Parecer de Retorno Social para o Estado (AC):**
  - **Educação em AC:** Financiaria **3.977.715 merendas escolares** (a R$ 1,50) ou **49.721 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em AC:** Garantiria **49.721 consultas médicas no SUS** ou **132.590 kits de medicamentos essenciais**.
  - **Segurança Pública em AC:** Custearia **17.047 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

---

## 4. Parecer Analítico Consolidado

### 4.1 Diagnóstico Técnico por Categoria de Gasto (CEAP)

A inspeção automatizada pela Rede Neural Multilayer Perceptron (`NnModel`) evidencia que os maiores desvios financeiros concentram-se em rubricas de difícil mensuração física imediata (como divulgação de atividade parlamentar, consultorias técnicas, locação de veículos e passagens aéreas). Confira a distribuição do impacto acumulado em 5 anos por categoria entre os casos suspeitos:

| Categoria de Despesa (CEAP) | Ocorrências Atípicas | Impacto Projetado em 5 Anos |
| :--- | :---: | :---: |
| `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR` | 6192 | **R$ 4.833.710.262,00** |
| `MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE PARLAMENTAR` | 6517 | **R$ 2.032.922.279,83** |
| `LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES` | 1878 | **R$ 840.120.839,57** |
| `LOCOMOÇÃO, HOSPEDAGEM, ALIMENTAÇÃO, COMBUSTÍVEIS E LUBRIFICANTES` | 2078 | **R$ 644.929.168,70** |
| `COMBUSTÍVEIS E LUBRIFICANTES` | 3406 | **R$ 602.735.161,42** |
| `CONTRATAÇÃO DE CONSULTORIAS, ASSESSORIAS, PESQUISAS, TRABALHOS TÉCNICOS E OUTROS SERVIÇOS DE APOIO AO EXERCÍCIO DO MANDATO PARLAMENTAR` | 1027 | **R$ 523.272.752,94** |
| `ALUGUEL DE IMÓVEIS PARA ESCRITÓRIO POLÍTICO, COMPREENDENDO DESPESAS CONCERNENTES A ELES` | 1276 | **R$ 341.639.850,78** |
| `TELEFONIA` | 5512 | **R$ 144.092.815,61** |
| `SERVIÇO DE SEGURANÇA PRESTADO POR EMPRESA ESPECIALIZADA` | 298 | **R$ 115.778.778,63** |
| `HOSPEDAGEM ,EXCETO DO PARLAMENTAR NO DISTRITO FEDERAL` | 1106 | **R$ 109.825.090,08** |
| `PASSAGENS AÉREAS, AQUÁTICAS E TERRESTRES NACIONAIS` | 637 | **R$ 102.898.372,56** |
| `LOCAÇÃO OU FRETAMENTO DE AERONAVES` | 142 | **R$ 92.954.312,62** |
| `AQUISIÇÃO DE MATERIAL DE CONSUMO PARA USO NO ESCRITÓRIO POLÍTICO, INCLUSIVE AQUISIÇÃO OU LOCAÇÃO DE SOFTWARE, DESPESAS POSTAIS, AQUISIÇÃO DE PUBLICAÇÕES, LOCAÇÃO DE MÓVEIS E DE EQUIPAMENTOS` | 367 | **R$ 84.898.269,79** |
| `SERVIÇO DE TÁXI, PEDÁGIO E ESTACIONAMENTO` | 9347 | **R$ 69.831.498,50** |
| `PASSAGEM AÉREA - REEMBOLSO` | 26 | **R$ 23.155.135,93** |
| `FORNECIMENTO DE ALIMENTAÇÃO DO PARLAMENTAR` | 7395 | **R$ 18.137.500,90** |
| `SERVIÇOS POSTAIS` | 146 | **R$ 8.487.578,93** |
| `ASSINATURA DE PUBLICAÇÕES` | 249 | **R$ 5.199.661,91** |
| `PASSAGENS TERRESTRES, MARÍTIMAS OU FLUVIAIS` | 143 | **R$ 2.533.468,56** |
| `PASSAGEM AÉREA - RPA` | 14 | **R$ 2.303.010,33** |
| `AQUISIÇÃO DE TOKENS E CERTIFICADOS DIGITAIS` | 2 | **R$ 56.271,55** |

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
