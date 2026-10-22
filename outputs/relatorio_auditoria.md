# Relatório de Auditoria Inteligente e Impacto Social (CEAP)

**Data da Auditoria:** 22/10/2026 às 00:36  
**Motor de Detecção:** Rede Neural Multilayer Perceptron (`NnModel` 100% NumPy com ativação `tanh`)  
**Motor de Parecer Social:** Motor Analítico Cidadão Offline (Fallback Inteligente Baseado em Custos Públicos Oficiais)  

---

## 1. Resumo Executivo da Auditoria no Conjunto de Teste

- **Gastos Classificados como Atípicos/Suspeitos (Classe 1):** `327` notas fiscais auditadas
- **Volume Mensal Auditado nesses Casos:** `R$ 1.559.866,34`
- **Excesso Mensal Acima da Média Aceitável por Categoria:** `R$ 1.238.886,02`
- **Projeção Acumulada de Desperdício em 5 Anos (60 meses):** **R$ 74.333.161,37**

---

## 2. Tradução em Custo de Oportunidade Social no Brasil (Projeção 5 Anos)

Caso o montante excedente identificado pela Rede Neural fosse realocado integralmente em políticas públicas essenciais, seria possível financiar:

| Área Prioritária | Indicador / Política Pública de Referência | Custo Unitário Médio | Quantidade Financiável (5 Anos) | Fonte Oficial |
| :--- | :--- | :---: | :---: | :--- |
| **Educação Básica** | Merendas Escolares (PNAE/FNDE) | R$ 1,50 | **49.555.440 refeições** | FNDE / MEC |
| **Educação Básica** | Kits Completos de Material Escolar | R$ 120,00 | **619.443 alunos atendidos** | FNDE / MEC |
| **Inclusão Digital** | Computadores para Escolas Públicas | R$ 2.800,00 | **26.547 equipamentos** | PNCP / Gov.br |
| **Saúde Pública (SUS)** | Consultas Médicas de Atenção Primária | R$ 65,00 | **1.143.587 consultas** | DataSUS / SIGTAP |
| **Saúde Pública (SUS)** | Kits Mensais da Farmácia Básica | R$ 45,00 | **1.651.848 tratamentos mensais** | RENAME / MS |
| **Segurança Pública** | Diárias Operacionais de Patrulhamento/EPIs | R$ 420,00 | **176.983 diárias táticas** | SENASP / MJSP |
| **Assistência Social** | Cestas Básicas Familiares (Ref. DIEESE) | R$ 750,00 | **99.110 famílias assistidas** | DIEESE |
| **Renda do Trabalho** | Salários Mínimos Equivalentes | R$ 1.518,00 | **48.967 salários mínimos** | Portal da Transparência |

---

## 3. Detalhamento dos 8 Casos Mais Críticos Detectados pela Rede Neural

### Caso #01 — Wellington Roberto (Dep. Federal) (PSD-PB)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `ELDORADO COMUNICACAO E JORNALISMO LTDA` (CNPJ/CPF: `378.947.490/0013-0` | Competência: `02/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/1703/2025/7900912.pdf)
- **Valor Líquido da Nota:** **R$ 35.000,00** *(Valor Base Aceitável na Categoria: R$ 2.285,75)*
- **Excesso Mensal Detectado:** `R$ 32.714,25` | **Confiança da Rede Neural (`NnModel`):** `99.7%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 1.962.854,83**
- **Parecer de Retorno Social para o Estado (PB):**
  - **Educação em PB:** Financiaria **1.308.569 merendas escolares** (a R$ 1,50) ou **16.357 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em PB:** Garantiria **30.197 consultas médicas no SUS** ou **43.618 kits de medicamentos essenciais**.
  - **Segurança Pública em PB:** Custearia **4.673 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #02 — Luiz Carlos Hauly (Dep. Federal) (PODE-PR)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `SVS COMUNICAÇÃO E MAARKETING DIGITAL LTDA` (CNPJ/CPF: `608.603.880/0012-2` | Competência: `02/2026`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/824/2026/8062113.pdf)
- **Valor Líquido da Nota:** **R$ 27.499,00** *(Valor Base Aceitável na Categoria: R$ 2.285,75)*
- **Excesso Mensal Detectado:** `R$ 25.213,25` | **Confiança da Rede Neural (`NnModel`):** `99.8%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 1.512.794,83**
- **Parecer de Retorno Social para o Estado (PR):**
  - **Educação em PR:** Financiaria **1.008.529 merendas escolares** (a R$ 1,50) ou **12.606 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em PR:** Garantiria **23.273 consultas médicas no SUS** ou **33.617 kits de medicamentos essenciais**.
  - **Segurança Pública em PR:** Custearia **3.601 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #03 — IZALCI LUCAS (Senador) (PL-DF)
- **Categoria da Despesa (CEAP):** `CONTRATAÇÃO DE CONSULTORIAS, ASSESSORIAS, PESQUISAS, TRABALHOS TÉCNICOS E OUTROS SERVIÇOS DE APOIO AO EXERCÍCIO DO MANDATO PARLAMENTAR`
- **Fornecedor Declarado:** `Edward Duarte Jorge` (CNPJ/CPF: `12.126.478/0001-41` | Competência: `10/2025`) | [Nota Fiscal Oficial na Câmara](https://www6g.senado.leg.br/transparencia/sen/)
- **Valor Líquido da Nota:** **R$ 26.000,00** *(Valor Base Aceitável na Categoria: R$ 1.127,29)*
- **Excesso Mensal Detectado:** `R$ 24.872,71` | **Confiança da Rede Neural (`NnModel`):** `100.0%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 1.492.362,49**
- **Parecer de Retorno Social para o Estado (DF):**
  - **Educação em DF:** Financiaria **994.908 merendas escolares** (a R$ 1,50) ou **12.436 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em DF:** Garantiria **22.959 consultas médicas no SUS** ou **33.163 kits de medicamentos essenciais**.
  - **Segurança Pública em DF:** Custearia **3.553 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #04 — Juscelino Filho (Dep. Federal) (PSDB-MA)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `L DOS S CARMO` (CNPJ/CPF: `508.422.300/0017-6` | Competência: `06/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/3020/2025/7940992.pdf)
- **Valor Líquido da Nota:** **R$ 25.000,00** *(Valor Base Aceitável na Categoria: R$ 2.285,75)*
- **Excesso Mensal Detectado:** `R$ 22.714,25` | **Confiança da Rede Neural (`NnModel`):** `99.8%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 1.362.854,83**
- **Parecer de Retorno Social para o Estado (MA):**
  - **Educação em MA:** Financiaria **908.569 merendas escolares** (a R$ 1,50) ou **11.357 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em MA:** Garantiria **20.966 consultas médicas no SUS** ou **30.285 kits de medicamentos essenciais**.
  - **Segurança Pública em MA:** Custearia **3.244 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #05 — Rubens Pereira Júnior (Dep. Federal) (PT-MA)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `STEVAM GREGORY ROCHA POLLON` (CNPJ/CPF: `441.902.180/0012-1` | Competência: `10/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/2902/2025/8012606.pdf)
- **Valor Líquido da Nota:** **R$ 25.000,00** *(Valor Base Aceitável na Categoria: R$ 2.285,75)*
- **Excesso Mensal Detectado:** `R$ 22.714,25` | **Confiança da Rede Neural (`NnModel`):** `99.8%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 1.362.854,83**
- **Parecer de Retorno Social para o Estado (MA):**
  - **Educação em MA:** Financiaria **908.569 merendas escolares** (a R$ 1,50) ou **11.357 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em MA:** Garantiria **20.966 consultas médicas no SUS** ou **30.285 kits de medicamentos essenciais**.
  - **Segurança Pública em MA:** Custearia **3.244 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #06 — Luiz Gastão (Dep. Federal) (PSD-CE)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `A DE LIMA CARDOSO` (CNPJ/CPF: `320.833.600/0019-0` | Competência: `02/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/3545/2025/7871350.pdf)
- **Valor Líquido da Nota:** **R$ 25.000,00** *(Valor Base Aceitável na Categoria: R$ 2.285,75)*
- **Excesso Mensal Detectado:** `R$ 22.714,25` | **Confiança da Rede Neural (`NnModel`):** `99.8%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 1.362.854,83**
- **Parecer de Retorno Social para o Estado (CE):**
  - **Educação em CE:** Financiaria **908.569 merendas escolares** (a R$ 1,50) ou **11.357 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em CE:** Garantiria **20.966 consultas médicas no SUS** ou **30.285 kits de medicamentos essenciais**.
  - **Segurança Pública em CE:** Custearia **3.244 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #07 — Delegada Katarina (Dep. Federal) (PSD-SE)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `CONCEITO COMUNICAÇÃO INTEGRADA LTDA` (CNPJ/CPF: `004.044.190/0010-9` | Competência: `04/2025`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/3543/2025/7898079.pdf)
- **Valor Líquido da Nota:** **R$ 24.647,00** *(Valor Base Aceitável na Categoria: R$ 2.285,75)*
- **Excesso Mensal Detectado:** `R$ 22.361,25` | **Confiança da Rede Neural (`NnModel`):** `99.8%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 1.341.674,83**
- **Parecer de Retorno Social para o Estado (SE):**
  - **Educação em SE:** Financiaria **894.449 merendas escolares** (a R$ 1,50) ou **11.180 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em SE:** Garantiria **20.641 consultas médicas no SUS** ou **29.814 kits de medicamentos essenciais**.
  - **Segurança Pública em SE:** Custearia **3.194 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

### Caso #08 — Delegado Matheus Laiola (Dep. Federal) (UNIÃO-PR)
- **Categoria da Despesa (CEAP):** `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR`
- **Fornecedor Declarado:** `Facebook Serviços Online do Brasil Ltda.` (CNPJ/CPF: `133.470.160/0011-7` | Competência: `04/2026`) | [Nota Fiscal Oficial na Câmara](https://www.camara.leg.br/cota-parlamentar/documentos/publ/3599/2026/8097894.pdf)
- **Valor Líquido da Nota:** **R$ 24.180,80** *(Valor Base Aceitável na Categoria: R$ 2.285,75)*
- **Excesso Mensal Detectado:** `R$ 21.895,05` | **Confiança da Rede Neural (`NnModel`):** `99.8%`
- **Impacto Acumulado em 5 Anos (60 meses):** **R$ 1.313.702,83**
- **Parecer de Retorno Social para o Estado (PR):**
  - **Educação em PR:** Financiaria **875.801 merendas escolares** (a R$ 1,50) ou **10.947 kits escolares completos** para crianças da rede pública.
  - **Saúde Básica em PR:** Garantiria **20.210 consultas médicas no SUS** ou **29.193 kits de medicamentos essenciais**.
  - **Segurança Pública em PR:** Custearia **3.127 diárias operacionais de patrulhamento/equipamentos** para proteção comunitária no estado.

---

## 4. Parecer Analítico Consolidado

### 4.1 Diagnóstico Técnico por Categoria de Gasto (CEAP)

A inspeção automatizada pela Rede Neural Multilayer Perceptron (`NnModel`) evidencia que os maiores desvios financeiros concentram-se em rubricas de difícil mensuração física imediata (como divulgação de atividade parlamentar, consultorias técnicas, locação de veículos e passagens aéreas). Confira a distribuição do impacto acumulado em 5 anos por categoria entre os casos suspeitos:

| Categoria de Despesa (CEAP) | Ocorrências Atípicas | Impacto Projetado em 5 Anos |
| :--- | :---: | :---: |
| `DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR` | 45 | **R$ 33.749.155,26** |
| `MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE PARLAMENTAR` | 49 | **R$ 12.898.212,02** |
| `LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES` | 22 | **R$ 7.591.934,12** |
| `COMBUSTÍVEIS E LUBRIFICANTES` | 69 | **R$ 4.129.063,59** |
| `PASSAGEM AÉREA - SIGEPA` | 46 | **R$ 4.022.183,24** |
| `CONTRATAÇÃO DE CONSULTORIAS, ASSESSORIAS, PESQUISAS, TRABALHOS TÉCNICOS E OUTROS SERVIÇOS DE APOIO AO EXERCÍCIO DO MANDATO PARLAMENTAR` | 12 | **R$ 3.641.824,10** |
| `LOCOMOÇÃO, HOSPEDAGEM, ALIMENTAÇÃO, COMBUSTÍVEIS E LUBRIFICANTES` | 15 | **R$ 2.690.936,50** |
| `ALUGUEL DE IMÓVEIS PARA ESCRITÓRIO POLÍTICO, COMPREENDENDO DESPESAS CONCERNENTES A ELES` | 11 | **R$ 1.719.712,73** |
| `PASSAGENS AÉREAS, AQUÁTICAS E TERRESTRES NACIONAIS` | 12 | **R$ 1.615.282,62** |
| `HOSPEDAGEM ,EXCETO DO PARLAMENTAR NO DISTRITO FEDERAL` | 16 | **R$ 832.507,58** |
| `TELEFONIA` | 21 | **R$ 644.869,81** |
| `SERVIÇO DE SEGURANÇA PRESTADO POR EMPRESA ESPECIALIZADA` | 2 | **R$ 568.812,14** |
| `SERVIÇO DE TÁXI, PEDÁGIO E ESTACIONAMENTO` | 5 | **R$ 167.714,86** |
| `AQUISIÇÃO DE MATERIAL DE CONSUMO PARA USO NO ESCRITÓRIO POLÍTICO, INCLUSIVE AQUISIÇÃO OU LOCAÇÃO DE SOFTWARE, DESPESAS POSTAIS, AQUISIÇÃO DE PUBLICAÇÕES, LOCAÇÃO DE MÓVEIS E DE EQUIPAMENTOS` | 1 | **R$ 34.661,73** |
| `PASSAGENS TERRESTRES, MARÍTIMAS OU FLUVIAIS` | 1 | **R$ 26.291,08** |

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
