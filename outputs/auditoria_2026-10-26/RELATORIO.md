# Revisão do OlhoCidadão — IA, dados e interface

Data: 26/10/2026. Revisão do código atual, artefatos locais, testes e portal em execução.

## Parecer

O projeto tem uma base funcional de consulta pública: coleta oficial integral, preservação dos CSVs, inferência vetorial, filtros compartilháveis, cache IndexedDB, Web Worker, resumo inicial leve e indicação de divergências municipais. O build e os 13 testes existentes passaram.

A IA permanece experimental para auditoria. A MLP aprende a reproduzir um rótulo gerado por uma regra estatística, sem exemplos confirmados de irregularidade. Ela recupera muitos positivos dessa regra, mas produz bastante ruído. A prioridade é melhorar a definição do problema, a validação e a apresentação das evidências antes de aumentar a rede ou contratar um modelo de linguagem maior.

Esta revisão gerou somente diagnósticos e este relatório. Os pesos, CSVs e comportamento do portal foram preservados. Não foi feita chamada autenticada ao Gemini, novo download nacional ou implantação.

## Verificações realizadas

- 9 testes Python e 4 testes JavaScript passaram; `pnpm run build` passou.
- Os hashes dos quatro CSVs principais coincidem com o relatório de qualidade existente.
- Os pesos exportados reproduzem as classes e probabilidades salvas nas 2.000 linhas de teste selecionadas pelo procedimento atual.
- O filtro MA/2026 mostrou 2.153 notas e R$ 5.651.857,01 no navegador.
- A aba de obras mostrou municípios, filtros e avisos de divergência; Arari/MA continua encontrando a escola cujo nome e geometria publicados divergem.
- Não foram medidos Core Web Vitals em dispositivo móvel/4G. Tamanhos de arquivos abaixo são observados em disco; não representam um resultado de Lighthouse.

## 1. O que as métricas da IA dizem

Todos os números desta seção usam **o rótulo estatístico atual**, não um parecer humano sobre fraude.

| Método no mesmo conjunto de teste | Acurácia | Precisão dos alertas | Recall |
| --- | ---: | ---: | ---: |
| MLP atual, limiar 0,50 | 91,70% | 43,36% | 96,88% |
| Marcar tudo como normal | 93,60% | 0% | 0% |
| Aplicar diretamente a regra que criou o rótulo | 100% | 100% | 100% |

A segunda linha mostra por que acurácia isolada é inadequada: ela pode ser alta ignorando todos os casos positivos. A terceira linha é uma identidade por construção; não demonstra capacidade de identificar fraude. Se a meta for apenas aplicar a regra atual, a regra direta é exata e dispensa sua aproximação pela MLP.

No teste da MLP: **124 verdadeiros positivos, 162 falsos positivos, 1.710 verdadeiros negativos e 4 falsos negativos**. Entre seus 286 alertas, 56,64% não satisfazem nem a regra que gerou o alvo. F1: 0,599. No conjunto integral, há 47.758 alertas da rede contra 20.750 positivos da regra; isso é diagnóstico descritivo e inclui linhas usadas no treino.

### Limiares: há melhoria possível sem aumentar a rede

| Limiar do score | Alertas | Precisão | Recall |
| --- | ---: | ---: | ---: |
| 0,50 | 286 | 43,36% | 96,88% |
| 0,70 | 189 | 63,49% | 93,75% |
| 0,80 | 154 | 74,68% | 89,84% |
| 0,90 | 123 | 82,11% | 78,91% |

São resultados **exploratórios no teste já observado**. Não basta escolher 0,80 por esta tabela e divulgar o ganho como nova validação: deve-se ajustar o limiar em uma validação separada e avaliar uma vez no teste final. A escolha depende da capacidade de revisão humana e do custo de perder um caso. [Referência sobre métricas](https://scikit-learn.org/stable/modules/model_evaluation.html).

### Os scores ainda não são probabilidades calibradas

O erro de calibração com dez faixas ficou em 12,03 pontos percentuais. Na faixa de scores 0,70–0,80, o score médio foi 75,1%, mas 14,3% das 35 notas satisfaziam o rótulo. A amostra da faixa é pequena, porém o resultado já impede interpretar o percentual como confiança demonstrada em uma irregularidade. Treino balanceado em 50/50 e população com aproximadamente 6,37% de positivos reforçam a necessidade de calibrar na distribuição de uso.

Implementar curva de confiabilidade, Brier score e calibração em validação independente; até lá, apresentar “score de atipicidade”. A própria calibração contra a regra também não transforma o score em probabilidade de corrupção. [Documentação de calibração](https://scikit-learn.org/stable/modules/calibration.html).

![Diagnóstico de limiares e calibração](<D:/Docs/Projetos Git/OlhoCidadão/outputs/auditoria_2026-10-26/diagnostico_modelo.png>)

## 2. Achados técnicos priorizados

### P1 — Avaliação circular e ausência de validação temporal

[src/data_fetcher.py:431](<D:/Docs/Projetos Git/OlhoCidadão/src/data_fetcher.py:431>) — média, desvio e limiares são calculados no consolidado antes da separação. O alvo é exatamente `valor > média + 1,2 × desvio`.

[main.py:257](<D:/Docs/Projetos Git/OlhoCidadão/main.py:257>) — teste aleatório de notas mistura anos. Todas as 2.000 notas de teste têm parlamentar presente em outras linhas da base. Isso permite avaliar novas notas de pessoas já vistas, mas não comprova desempenho futuro nem para parlamentares/fornecedores novos.

Proposta: separar treino, validação e teste por períodos; por exemplo, treino em 2025, validação no início de 2026 e teste nos meses finais disponíveis. Calcular estatísticas e transformações usando apenas o passado. Acrescentar avaliação agrupada por parlamentar e fornecedor. Guardar os índices e a versão do dataset. [Validação temporal](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

### P1 — Linguagem de desperdício excede a evidência disponível

[index.html:180](<D:/Docs/Projetos Git/OlhoCidadão/index.html:180>) — o KPI usa “Excesso real no período”. A diferença para uma média não comprova gasto indevido ou economia recuperável.

[src/impact_analyzer.py:52](<D:/Docs/Projetos Git/OlhoCidadão/src/impact_analyzer.py:52>) — o prompt chama o montante de “desperdício” e multiplica o excedente por 60 meses. Uma nota não é uma despesa mensal recorrente demonstrada. O relatório possui ressalvas, mas o prompt pode induzir uma narrativa mais forte que os dados permitem.

Proposta: “Excedente estatístico estimado”, com metodologia acessível; apresentar cenários sociais como equivalências hipotéticas. Separar fatos publicados, sinais do modelo e conclusões humanas. Exibir o período efetivamente observado em todos os pareceres.

### P1 — Modelo de linguagem não participa do resumo do dashboard

[web/app.js:590](<D:/Docs/Projetos Git/OlhoCidadão/web/app.js:590>) — o resumo é calculado por templates JavaScript. Isso é reproduzível e econômico, porém deve ser distinguido da classificação da rede.

[src/impact_analyzer.py:22](<D:/Docs/Projetos Git/OlhoCidadão/src/impact_analyzer.py:22>) — Gemini é opcional e usado no relatório Markdown; sem chave ou após erro, entra o texto offline. O identificador `gemini-3.8-flash` consta na documentação atual, portanto não há evidência para tratá-lo como identificador inexistente. A integração autenticada não foi testada nesta revisão. [Modelo publicado](https://ai.google.dev/gemini-api/docs/latest-model).

Proposta: informar o motor que produziu cada texto e adicionar uma consulta em linguagem natural baseada em dados recuperados. O backend calcula os valores; o modelo organiza explicações com IDs das notas, links e limitações. Uma saída estruturada pode exigir campos como `evidencias`, `periodo`, `fontes` e `limitacoes`, mas a aderência ao esquema não garante veracidade. [Saída estruturada do Gemini](https://ai.google.dev/gemini-api/docs/structured-output).

### P2 — Variáveis pouco representativas e categorias tratadas como escala numérica

[src/data_fetcher.py:460](<D:/Docs/Projetos Git/OlhoCidadão/src/data_fetcher.py:460>) — são apenas cinco entradas: valor, categoria codificada, mês, partido codificado e UF codificada. CNPJ, recorrência, histórico e concentração não entram na rede. Códigos arbitrários de partido e UF são normalizados como números contínuos.

No teste de sensibilidade, variar somente o código de UF alterou a classe em alguma alternativa para 289/2.000 notas; variar partido alterou 357/2.000. Algumas combinações contrafactuais são artificiais: isso não comprova discriminação em uso real. Mostra, porém, dependência de variáveis que nem fazem parte da definição do alvo atual.

Proposta: experimentar remover partido; comparar modelos com e sem UF; usar representação categórica apropriada quando necessária. Incluir `log1p(valor)`, razão para mediana histórica do grupo comparável, MAD/IQR, mudanças em relação aos meses anteriores, concentração por fornecedor e frequência de lançamentos. Toda informação histórica precisa respeitar a data do registro. [Codificação de categorias](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

### P2 — Gradiente e métricas de treino precisam de convenções consistentes

[src/nn_model.py:175](<D:/Docs/Projetos Git/OlhoCidadão/src/nn_model.py:175>) — o gradiente divide por `N/8`, enquanto a loss é média por `N`. O teste numérico isolado com 16 exemplos encontrou gradiente aplicado exatamente oito vezes o gradiente da loss. Isso equivale a uma taxa efetiva maior para esses lotes e dificulta comparar hiperparâmetros; não significa que o treino atual não convergiu.

[src/nn_model.py:260](<D:/Docs/Projetos Git/OlhoCidadão/src/nn_model.py:260>) — as curvas e a acurácia de treino reportada são suavizadas por EMA. Manter valores reais por avaliação e deixar a suavização apenas como opção visual. Padronizar derivadas e retunar a taxa de aprendizagem; incluir teste de gradientes e paridade de inferência Python/JavaScript.

### P2 — O peso do dataset domina o carregamento

[web/dataset.js:98](<D:/Docs/Projetos Git/OlhoCidadão/web/dataset.js:98>) — o portal baixa e interpreta o mesmo JSON contendo notas e obras: **212,5 MB descomprimidos / 28,2 MB gzip**. O bundle principal do build é aproximadamente 48,6 kB, ou 16,8 kB gzip. O problema principal está nos dados, apesar dos ganhos já implementados de cache, resumo inicial e Worker.

Proposta: separar despesas e obras; carregar obras ao abrir a aba; dividir dados por UF/ano e município ou consultar por uma API paginada. Manter os CSVs com os nomes atuais como arquivos de origem. Acrescentar manifestos com hash e versão. Migrar também agregações de gráficos para Worker quando houver ganho medido; a filtragem já ocorre fora da thread principal. Medir INP/LCP/CLS em celular e rede limitada antes/depois. [INP atual](https://web.dev/articles/inp).

### P2 — Qualidade da origem precisa acompanhar a confiança da localização

[src/public_works.py:103](<D:/Docs/Projetos Git/OlhoCidadão/src/public_works.py:103>) — há tratamento útil de conflitos entre nome e geometria. O snapshot registra 14.716 divergências. No navegador, uma escola de Arari aparece com tipo “Saneamento / Dragagem”, evidenciando que os demais metadados também podem precisar de revisão.

Proposta: distinguir “geometria publicada”, “município identificado no nome” e “revisado manualmente”; permitir filtrar registros divergentes e manter a evidência original. Uma eventual reclassificação automática de setor deve ser uma sugestão separada do campo oficial. Usar mapas municipais agregados enquanto coordenadas exatas não estiverem verificadas.

### P3 — Reprodutibilidade e cobertura de testes

[requirements.txt:1](<D:/Docs/Projetos Git/OlhoCidadão/requirements.txt:1>) — dependências Python não têm versões fixadas. A suíte atual cobre parsing, balanceamento, inferência básica e municípios, mas não mede qualidade da IA, calibração, gradientes, paridade do Worker ou regressões completas do navegador.

Proposta: lock de dependências, execução de build/testes em CI, validação de dados em fixture, manifestos de coleta e do modelo, testes de importação sem executar rede, validação temporal reproduzível e conjunto de casos revisados por humanos. O ganho deve ser medido nesse conjunto, não somente nos próprios rótulos gerados.

## 3. Evolução da IA que eu implementaria

| Etapa | Implementação | Critério para aceitar |
| --- | --- | --- |
| 1. Evidência e avaliação | Separar regra, score da MLP e conclusão humana; split temporal; métricas por Casa/categoria/UF | Resultados reproduzíveis, com teste final separado |
| 2. Reduzir ruído | Ajustar limiar em validação; calibrar score; revisar features e categorias | Melhor precisão/recall e precisão entre os primeiros K alertas revisados |
| 3. Comparar detectores | Baseline robusto por pares, MLP atual e Isolation Forest com dados históricos | Ganho verificado no mesmo protocolo, com custo de CPU medido |
| 4. Explicar cada alerta | Valor, distribuição comparável, recorrência, fornecedor, motivo e documentos | Explicação reproduzível a partir das evidências |
| 5. Revisão humana | Registrar “justificado”, “erro da fonte”, “precisa investigar” e evidências | Rótulos confiáveis para avaliar a próxima versão |
| 6. Assistente cidadão | Perguntas sobre o recorte, recuperação de notas e respostas com fontes | Cálculos corretos e rastreáveis; abstenção quando não há dados |

Isolation Forest é um candidato de detecção não supervisionada, não uma melhoria já demonstrada aqui. Autoencoder, grafo de fornecedores e extração de documentos via OCR podem ser etapas posteriores, após existirem dados e critérios de avaliação que justifiquem sua complexidade. [Documentação de detecção de anomalias](https://scikit-learn.org/stable/modules/outlier_detection.html).

Se a rede NumPy fizer parte de uma exigência acadêmica, ela pode ser mantida e comparada com os outros métodos. Não há necessidade demonstrada de aumentar camadas ou parâmetros nesta fase.

## 4. Melhorias visuais e de produto

### Visão inicial

- Preservar a identidade azul/verde e os temas claro/escuro; aumentar a hierarquia entre número principal, contexto e ação.
- Manter busca, UF e período em destaque. Recolher partido, categoria e parâmetros secundários em “Mais filtros”, com chips dos filtros ativos.
- Adicionar uma faixa compacta com data da coleta, período disponível, fonte, versão do modelo e link da metodologia.
- Trocar “Excesso real” por linguagem estatística. Usar verde para situações resolvidas/compatíveis; alertas e incerteza merecem azul ou âmbar, com texto explicativo.

### Alertas e explicações

- Uma lista de revisão priorizada com “Por que recebeu alerta?”, comparação com pares, evidências e ação “Abrir documento”.
- Separar score do modelo, severidade do desvio e estado de revisão. Evitar um percentual isolado parecer certeza de irregularidade.
- Usar painel lateral para detalhes e histórico; permitir comparar parlamentares ou fornecedores no mesmo período, com bases equivalentes.
- Ordenar por relevância e oferecer critérios explícitos: maior desvio relativo, maior valor e maior recorrência. O valor absoluto sozinho favorece despesas naturalmente caras.

### Obras por cidade

- Cidade com busca por nome; o seletor nativo atual oferece centenas de opções em alguns estados.
- Cartão com cidade/endereço e valor em destaque; situação e qualidade da localização como badges curtos; detalhes e conflitos expansíveis.
- Alternância lista/mapa municipal; mostrar investimentos previstos, contratos, empenhos e valores pagos somente quando cada informação existir na fonte.
- Tornar visível que o recorte de cidade pertence às obras. A comparação usa o excedente das despesas parlamentares do recorte e não demonstra vínculo de financiamento.

### Acessibilidade e consistência

Achados observados, no formato arquivo:linha:

- [index.html:22](<D:/Docs/Projetos Git/OlhoCidadão/index.html:22>) — falta `h1`; a página começa semanticamente por títulos `h3`. Introduzir um título principal e uma hierarquia por aba.
- [index.html:382](<D:/Docs/Projetos Git/OlhoCidadão/index.html:382>) — imagem da curva sem `width`/`height`; reservar proporção para reduzir mudanças de layout.
- [web/app.js:133](<D:/Docs/Projetos Git/OlhoCidadão/web/app.js:133>) — URL preserva filtros, mas não a aba ativa/página; compartilhar uma obra deveria abrir diretamente a aba relevante.
- [web/app.js:570](<D:/Docs/Projetos Git/OlhoCidadão/web/app.js:570>) — lista de cidades é reconstruída durante a aplicação de filtros; indexar e reutilizar por UF.

Pontos já adequados: todos os controles inspecionados têm label acessível; há skip link, foco visível, navegação das abas por setas, sinalização textual dos alertas, toast com `aria-live` e respeito a movimento reduzido. As melhorias propostas não exigem reescrever a aplicação em outro framework. [Guidelines consultadas](https://github.com/vercel-labs/web-interface-guidelines).

## 5. Ordem recomendada

1. Corrigir linguagem e distinguir regra, modelo e parecer; montar validação temporal e conjunto de referência.
2. Ajustar limiar/calibração e features; comparar com regra robusta e detector alternativo.
3. Separar payload de despesas/obras e carregar dados por recorte.
4. Melhorar filtros, hierarquia, explicação dos alertas e confiança dos dados municipais.
5. Adicionar revisão humana e assistente de perguntas com evidências.

## Artefatos e reprodução

- [Script offline de avaliação](<D:/Docs/Projetos Git/OlhoCidadão/outputs/auditoria_2026-10-26/avaliar_ia.py>) — executar `python outputs/auditoria_2026-10-26/avaliar_ia.py` na raiz.
- [Métricas completas](<D:/Docs/Projetos Git/OlhoCidadão/outputs/auditoria_2026-10-26/metricas_ia.json>).
- [Gráfico do diagnóstico](<D:/Docs/Projetos Git/OlhoCidadão/outputs/auditoria_2026-10-26/diagnostico_modelo.png>).
- [Captura da interface atual](<D:/Docs/Projetos Git/OlhoCidadão/outputs/auditoria_2026-10-26/interface_atual.png>).
