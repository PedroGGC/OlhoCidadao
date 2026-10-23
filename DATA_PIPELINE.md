# Coleta integral e reprocessamento — 22/10/2026

Os nomes dos CSVs em `data/` foram mantidos. Os arquivos bruto, processado e o dashboard agora contêm a base integral válida, sem o corte anterior de 10.000/3.000 registros.

## Executar

```powershell
python main.py
python main.py --offline
python -m unittest discover -s tests -v
python scripts/validate_dataset.py
pnpm run build
```

A execução padrão atualiza os dumps anuais. `--offline` reutiliza os dumps locais para pré-processar, treinar e exportar novamente. Ambos os anos e as duas Casas precisam ser lidos e validados antes de substituir os dumps locais; falhas não são tratadas como sucesso parcial. As saídas são escritas em arquivos temporários antes da publicação. Se um visualizador do Windows mantiver o arquivo antigo mapeado, a versão anterior poderá ficar em `outputs/replaced_files/` até ser liberada.

## Fontes e escopo

- Câmara: [2025](https://www.camara.leg.br/cotas/Ano-2025.csv.zip) e [2026](https://www.camara.leg.br/cotas/Ano-2026.csv.zip). ZIP aberto em memória, CSV separado por ponto e vírgula e UTF-8. O CSV completo conserva suas 13 colunas anteriores.
- Senado: [2025](https://www.senado.leg.br/transparencia/LAI/verba/despesa_ceaps_2025.csv) e [2026](https://www.senado.leg.br/transparencia/LAI/verba/despesa_ceaps_2026.csv). Latin-1, preâmbulo de atualização ignorado, valores brasileiros convertidos sem perder milhares ou centavos.
- UF e código dos senadores: [cadastro da 57ª legislatura](https://legis.senado.leg.br/dadosabertos/senador/lista/legislatura/57), incluindo suplentes. Não há fallback de UF para DF. Nomes abreviados só são resolvidos por prefixo inequívoco de nome oficial.
- O CSV CEAPS não informa partido na data da despesa nem tipo documental. Esses campos ficam `N/I` e vazio, respectivamente; não se infere partido histórico pelo cadastro atual.
- CNPJ/CPF e número documental são preservados como texto, inclusive zeros iniciais. Uma data inválida publicada em 2025 foi preservada literalmente; ano/mês de competência vêm das colunas oficiais.
- Entram valores líquidos/reembolsados **positivos**. Na Câmara, só entram registros com identificador parlamentar e UF válida; despesas de lideranças sem UF ficam fora. Não se aplica deduplicação por número de nota, pois um documento pode aparecer em lançamentos distintos.
- Os totais são somas dos lançamentos positivos publicados, **não saldo líquido após estornos**. 2026 é parcial conforme a atualização da fonte, e não uma previsão de exercício completo. Não se multiplica nem repondera nenhuma nota.

## Resultado conciliado

| Base | Registros |
| --- | ---: |
| Câmara 2025 | 169.741 |
| Câmara 2026 | 116.511 |
| Senado 2025 | 23.736 |
| Senado 2026 | 15.796 |
| Total | 325.784 |

Total positivo: **R$ 418.329.396,80**, nas 27 UFs. O menor total estadual é 2.016 registros no conjunto dos dois anos. O Senado inclui 90 pessoas com despesas no período, pois titulares e suplentes podem ocupar as 81 cadeiras.

| Maranhão | Registros | Valor positivo |
| --- | ---: | ---: |
| 2025 | 3.784 | R$ 9.679.640,60 |
| 2026 | 2.153 | R$ 5.651.857,01 |

`outputs/data_quality_report.json` registra contagens, somas por UF/ano, tamanhos e hashes SHA-256. O validador concilia os valores em centavos entre os dumps, o bruto, o processado e o payload; também verifica a integridade do gzip e das inferências.

## Treinamento e dashboard

`NnModel` é inicializada do zero. São separadas 2.000 notas para teste; o treino seleciona até 5.000 outras notas, sem reposição, metade por classe e com representação de UF, Casa e ano. A normalização usa somente o treino. O `forward` é executado sobre os 325.784 registros para gerar as classes e probabilidades finais, salvas também no CSV processado.

O alvo é a regra estatística de categoria `valor > média + 1,2 × desvio-padrão`. Foi removido o ajuste anterior por percentil que forçava mais alertas. O excedente comparativo usa a diferença positiva em relação à média de referência, sem piso artificial de 30%.

As métricas medem concordância com esse alvo heurístico, não detecção comprovada de fraude. Nesta execução: acurácia de teste 91,70%, precisão 43,36% e recall 96,87%. A precisão evidencia falsos positivos relevantes; os alertas exigem análise humana. As estatísticas de categoria descrevem a base consolidada, portanto não se trata de avaliação temporal de gastos futuros.

O JSON usa colunas e dicionários de textos repetidos, preservando cada registro. `dashboard_data.json` tem 57,3 MB e sua versão gzip 9,8 MB. O navegador prefere gzip e possui fallback para JSON; as tabelas continuam paginadas. Totais e filtros sempre operam sobre a base inteira. Não se utiliza amostragem para aumentar artificialmente o volume estadual.

Os campos legados `excessoMensal`/`monthlyExcessTotal` representam o excedente das notas do recorte, não uma despesa mensal recorrente. `impacto5Anos` é apenas cenário hipotético de repetição por 60 meses; não é gasto público observado nem projeção estatística validada.

## Referências de custos

`src/public_costs.py` centraliza os indicadores exportados para CSV, JSON, relatório e comparações no navegador. As faixas solicitadas foram adotadas como **cenários comparativos**, sem atribuir a eles um convênio ou repasse real:

| Cenário | Valor |
| --- | ---: |
| Reforma de UBS | R$ 250.000 |
| Climatização escolar | R$ 130.000 |
| Ambulância SAMU | R$ 280.000 |
| Trecho de pavimentação | R$ 450.000 |
| Aluno/mês, aproximação FUNDEB | R$ 480 |
| Consulta especializada | R$ 120 |
| Cesta assistencial ou diária de segurança | R$ 350 |

A [tabela CISCOPAR](https://portal.ciscopar.com.br/credenciamento/tabelas?tp_tabela=3) fornece exemplo de consulta especializada a R$ 120. O FUNDEB varia por rede/modalidade e não define custo nacional uniforme de R$ 480/mês; veja as [publicações oficiais de 2025](https://www.gov.br/fnde/pt-br/acesso-a-informacao/acoes-e-programas/financiamento/fundeb/2025). A referência de ambulância é arredondada: [Curitiba informou R$ 289 mil por veículo](https://www.curitiba.pr.gov.br/noticias/prefeito-eduardo-pimentel-entrega-20-ambulancias-para-renovacao-da-frota-do-samu/78124). A [pavimentação de três ruas em São Cristóvão](https://www.saocristovao.se.gov.br/noticia/prefeitura-de-sao-cristovao-inaugura-pavimentacao-de-tres-ruas-no-povoado-tinhare) é exemplo de investimento de cerca de R$ 450 mil, com escopo local específico.

O catálogo mantém dez setores, mas deixa explícito que seus custos são estimativas. Não inventa municípios contratantes, situação de paralisação, execução física ou parcelas mensais. Não houve importação de contratos do Transferegov nesta implementação. Cesta assistencial de R$ 350 não é a cesta nacional DIEESE, e diária operacional não é tarifa nacional uniforme.


## Atualização: obras oficiais por município

A lista de obras agora usa a [API pública atual do ObrasGov](https://api-publica.obrasgov.gestao.gov.br/obras). A consulta nacional pagina `/projeto-investimento` e `/geometria` até completar os totais declarados pela fonte. O cruzamento usa `id_projeto_investimento`; município, UF e código IBGE vêm da geometria publicada. Quando o nome oficial explicita município/UF e diverge de uma geometria única, usa-se essa menção com o código municipal correspondente, preservando a geometria original e sinalizando a divergência. Não se extrai município por semelhança, proximidade ou escolha aleatória. A mesma obra pode abranger vários municípios e continua representada por uma única linha, sem multiplicar seu investimento.

```powershell
python scripts/update_public_works.py --refresh
```

O comando atualiza `obras_projetos_publicos_br.csv`, o JSON e o gzip do dashboard preservando as inferências parlamentares existentes. Sem `--refresh`, reutiliza a coleta local em `outputs/obrasgov_cache/`. O pipeline principal preserva o cadastro municipal existente ao reprocessar as despesas. O relatório da coleta fica em `outputs/obrasgov_quality_report.json`.

Na aba de obras há filtros próprios de estado, cidade, tipo de registro, ano de cadastro, setor e situação. O estado pode acompanhar o recorte parlamentar ou ser escolhido independentemente. Cidade usa código IBGE e pode ser compartilhada pelo parâmetro `cidade` da URL. As referências genéricas de custo ficam disponíveis em “Referências de custo”; a visão inicial mostra projetos oficiais.

Os filtros de cidade selecionam somente as obras. Notas parlamentares não identificam município beneficiário, portanto seu excedente permanece relativo ao recorte parlamentar escolhido. O ano/mês das despesas não determina o ano do investimento da obra; o filtro separado “Ano de cadastro” usa a data real do cadastro. A situação é a atual publicada pela API. Valores são investimentos previstos, não repasses mensais ou pagamentos efetuados. Valores ausentes e municípios não informados são indicados, sem preenchimento artificial. Endereço e link para o cadastro oficial aparecem nos cartões quando disponíveis.

Coleta municipal de 22/10/2026: 154.737 projetos com UF válida, 154.306 com município identificado na fonte, abrangendo 5.569 códigos municipais. Foram sinalizadas 14.716 divergências entre municípios citados no nome oficial e na geometria cadastral. Os números representam o cadastro publicado, cuja localização pode exigir conferência no documento original. Valores oficiais simbólicos de até R$ 1.000 são preservados nos cartões e excluídos da comparação orçamentária.
