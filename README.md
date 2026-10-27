# OlhoCidadão

Consulta de despesas parlamentares e obras públicas brasileiras, com rede neural construída do zero em NumPy. A IA aprende atipicidade estatística; seus alertas não comprovam fraude. [Fontes, metodologia e resultados](docs/METODOLOGIA.md).

## Começar

Requisitos para o portal: **Node.js 22.14+** e **pnpm 12.10.1**. Não precisa de Python, chave de IA ou Git LFS para consultar os dados publicados.

```sh
git clone --depth 1 https://github.com/PedroGGC/OlhoCidadao.git
cd OlhoCidadao
pnpm install --frozen-lockfile
pnpm dev
```

A instalação prepara automaticamente o snapshot fixado em [data-manifest.json](data-manifest.json): baixa os pacotes comprimidos, confere SHA-256 e restaura os mesmos nomes de CSV em `data/`, além do dashboard e relatórios. A base contém **325.784 notas**, sem amostragem no portal.

Na primeira instalação são cerca de **87 MB de download**. Com dados íntegros, as próximas preparações usam o cache e não acessam a rede para baixar os dados. O treinamento não é executado ao iniciar a aplicação.

## Atualizar / Sync

Após a instalação inicial, os hooks preparam dependências e dados automaticamente depois de pull, rebase ou troca de branch. Para atualizar explicitamente com um comando:

```sh
pnpm sync
```

O comando aplica `git pull --ff-only` e prepara a versão de dados exigida pelo código. Se houver conflito, falta de rede ou checksum incorreto, informa o problema; não usa silenciosamente outro snapshot. Downloads e extração são validados antes da substituição dos arquivos ativos. `pnpm dev`, `pnpm build` e `pnpm preview` também conferem os dados antes de executar.

Git não instala hooks ao clonar um repositório: execute `pnpm install` uma vez. Se você já usa hooks personalizados, eles são preservados; nesse caso, use `pnpm sync`. Se seu cliente ignorar hooks ou você usar `--ignore-scripts`, execute `pnpm setup`; a inicialização continua protegida pela conferência automática.

| Comando | Função |
| --- | --- |
| `pnpm dev` / `pnpm start` | Prepara os dados publicados e inicia o portal |
| `pnpm sync` | Atualiza código, dependências e dados |
| `pnpm setup` | Repara os arquivos necessários sem atualizar o código |
| `pnpm data:verify` | Recalcula SHA-256 e repara divergências |
| `pnpm test` | Executa testes JavaScript, incluindo integridade e sincronização |
| `pnpm build` | Prepara dados e gera o site em `dist/` |
| `pnpm preview` | Serve o build local |

## Pesquisar / treinar a IA

Para treinamento e validação científica, instale **Python 3.12+** e as dependências:

```sh
pip install -r requirements.txt
pnpm train
pnpm dev:local
```

O treino padrão usa os dados publicados em modo offline, com fit em até 5.000 notas e inferência na base integral. Para atualizar as fontes oficiais use `python main.py` após `pnpm setup`. A geração local é vista com `pnpm dev:local`; os comandos normais restauram o snapshot publicado, garantindo o mesmo resultado entre colaboradores.

Validação completa:

```sh
python -m unittest discover -s tests
pnpm test
python scripts/validate_model.py
python scripts/validate_dataset.py --check
pnpm data:verify
pnpm build
```

Relatórios e casos de paridade são restaurados em `outputs/`. Gemini é opcional apenas para o texto complementar; nenhuma revisão humana participa da classificação ou avaliação.

## Publicar novos dados (mantenedor)

Com uma geração local consistente e GitHub CLI autenticado:

```sh
pnpm data:publish
```

O comando valida as inferências e os valores monetários, cria pacotes determinísticos, publica uma Release identificada pelo conteúdo, verifica os digests SHA-256 retornados pelo GitHub e só então atualiza o manifesto. Depois, faça commit e push do novo manifesto junto de qualquer alteração de código necessária. Assets de uma versão existente nunca são sobrescritos. Para apenas preparar os pacotes, use `python scripts/publish_snapshot.py`.

Os dados grandes e relatórios gerados ficam fora do Git. O JSON descompactado é reconstruído a partir do gzip, sem um segundo upload. O perfil padrão é completo; em ambientes que precisam somente do portal, `OLHO_DATA_PROFILE=web` baixa cerca de 28 MB. A versão é fixada pelo manifesto, nunca por uma Release “latest” mutável.

A limpeza preserva o histórico antigo do repositório. O clone com `--depth 1` evita baixar esse histórico; os próximos pushes contêm código e manifesto, sem reenviar CSVs ou JSONs completos.
