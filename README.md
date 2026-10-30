# OlhoCidadão

Consulta de despesas parlamentares e obras públicas brasileiras, com rede neural construída do zero em NumPy. A IA aprende atipicidade estatística; seus alertas não comprovam fraude. [Fontes, metodologia e resultados](docs/METODOLOGIA.md).

## Rodar do zero no Windows

### 1. Instalar os requisitos

Instale **Git para Windows** e **Node.js 22.14 ou superior**. Abra o PowerShell e instale a versão de pnpm usada pelo projeto:

```powershell
npm install --global pnpm@12.10.1
```

Para consultar o portal, não é necessário instalar Python, Git LFS, WSL ou 7-Zip, nem configurar chave de IA. Os pacotes `.tar.gz` da Release são extraídos automaticamente pelo Node.js, inclusive no Windows; não os baixe ou descompacte manualmente.

### 2. Clonar o projeto

```powershell
git clone --depth 1 https://github.com/PedroGGC/OlhoCidadao.git
cd OlhoCidadao
```

O clone com `--depth 1` evita baixar o histórico antigo com arquivos grandes. Nos próximos passos, mantenha o terminal na pasta do projeto.

### 3. Instalar e preparar os dados

```powershell
pnpm install --frozen-lockfile
```

Essa instalação executa automaticamente a preparação da versão de dados fixada em [data-manifest.json](data-manifest.json):

1. Instala as dependências do site nas versões do lockfile.
2. Baixa os pacotes `web.tar.gz` e `research.tar.gz` da Release correspondente.
3. Confere os tamanhos e os hashes SHA-256 antes de publicar os arquivos locais.
4. Restaura os CSVs em `data/`, o dashboard em `public/data/` e os relatórios em `outputs/`.
5. Ativa a preparação automática após futuras atualizações do Git, quando não houver hooks personalizados.

Na primeira instalação são cerca de **87 MB de download de dados**, além das dependências do site. A base contém **328.706 notas**, sem amostragem no portal. Com dados íntegros, as próximas preparações reutilizam os arquivos e o cache, sem baixar novamente os pacotes.

### 4. Iniciar e abrir o site

```powershell
pnpm dev
```

Abra no navegador o endereço exibido no terminal, normalmente **http://localhost:5173**. Se essa porta já estiver ocupada, use o endereço que o Vite informar. Mantenha o terminal aberto enquanto usar o site; para encerrar, pressione `Ctrl+C`.

O portal já recebe os pesos treinados e as classificações da base integral. Abrir o site **não treina a IA nem faz uma nova coleta das APIs oficiais**: todos os colaboradores usam o snapshot associado à versão do código. Para conferir a instalação, execute `pnpm test`.

## Atualizar / Sync

Após a instalação inicial, os hooks preparam dependências e dados automaticamente depois de pull, rebase ou troca de branch. Para atualizar explicitamente com um comando:

```powershell
pnpm sync
pnpm dev
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

## Se os dados não carregarem

Com conexão disponível, execute na pasta do projeto:

```powershell
pnpm setup
pnpm dev
```

`pnpm setup` também prepara os dados se a instalação tiver sido executada com `--ignore-scripts`. Para forçar uma nova conferência de todos os hashes e reparar divergências, use `pnpm data:verify`. Se o download falhar, corrija a conexão e repita a preparação; não é necessário treinar a IA para recuperar a base publicada.

Se o PowerShell bloquear `npm.ps1` ou `pnpm.ps1`, execute os mesmos comandos com `npm.cmd` ou `pnpm.cmd`, respectivamente.

## Pesquisar / treinar a IA

Para treinamento e validação científica, instale **Python 3.12+** e as dependências:

```powershell
python -m pip install -r requirements.txt
pnpm train
pnpm dev:local
```

O treino padrão usa os dados publicados em modo offline, com fit em até 5.000 notas e inferência na base integral. Para atualizar as fontes oficiais use `python main.py` após `pnpm setup`. A geração local é vista com `pnpm dev:local`; os comandos normais restauram o snapshot publicado, garantindo o mesmo resultado entre colaboradores.

Validação completa:

```powershell
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

```powershell
pnpm data:publish
```

O comando valida as inferências e os valores monetários, cria pacotes determinísticos, publica uma Release identificada pelo conteúdo, verifica os digests SHA-256 retornados pelo GitHub e só então atualiza o manifesto. Depois, faça commit e push do novo manifesto junto de qualquer alteração de código necessária. Assets de uma versão existente nunca são sobrescritos. Para apenas preparar os pacotes, use `python scripts/publish_snapshot.py`.

Os dados grandes e relatórios gerados ficam fora do Git. O JSON descompactado é reconstruído a partir do gzip, sem um segundo upload. O perfil padrão é completo; em ambientes que precisam somente do portal, `OLHO_DATA_PROFILE=web` baixa cerca de 28 MB. A versão é fixada pelo manifesto, nunca por uma Release “latest” mutável.

A limpeza preserva o histórico antigo do repositório; os próximos pushes contêm código e manifesto, sem reenviar CSVs ou JSONs completos.
