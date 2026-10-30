/** Uma entrada para instalação, sync, execução e publicação do projeto. */
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { ensureData, ROOT } from './dataset-sync.mjs';

const cache = path.join(ROOT, '.cache', 'olho-cidadao');
const notice = (message) => console.log(`[OlhoCidadão] ${message}`);

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: ROOT, stdio: 'inherit',
    shell: process.platform === 'win32' && command === 'pnpm', ...options });
  if (result.error) throw new Error(`Não foi possível executar ${command}: ${result.error.message}`);
  if (result.status !== 0) throw new Error(`${command} terminou com código ${result.status ?? result.signal}.`);
}

async function dependencyHash() {
  const hash = createHash('sha256');
  for (const filename of ['package.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml']) {
    hash.update(await readFile(path.join(ROOT, filename)));
  }
  return hash.digest('hex');
}

async function markDependencies() {
  await mkdir(cache, { recursive: true });
  await writeFile(path.join(cache, 'dependencies.sha256'), await dependencyHash());
}

async function ensureDependencies() {
  const saved = await readFile(path.join(cache, 'dependencies.sha256'), 'utf8').catch(() => '');
  if (!existsSync(path.join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js')) || saved !== await dependencyHash()) {
    notice('Dependências mudaram; instalando a versão do lockfile...');
    run('pnpm', ['install', '--frozen-lockfile']);
  }
}

async function installHooks() {
  const configured = spawnSync('git', ['config', '--local', '--get', 'core.hooksPath'], { cwd: ROOT, encoding: 'utf8' });
  if (configured.error || !existsSync(path.join(ROOT, '.git'))) return;
  const current = configured.stdout.trim();
  if (current && current !== '.githooks') {
    notice(`Hooks existentes preservados (${current}). Use pnpm sync para preparar dados após atualizar.`);
    return;
  }
  if (!current) {
    const found = spawnSync('git', ['rev-parse', '--git-path', 'hooks'], { cwd: ROOT, encoding: 'utf8' });
    const folder = path.resolve(ROOT, found.stdout.trim());
    const active = (await readdir(folder).catch(() => [])).filter((name) => !name.endsWith('.sample'));
    for (const name of active) {
      const content = await readFile(path.join(folder, name), 'utf8');
      const lines = content.split(/\r?\n/).filter((line) => line.trim());
      const standardLfs = ['post-checkout', 'post-commit', 'post-merge', 'pre-push'].includes(name) &&
        lines.length === 3 && lines[0] === '#!/bin/sh' &&
        lines[1].startsWith('command -v git-lfs >/dev/null 2>&1 || {') &&
        lines[2] === `git lfs ${name} "$@"`;
      if (!standardLfs) {
        notice(`Hook personalizado ${name} preservado. Use pnpm sync para preparar os dados.`);
        return;
      }
    }
    run('git', ['config', '--local', 'core.hooksPath', '.githooks']);
    notice('Preparação automática ativada após pull, rebase e troca de branch.');
  }
}

async function prepare({ verify = false, local = false } = {}) {
  if (local) {
    for (const file of ['public/data/dashboard_data.json.gz', 'public/data/resumo_geral.json']) {
      if (!existsSync(path.join(ROOT, file))) throw new Error(`Geração local ausente: ${file}. Execute pnpm train.`);
    }
    notice('Usando geração local de pesquisa; o snapshot publicado não será restaurado.');
    return;
  }
  return ensureData({ profile: process.env.OLHO_DATA_PROFILE || 'full', verify, log: notice });
}

const [command = 'setup', ...args] = process.argv.slice(2);
try {
  if (command === 'postinstall' || command === 'setup') {
    await installHooks();
    await prepare();
    if (command === 'postinstall') await markDependencies();
  } else if (command === 'after-sync' || command === 'sync') {
    if (command === 'sync') run('git', ['pull', '--ff-only'], { env: { ...process.env, OLHO_SKIP_GIT_HOOK: '1' } });
    await ensureDependencies();
    await installHooks();
    await prepare();
  } else if (command === 'data:verify') {
    await prepare({ verify: true });
  } else if (['dev', 'build', 'preview', 'dev:local'].includes(command)) {
    await ensureDependencies();
    await installHooks();
    await prepare({ local: command === 'dev:local' });
    run(process.execPath, [path.join(ROOT, 'node_modules/vite/bin/vite.js'),
      ...(command === 'dev' || command === 'dev:local' ? [] : [command]), ...args],
      { env: { ...process.env, OLHO_DATA_MODE: command === 'dev:local' ? 'local' : 'snapshot' } });
    if (command === 'build') {
      const uncompressed = path.join(ROOT, 'dist', 'data', 'dashboard_data.json');
      const compressed = path.join(ROOT, 'dist', 'data', 'dashboard_data.json.gz');
      if (existsSync(uncompressed) && existsSync(compressed)) {
        await rm(uncompressed, { force: true });
        notice('Otimizado para hospedagem estática/Vercel: dist/data/dashboard_data.json (>100MB) removido em favor do .gz (26MB).');
      }
    }
  } else if (command === 'train') {
    await ensureData({ profile: 'full', log: notice });
    run(process.env.PYTHON || 'python', ['main.py', ...(args.length ? args : ['--offline'])]);
    notice('Treino concluído. Para visualizar esta geração sem restaurar o snapshot: pnpm dev:local');
  } else if (command === 'test') {
    await ensureData({ profile: 'web', log: notice });
    const names = (await readdir(path.join(ROOT, 'tests'))).filter((name) => name.endsWith('.test.mjs'));
    run(process.execPath, ['--test', ...names.map((name) => path.join(ROOT, 'tests', name))]);
  } else if (command === 'data:publish') {
    // Publica a geração local, sem substituí-la pelo snapshot anterior.
    run(process.env.PYTHON || 'python', ['scripts/publish_snapshot.py', '--publish', ...args]);
  } else throw new Error(`Comando desconhecido: ${command}`);
} catch (error) {
  console.error(`[OlhoCidadão] ${error.message}`);
  console.error('Não inicie com dados incompletos. Resolva o erro e execute pnpm setup novamente.');
  process.exitCode = 1;
}
