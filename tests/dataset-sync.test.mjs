import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import { spawnSync } from 'node:child_process';
import { ensureData, validateManifest } from '../scripts/dataset-sync.mjs';

const sha = (value) => createHash('sha256').update(value).digest('hex');
function tar(entries) {
  const chunks = [];
  for (const [name, content, type = '0'] of entries) {
    const body = Buffer.from(content);
    const header = Buffer.alloc(512);
    header.write(name, 0, 100);
    header.write('0000644\0', 100); header.write('0000000\0', 108); header.write('0000000\0', 116);
    header.write(body.length.toString(8).padStart(11, '0')+'\0', 124);
    header.write('00000000000\0', 136); header.fill(32, 148, 156); header.write(type, 156);
    header.write('ustar\0', 257); header.write('00', 263);
    header.write(header.reduce((a, b) => a+b, 0).toString(8).padStart(6, '0')+'\0 ', 148);
    chunks.push(header, body, Buffer.alloc((512-body.length%512)%512));
  }
  return gzipSync(Buffer.concat([...chunks, Buffer.alloc(1024)]));
}

async function fixture(t, { mutateEntries = (entries) => entries, recordCount = 3, prefix = 'olho-sync-test-' } = {}) {
  const root = await mkdtemp(path.join(tmpdir(), prefix));
  t.after(async () => {
    assert.ok(path.resolve(root).startsWith(path.resolve(tmpdir())+path.sep+'olho-sync-test-'));
    await rm(root, { recursive: true, force: true });
  });
  const json = Buffer.from(JSON.stringify({ recordCount, modelParams: { version: 2 } }));
  const entries = [['public/data/dashboard_data.json.gz', gzipSync(json)],
    ['outputs/model_parity.json', Buffer.from('{"cases":[]}')],
    ['data/processed_dataset.csv', Buffer.from('valorLiquido\n100\n200\n300\n')]];
  const files = entries.map(([name, value]) => ({ path: name, bytes: value.length, sha256: sha(value),
    profile: name.startsWith('data/') ? 'research' : 'web' }));
  files.push({ path: 'public/data/dashboard_data.json', bytes: json.length, sha256: sha(json),
    gunzipFrom: 'public/data/dashboard_data.json.gz', profile: 'web' });
  const buffers = {};
  const archives = ['web', 'research'].map((profile) => {
    const selected = entries.filter(([name]) => files.find((item) => item.path === name).profile === profile);
    const archive = tar(mutateEntries(selected));
    const url = `https://github.com/PedroGGC/OlhoCidadao/releases/download/data-test/${profile}.tar.gz`;
    buffers[url] = archive;
    return { profile, url, bytes: archive.length, sha256: sha(archive), files: selected.map(([name]) => name) };
  });
  const manifest = { schemaVersion: 1, snapshot: 'data-test', files, archives };
  await writeFile(path.join(root, 'data-manifest.json'), JSON.stringify(manifest));
  let calls = 0;
  const fetcher = async (url) => { calls++; return new Response(buffers[url]); };
  return { root, manifest, json, buffers, fetcher, calls: () => calls, log: () => {} };
}

test('setup completo preserva nomes e bytes; segundo sync funciona offline sem download', async (t) => {
  const f = await fixture(t);
  const first = await ensureData(f);
  assert.equal(first.downloaded, 2);
  assert.deepEqual(await readFile(path.join(f.root, 'public/data/dashboard_data.json')), f.json);
  assert.equal(await readFile(path.join(f.root, 'data/processed_dataset.csv'), 'utf8'), 'valorLiquido\n100\n200\n300\n');
  const second = await ensureData({ ...f, verify: true, fetcher: () => { throw new Error('rede não deveria ser usada'); } });
  assert.equal(second.downloaded, 0);
  assert.equal(second.repaired, 0);
  assert.equal(f.calls(), 2);
});

test('perfil web não baixa CSV de pesquisa', async (t) => {
  const f = await fixture(t);
  assert.equal((await ensureData({ ...f, profile: 'web' })).downloaded, 1);
  await assert.rejects(readFile(path.join(f.root, 'data/processed_dataset.csv')), { code: 'ENOENT' });
});

test('corrupção local é reparada usando pacote validado em cache, mesmo offline', async (t) => {
  const f = await fixture(t);
  await ensureData(f);
  await writeFile(path.join(f.root, 'data/processed_dataset.csv'), 'arquivo alterado');
  const result = await ensureData({ ...f, fetcher: () => { throw new Error('offline'); } });
  assert.equal(result.repaired, 1);
  assert.equal(result.downloaded, 0);
  assert.equal(f.calls(), 2);
});

test('download interrompido ou inválido preserva os arquivos ativos', async (t) => {
  const f = await fixture(t);
  const target = path.join(f.root, 'outputs/model_parity.json');
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, 'geração anterior');
  await assert.rejects(ensureData({ ...f, fetcher: async () => new Response('incompleto') }), /preservados/);
  assert.equal(await readFile(target, 'utf8'), 'geração anterior');
});

test('arquivo interno corrompido impede publicar todos os arquivos do pacote', async (t) => {
  const f = await fixture(t, { mutateEntries: (entries) => entries.map(([name, bytes]) =>
    [name, name.endsWith('model_parity.json') ? Buffer.alloc(bytes.length, 65) : bytes]) });
  const target = path.join(f.root, 'outputs/model_parity.json');
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, 'geração anterior');
  await assert.rejects(ensureData(f), /Checksum inválido/);
  assert.equal(await readFile(target, 'utf8'), 'geração anterior');
  await assert.rejects(readFile(path.join(f.root, 'public/data/dashboard_data.json.gz')), { code: 'ENOENT' });
});

test('TAR com symlink é rejeitado antes de substituir dados', async (t) => {
  const f = await fixture(t, { mutateEntries: (entries) => entries.map(([name, bytes]) => [name, bytes, '2']) });
  await assert.rejects(ensureData(f), /Entrada TAR inesperada/);
});

test('manifesto rejeita caminhos externos, arquivos duplicados e URLs não permitidas', async (t) => {
  const f = await fixture(t);
  const clone = () => structuredClone(f.manifest);
  const traversal = clone(); traversal.files[0].path = '../fora.csv';
  assert.throws(() => validateManifest(traversal), /Caminho inválido/);
  const duplicate = clone(); duplicate.files.push(duplicate.files[0]);
  assert.throws(() => validateManifest(duplicate), /Registro inválido/);
  const external = clone(); external.archives[0].url = 'https://example.com/dados.tar.gz';
  assert.throws(() => validateManifest(external), /Release inválido/);
});

test('dois preparos simultâneos compartilham o snapshot sem duplicar downloads', async (t) => {
  const f = await fixture(t);
  await Promise.all([ensureData(f), ensureData(f)]);
  assert.equal(f.calls(), 2);
});

test('mudança do manifesto restaura a versão correta sem reutilizar JSON antigo', async (t) => {
  const first = await fixture(t);
  await ensureData(first);
  const second = await fixture(t, { recordCount: 4 });
  second.manifest.snapshot = 'data-next';
  await writeFile(path.join(first.root, 'data-manifest.json'), JSON.stringify(second.manifest));
  const result = await ensureData({ ...second, root: first.root });
  assert.equal(result.downloaded, 1); // CSV permaneceu idêntico; só o pacote web mudou.
  assert.deepEqual(await readFile(path.join(first.root, 'public/data/dashboard_data.json')), second.json);
  assert.equal(result.snapshot, 'data-next');
});

test('falha no segundo pacote não publica nem mesmo o primeiro pacote já validado', async (t) => {
  const f = await fixture(t);
  const target = path.join(f.root, 'outputs/model_parity.json');
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, 'geração anterior');
  const fetcher = (url, options) => url.endsWith('research.tar.gz')
    ? Promise.resolve(new Response('incompleto')) : f.fetcher(url, options);
  await assert.rejects(ensureData({ ...f, fetcher }), /preservados/);
  assert.equal(await readFile(target, 'utf8'), 'geração anterior');
  await assert.rejects(readFile(path.join(f.root, 'public/data/dashboard_data.json.gz')), { code: 'ENOENT' });
});

test('instalação ativa hooks reais: checkout e merge recuperam CSV ausente sem rede', async (t) => {
  const f = await fixture(t, { prefix: 'olho-sync-test-com espaços-' });
  const run = (command, args) => {
    const result = spawnSync(command, args, { cwd: f.root, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr+'\n'+result.stdout);
    return result.stdout;
  };
  await mkdir(path.join(f.root, 'scripts'));
  for (const name of ['dataset-sync.mjs', 'project.mjs']) {
    await writeFile(path.join(f.root, 'scripts', name), await readFile(new URL('../scripts/'+name, import.meta.url)));
  }
  await mkdir(path.join(f.root, '.githooks'));
  for (const name of ['post-checkout', 'post-merge', 'post-rewrite', 'pre-push']) {
    await writeFile(path.join(f.root, '.githooks', name), await readFile(new URL('../.githooks/'+name, import.meta.url)), { mode: 0o755 });
  }
  await writeFile(path.join(f.root, 'package.json'), '{"type":"module","packageManager":"pnpm@12.10.1"}');
  await writeFile(path.join(f.root, 'pnpm-lock.yaml'), 'lockfileVersion: "9.0"\n');
  await writeFile(path.join(f.root, 'pnpm-workspace.yaml'), 'allowBuilds: {}\n');
  await writeFile(path.join(f.root, '.gitignore'), '.cache/\ndata/\npublic/\noutputs/\nnode_modules/\n');
  await mkdir(path.join(f.root, 'node_modules/vite/bin'), { recursive: true });
  await writeFile(path.join(f.root, 'node_modules/vite/bin/vite.js'), '');
  const cache = path.join(f.root, '.cache/olho-cidadao');
  await mkdir(cache, { recursive: true });
  for (const archive of f.manifest.archives) {
    await writeFile(path.join(cache, archive.sha256+'.tar.gz'), f.buffers[archive.url]);
  }
  run('git', ['init', '-b', 'main']);
  run('git', ['config', 'user.name', 'Snapshot fixture']);
  run('git', ['config', 'user.email', 'fixture@example.invalid']);
  run('git', ['add', '.']);
  run('git', ['-c', 'commit.gpgsign=false', 'commit', '-m', 'fixture']);
  run(process.execPath, ['scripts/project.mjs', 'postinstall']);
  assert.equal(run('git', ['config', '--local', '--get', 'core.hooksPath']).trim(), '.githooks');
  const csv = path.join(f.root, 'data/processed_dataset.csv');
  await rm(csv);
  run('git', ['checkout', '-b', 'feature']);
  assert.equal(await readFile(csv, 'utf8'), 'valorLiquido\n100\n200\n300\n');
  await writeFile(path.join(f.root, 'example.txt'), 'alteração de código');
  run('git', ['add', 'example.txt']);
  run('git', ['-c', 'commit.gpgsign=false', 'commit', '-m', 'feature']);
  run('git', ['checkout', 'main']);
  await rm(csv);
  run('git', ['merge', '--ff-only', 'feature']);
  assert.equal(await readFile(csv, 'utf8'), 'valorLiquido\n100\n200\n300\n');
  // Um hook misto (LFS + código personalizado) também precisa ser preservado.
  run('git', ['config', '--local', '--unset', 'core.hooksPath']);
  const custom = '#!/bin/sh\ngit lfs post-merge "$@"\nprintf custom\n';
  const customPath = path.join(f.root, '.git/hooks/post-merge');
  await writeFile(customPath, custom);
  run(process.execPath, ['scripts/project.mjs', 'postinstall']);
  assert.equal(spawnSync('git', ['config', '--local', '--get', 'core.hooksPath'], { cwd: f.root }).status, 1);
  assert.equal(await readFile(customPath, 'utf8'), custom);
});
