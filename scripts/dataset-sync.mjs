/** Snapshots fixados pelo código, verificados antes de substituir dados locais. */
import { createHash, randomUUID } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, open, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';
import { createGunzip } from 'node:zlib';
import { fileURLToPath } from 'node:url';

export const ROOT = fileURLToPath(new URL('../', import.meta.url));
const digest = (value) => createHash('sha256').update(value).digest('hex');
export async function hashFile(filename) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

function checkedPath(name) {
  if (typeof name !== 'string' || !/^(data|public\/data|outputs)\/[a-zA-Z0-9_.-]+$/.test(name) || name.includes('..')) {
    throw new Error(`Caminho inválido no snapshot: ${name}`);
  }
  return name;
}

export function validateManifest(manifest) {
  if (manifest.schemaVersion !== 1 || !manifest.snapshot || !Array.isArray(manifest.files) || !manifest.files.length ||
      !Array.isArray(manifest.archives) || !manifest.archives.length) {
    throw new Error('Manifesto de dados inválido ou incompatível.');
  }
  const files = new Map();
  for (const item of manifest.files) {
    checkedPath(item.path);
    if (files.has(item.path) || !/^[a-f0-9]{64}$/.test(item.sha256) || !Number.isSafeInteger(item.bytes) || item.bytes <= 0) {
      throw new Error(`Registro inválido no manifesto: ${item.path}`);
    }
    if (!['web', 'research'].includes(item.profile)) throw new Error('Perfil de dados desconhecido.');
    files.set(item.path, item);
  }
  const packed = new Set();
  for (const archive of manifest.archives) {
    const url = new URL(archive.url);
    if (url.protocol !== 'https:' || url.hostname !== 'github.com' || url.username || url.password ||
        !url.pathname.startsWith('/PedroGGC/OlhoCidadao/releases/download/') ||
        !/^[a-f0-9]{64}$/.test(archive.sha256) || !Number.isSafeInteger(archive.bytes) || archive.bytes <= 0 ||
        !['web', 'research'].includes(archive.profile) || !Array.isArray(archive.files)) {
      throw new Error('Arquivo de Release inválido no manifesto.');
    }
    for (const name of archive.files) {
      const item = files.get(name);
      if (!item || item.gunzipFrom || item.profile !== archive.profile || packed.has(name)) throw new Error(`Entrada de pacote inválida: ${name}`);
      packed.add(name);
    }
  }
  for (const item of files.values()) {
    if (item.gunzipFrom) {
      const source = files.get(item.gunzipFrom);
      if (!source || source.gunzipFrom || source.profile !== item.profile || !item.gunzipFrom.endsWith('.gz')) {
        throw new Error('Dependência gzip inválida.');
      }
    } else if (!packed.has(item.path)) throw new Error(`Arquivo sem pacote: ${item.path}`);
  }
  return files;
}

async function readJson(filename, fallback = null) {
  try { return JSON.parse(await readFile(filename, 'utf8')); } catch (error) {
    if (error.code === 'ENOENT' || error instanceof SyntaxError) return fallback;
    throw error;
  }
}

async function matches(filename, expected, memo = null, verify = false) {
  try {
    const info = await stat(filename);
    if (!info.isFile() || info.size !== expected.bytes) return null;
    const fingerprint = { size: info.size, mtime: info.mtimeMs, ctime: info.ctimeMs, sha256: expected.sha256 };
    if (!verify && memo && Object.entries(fingerprint).every(([key, value]) => memo[key] === value)) return fingerprint;
    return await hashFile(filename) === expected.sha256 ? fingerprint : null;
  } catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}

async function atomicJson(filename, value) {
  await mkdir(path.dirname(filename), { recursive: true });
  const temporary = `${filename}.${randomUUID()}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2)+'\n');
  await rename(temporary, filename);
}

async function* inflateFile(filename) {
  const source = createReadStream(filename);
  const decompressed = createGunzip();
  source.on('error', (error) => decompressed.destroy(error));
  source.pipe(decompressed);
  try { yield* decompressed; } finally { source.destroy(); decompressed.destroy(); }
}

async function publishPrepared(source, target, cache) {
  for (let attempt = 0; ; attempt++) {
    try { await rename(source, target); return; } catch (error) {
      if (!['EPERM', 'EACCES', 'EBUSY'].includes(error.code)) throw error;
      if (attempt < 4) { await new Promise((resolve) => setTimeout(resolve, 50*2**attempt)); continue; }
      // Windows pode manter a versão anterior mapeada em um editor/visualizador.
      const backup = path.join(cache, `replaced-${randomUUID()}-${path.basename(target)}`);
      await rename(target, backup);
      try { await rename(source, target); } catch (failure) { await rename(backup, target); throw failure; }
      try { await rm(backup); } catch (failure) {
        if (!['EPERM', 'EACCES', 'EBUSY'].includes(failure.code)) throw failure;
      }
      return;
    }
  }
}

async function withLock(cache, work) {
  await mkdir(cache, { recursive: true });
  const filename = path.join(cache, 'sync.lock');
  const started = Date.now();
  let lock;
  while (!lock) {
    try { lock = await open(filename, 'wx'); await lock.writeFile(String(process.pid)); }
    catch (error) {
      if (error.code !== 'EEXIST') throw error;
      const pid = Number(await readFile(filename, 'utf8').catch(() => '0'));
      if (Number.isSafeInteger(pid) && pid > 0) {
        try { process.kill(pid, 0); } catch (e) { if (e.code === 'ESRCH') { await rm(filename, { force: true }); continue; } }
      }
      if (!pid) {
        const info = await stat(filename).catch(() => null);
        if (info && Date.now()-info.mtimeMs > 10000) { await rm(filename, { force: true }); continue; }
      }
      if (Date.now()-started > 15*60*1000) throw new Error('Outra preparação dos dados não terminou. Tente novamente após concluir o outro processo.');
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
  }
  try { return await work(); } finally { await lock.close(); await rm(filename, { force: true }); }
}

async function downloadArchive(archive, filename, fetcher, log) {
  if (await matches(filename, archive, null, true)) return;
  const temporary = `${filename}.${randomUUID()}.part`;
  let failure;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      log(`Baixando ${archive.profile}: ${(archive.bytes/1e6).toFixed(1)} MB (tentativa ${attempt}/3)...`);
      const response = await fetcher(archive.url, { signal: AbortSignal.timeout(10*60*1000) });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      let received = 0, lastNotice = 0;
      async function* bounded() {
        for await (const chunk of Readable.fromWeb(response.body)) {
          received += chunk.length;
          if (received > archive.bytes) throw new Error('Pacote excedeu o tamanho declarado.');
          if (received-lastNotice >= 8*1024*1024) {
            log(`${archive.profile}: ${Math.floor(received/archive.bytes*100)}%`); lastNotice = received;
          }
          yield chunk;
        }
      }
      await pipeline(bounded(), createWriteStream(temporary, { flags: 'wx' }));
      if (!await matches(temporary, archive, null, true)) throw new Error('Checksum SHA-256 ou tamanho do download não confere.');
      await rename(temporary, filename);
      return;
    } catch (error) {
      failure = error;
      await rm(temporary, { force: true });
      if (attempt < 3) await new Promise((resolve) => setTimeout(resolve, attempt*300));
    }
  }
  throw new Error(`Falha ao baixar dados; os arquivos locais foram preservados. ${failure.message}`);
}

/** Apenas USTAR regular, com nomes, tamanhos e hashes exatamente como o manifesto. */
export async function extractArchive(filename, destination, expectedFiles) {
  const expected = new Map(expectedFiles.map((item) => [checkedPath(item.path), item]));
  const seen = new Set();
  let buffer = Buffer.alloc(0), current = null, padding = 0, ended = false;
  let expandedBytes = 0;
  const maximum = expectedFiles.reduce((sum, item) => sum+512+Math.ceil(item.bytes/512)*512, 10240);
  const octal = (bytes) => {
    const value = bytes.toString('ascii').replace(/\0.*$/, '').trim();
    if (!/^[0-7]+$/.test(value)) throw new Error('Cabeçalho TAR inválido.');
    return parseInt(value, 8);
  };
  try {
    for await (const chunk of inflateFile(filename)) {
      expandedBytes += chunk.length;
      if (expandedBytes > maximum) throw new Error('TAR excedeu o tamanho descomprimido esperado.');
      buffer = Buffer.concat([buffer, chunk]);
      while (buffer.length) {
        if (ended) { if (buffer.some((x) => x !== 0)) throw new Error('Conteúdo inesperado após o TAR.'); buffer = Buffer.alloc(0); break; }
        if (current) {
          const count = Math.min(current.remaining, buffer.length);
          const part = buffer.subarray(0, count);
          await current.handle.writeFile(part); current.hash.update(part);
          buffer = buffer.subarray(count); current.remaining -= count;
          if (current.remaining) break;
          await current.handle.close(); current.closed = true;
          if (current.hash.digest('hex') !== current.item.sha256) throw new Error(`Checksum inválido: ${current.item.path}`);
          seen.add(current.item.path); padding = (512-current.item.bytes%512)%512; current = null;
        }
        if (padding) { const count = Math.min(padding, buffer.length); buffer = buffer.subarray(count); padding -= count; if (padding) break; }
        if (buffer.length < 512) break;
        const header = buffer.subarray(0, 512); buffer = buffer.subarray(512);
        if (header.every((x) => x === 0)) { ended = true; continue; }
        const checksum = header.reduce((sum, value, i) => sum+(i >= 148 && i < 156 ? 32 : value), 0);
        if (checksum !== octal(header.subarray(148, 156))) throw new Error('Checksum do cabeçalho TAR inválido.');
        const name = header.subarray(0, 100).toString('utf8').replace(/\0.*$/, '');
        const prefix = header.subarray(345, 500).toString('utf8').replace(/\0.*$/, '');
        const item = expected.get(checkedPath(prefix ? `${prefix}/${name}` : name));
        if (!item || seen.has(item.path) || ![0, 48].includes(header[156]) || octal(header.subarray(124, 136)) !== item.bytes) {
          throw new Error(`Entrada TAR inesperada: ${name}`);
        }
        const target = path.join(destination, item.path);
        await mkdir(path.dirname(target), { recursive: true });
        current = { item, handle: await open(target, 'wx'), remaining: item.bytes, hash: createHash('sha256') };
      }
    }
    if (!ended || current || padding || seen.size !== expected.size) throw new Error('Pacote TAR incompleto.');
  } finally { if (current && !current.closed) await current.handle.close(); }
}

export async function ensureData({ root = ROOT, profile = 'full', verify = false, fetcher = fetch, log = console.log } = {}) {
  if (!['web', 'full'].includes(profile)) throw new Error('Perfil deve ser web ou full.');
  const rawManifest = await readFile(path.join(root, 'data-manifest.json'));
  const manifest = JSON.parse(rawManifest);
  const all = validateManifest(manifest);
  const files = manifest.files.filter((item) => profile === 'full' || item.profile === 'web');
  const cache = path.join(root, '.cache', 'olho-cidadao');
  return withLock(cache, async () => {
    const statePath = path.join(cache, 'verified.json');
    const old = await readJson(statePath, { files: {} });
    const state = { snapshot: manifest.snapshot, manifestSha256: digest(rawManifest), files: { ...old.files } };
    const missing = [];
    for (const item of files) {
      const fingerprint = await matches(path.join(root, item.path), item, old.files?.[item.path], verify);
      if (fingerprint) state.files[item.path] = fingerprint; else missing.push(item);
    }
    if (!missing.length) {
      await atomicJson(statePath, state);
      log(`Dados ${manifest.snapshot} íntegros (${profile}); nenhum download necessário.`);
      return { downloaded: 0, repaired: 0, snapshot: manifest.snapshot };
    }
    const stage = path.join(cache, `stage-${randomUUID()}`);
    await mkdir(stage);
    let downloaded = 0;
    const staged = new Set();
    try {
      for (const archive of manifest.archives) {
        if (!archive.files.some((name) => missing.some((item) => item.path === name))) continue;
        const cached = path.join(cache, `${archive.sha256}.tar.gz`);
        if (!await matches(cached, archive, null, true)) downloaded++;
        await downloadArchive(archive, cached, fetcher, log);
        await extractArchive(cached, stage, archive.files.map((name) => all.get(name)));
        archive.files.forEach((name) => staged.add(name));
      }
      for (const item of files.filter((entry) => entry.gunzipFrom)) {
        if (!missing.some((entry) => entry.path === item.path)) continue;
        const source = path.join(staged.has(item.gunzipFrom) ? stage : root, item.gunzipFrom);
        const target = path.join(stage, item.path);
        await mkdir(path.dirname(target), { recursive: true });
        let bytes = 0;
        async function* bounded(stream) {
          for await (const chunk of stream) { bytes += chunk.length; if (bytes > item.bytes) throw new Error('JSON excedeu o tamanho declarado.'); yield chunk; }
        }
        await pipeline(bounded(inflateFile(source)), createWriteStream(target, { flags: 'wx' }));
        if (!await matches(target, item, null, true)) throw new Error('JSON descomprimido não corresponde ao manifesto.');
        staged.add(item.path);
      }
      // Download e extração inteiros são validados antes de alterar os dados ativos.
      // O gzip é publicado por último: pesos, registros e obras mudam juntos.
      const ordered = [...missing].sort((a, b) => Number(a.path.endsWith('dashboard_data.json.gz'))-Number(b.path.endsWith('dashboard_data.json.gz')));
      for (const item of ordered) {
        if (!staged.has(item.path)) throw new Error(`Arquivo não preparado: ${item.path}`);
        const target = path.join(root, item.path);
        await mkdir(path.dirname(target), { recursive: true });
        await publishPrepared(path.join(stage, item.path), target, cache);
        state.files[item.path] = await matches(target, item, null, true);
        if (!state.files[item.path]) throw new Error(`Falha ao publicar ${item.path}`);
      }
      await atomicJson(statePath, state);
      log(`Dados ${manifest.snapshot} prontos: ${missing.length} arquivos restaurados; ${downloaded} pacotes baixados.`);
      return { downloaded, repaired: missing.length, snapshot: manifest.snapshot };
    } finally {
      if (!path.resolve(stage).startsWith(path.resolve(cache)+path.sep)) throw new Error('Diretório temporário fora do cache.');
      await rm(stage, { recursive: true, force: true });
    }
  });
}
