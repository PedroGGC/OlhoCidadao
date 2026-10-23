// A representação compacta transporta todas as notas, sem pesos ou extrapolação.
export function decodeRecords(payload) {
  if (!payload.recordColumns) return payload.records || [];
  const columns = Object.entries(payload.recordColumns);
  const records = Array.from({ length: payload.recordCount }, () => ({}));
  for (const [key, column] of columns) {
    if (column.values.length !== records.length) throw new Error(`Coluna inválida: ${key}`);
    const dictionary = column.dictionary;
    for (let index = 0; index < records.length; index += 1) {
      records[index][key] = dictionary ? dictionary[column.values[index]] : column.values[index];
    }
  }
  return records;
}

/** Colunas necessárias para filtrar (enviadas ao Web Worker). */
export const FILTER_COLUMNS = [
  'nomeParlamentar', 'cargo', 'regiao', 'siglaUF', 'siglaPartido', 'ano', 'mes',
  'tipoDespesa', 'classePredita', 'nomeFornecedor', 'cnpjCpfFornecedor',
];

export function pickFilterColumns(payload) {
  const picked = {};
  for (const key of FILTER_COLUMNS) picked[key] = payload.recordColumns[key];
  return picked;
}

/* ---- Cache local (IndexedDB) do arquivo comprimido ---- */
const DB_NAME = 'olho-cidadao';
const STORE = 'files';

function openDb() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') { reject(new Error('IndexedDB indisponível')); return; }
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idbGet(db, key) {
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE).objectStore(STORE).get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idbPut(db, key, value) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

const isGzip = (bytes) => bytes[0] === 0x1f && bytes[1] === 0x8b;

/** Baixa o .gz usando o cache local quando a versão (ETag/tamanho/data) não mudou. */
async function fetchGzipBytes(url) {
  let db = null;
  let version = '';
  try {
    db = await openDb();
    const head = await fetch(url, { method: 'HEAD' });
    if (head.ok) {
      version = [head.headers.get('etag'), head.headers.get('last-modified'), head.headers.get('content-length')]
        .filter(Boolean).join('|');
    }
    if (version) {
      const cached = await idbGet(db, url);
      if (cached && cached.version === version && cached.bytes) return new Uint8Array(cached.bytes);
    }
  } catch (error) {
    console.info('Cache local indisponível:', error.message);
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (db && version && isGzip(bytes)) {
    // Só guarda o arquivo comprimido (hosts que já descomprimem não entram no cache).
    idbPut(db, url, { version, bytes: bytes.buffer.slice(0) }).catch(() => {});
  }
  return bytes;
}

export async function loadSummary() {
  try {
    const response = await fetch('/data/resumo_geral.json');
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

export async function loadDashboardData() {
  if (typeof DecompressionStream !== 'undefined') {
    try {
      // Alguns hosts já descomprimem arquivos com Content-Encoding: gzip.
      const bytes = await fetchGzipBytes('/data/dashboard_data.json.gz');
      const blob = new Blob([bytes]);
      const stream = isGzip(bytes)
        ? blob.stream().pipeThrough(new DecompressionStream('gzip')) : blob.stream();
      return await new Response(stream).json();
    } catch (error) {
      console.info('Carregando versão JSON do dataset:', error.message);
    }
  }
  const response = await fetch('/data/dashboard_data.json');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}
