/* Filtragem das notas fiscais fora da thread principal.
   Mantém apenas as colunas de filtro (códigos de dicionário em typed arrays) e devolve índices. */
const norm = (v) => String(v ?? '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
let cols = null;
let total = 0;

const SPEC = [
  ['parlamentar', 'nomeParlamentar', false],
  ['cargo', 'cargo', false],
  ['regiao', 'regiao', false],
  ['uf', 'siglaUF', false],
  ['partido', 'siglaPartido', false],
  ['ano', 'ano', true],
  ['mes', 'mes', true],
  ['categoria', 'tipoDespesa', false],
  ['alerta', 'classePredita', true],
];

function init(payload) {
  cols = {};
  for (const [key, col] of Object.entries(payload)) {
    const dictionary = col.dictionary ? col.dictionary : null;
    const values = dictionary ? Int32Array.from(col.values) : col.values;
    const lower = dictionary ? dictionary.map(norm) : null;
    cols[key] = { dictionary, values, lower };
  }
  total = cols.nomeParlamentar.values.length;
}

function lowerAt(col, i) {
  return col.lower ? col.lower[col.values[i]] : norm(col.values[i]);
}

/** Devolve Uint32Array com os índices que satisfazem o filtro `f`. */
function run(f) {
  const preds = [];
  for (const [key, field, numeric] of SPEC) {
    if (f[key] === 'all' || f[key] === undefined) continue;
    const col = cols[field];
    let target;
    if (col.dictionary) {
      target = col.dictionary.indexOf(numeric ? Number(f[key]) : f[key]);
      if (target < 0 && numeric) target = col.dictionary.indexOf(String(f[key]));
      if (target < 0) return new Uint32Array(0);
    } else {
      target = numeric ? Number(f[key]) : f[key];
    }
    preds.push([col.values, target]);
  }
  const terms = norm(f.busca).split(/\s+/).filter(Boolean);
  const searchCols = terms.length
    ? ['nomeParlamentar', 'siglaPartido', 'siglaUF', 'nomeFornecedor', 'cnpjCpfFornecedor', 'tipoDespesa'].map((k) => cols[k])
    : null;

  const out = new Uint32Array(total);
  let n = 0;
  outer: for (let i = 0; i < total; i += 1) {
    for (let p = 0; p < preds.length; p += 1) {
      if (preds[p][0][i] !== preds[p][1]) continue outer;
    }
    if (searchCols) {
      // Texto montado sob demanda (sem guardar uma string por nota na memória).
      let text = '';
      for (let c = 0; c < searchCols.length; c += 1) text += `${lowerAt(searchCols[c], i)} `;
      if (!terms.every((t) => text.includes(t))) continue;
    }
    out[n] = i;
    n += 1;
  }
  return out.slice(0, n);
}

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === 'init') {
    init(msg.cols);
    self.postMessage({ type: 'ready' });
    return;
  }
  if (msg.type === 'filter') {
    const filtered = run(msg.f);
    const otherYear = msg.otherYear ? run({ ...msg.f, ano: msg.otherYear }) : null;
    const noMonth = msg.f.mes !== 'all' ? run({ ...msg.f, mes: 'all' }) : null;
    const transfer = [filtered.buffer];
    if (otherYear) transfer.push(otherYear.buffer);
    if (noMonth) transfer.push(noMonth.buffer);
    self.postMessage({ type: 'result', id: msg.id, filtered, otherYear, noMonth }, transfer);
  }
};
