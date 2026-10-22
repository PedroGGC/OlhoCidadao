/* =========================================================================
   OlhoCidadão — Portal de consulta de gastos de Deputados e Senadores
   ========================================================================= */

const brl = new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' });
const brlCompact = new Intl.NumberFormat('pt-BR', {
  style: 'currency', currency: 'BRL', notation: 'compact', maximumFractionDigits: 1,
});
const intFmt = new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 0 });
const pctFmt = new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 1, signDisplay: 'exceptZero' });

const MESES_CURTOS = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'];
const MESES_NOMES = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
  'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'];

/** Campos de filtro ↔ parâmetros de URL (permite compartilhar a consulta). */
const FILTERS = [
  { id: 'f-busca', key: 'busca', param: 'q', type: 'text' },
  { id: 'f-parlamentar', key: 'parlamentar', param: 'p' },
  { id: 'f-cargo', key: 'cargo', param: 'casa' },
  { id: 'f-regiao', key: 'regiao', param: 'regiao' },
  { id: 'f-uf', key: 'uf', param: 'uf' },
  { id: 'f-partido', key: 'partido', param: 'partido' },
  { id: 'f-ano', key: 'ano', param: 'ano' },
  { id: 'f-mes', key: 'mes', param: 'mes' },
  { id: 'f-categoria', key: 'categoria', param: 'despesa' },
  { id: 'f-alerta', key: 'alerta', param: 'alerta' },
];

const STORAGE_PAGE = 'olho-page-size';
const STORAGE_THEME = 'color-scheme';

const state = {
  data: null,
  all: [],
  filtered: [],
  rankMode: 'parlamentar',
  page: 1,
  pageSize: 30,
  sort: { key: null, dir: 'desc' },
  mediaGlobal: 0,
};

const $ = (id) => document.getElementById(id);
const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Escapa texto vindo das bases públicas antes de inserir via innerHTML. */
function esc(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function safeStorageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function safeStorageSet(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch { /* armazenamento bloqueado: segue sem persistir */ }
}

let toastTimer;
function showToast(message) {
  const el = $('toast');
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2400);
}

const AVATAR_FALLBACK = 'data:image/svg+xml;utf8,' + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="#8d99ab" stroke-width="1.5"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 4-6 8-6s8 2 8 6"/></svg>'
);

/* -------------------------------------------------------------------------
   Rede neural no navegador (usada no simulador da Ficha Técnica)
   ------------------------------------------------------------------------- */
function predictMlpBrowser(features, modelParams) {
  const { mean, std, W1, b1, W2, b2 } = modelParams;
  const xNorm = features.map((val, i) => (val - mean[i]) / (std[i] + 1e-8));

  const a1 = b1.map((bias, j) => {
    let z = bias;
    for (let i = 0; i < xNorm.length; i++) z += xNorm[i] * W1[i][j];
    return Math.tanh(z);
  });

  const z2 = b2.map((bias, k) => {
    let s = bias;
    for (let j = 0; j < a1.length; j++) s += a1[j] * W2[j][k];
    return s;
  });

  const maxZ = Math.max(...z2);
  const exps = z2.map((v) => Math.exp(v - maxZ));
  const sum = exps.reduce((a, v) => a + v, 0);
  const probs = exps.map((v) => v / sum);
  return { classePredita: probs[1] >= 0.5 ? 1 : 0, probabilidadeAtipico: probs[1], a1 };
}

/* -------------------------------------------------------------------------
   Abas (com navegação por setas do teclado)
   ------------------------------------------------------------------------- */
function setupTabs() {
  const tabs = Array.from(document.querySelectorAll('.tab-btn'));
  const activate = (btn, focus = false) => {
    tabs.forEach((b) => {
      const on = b === btn;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', String(on));
      b.tabIndex = on ? 0 : -1;
      const panel = $(b.getAttribute('aria-controls'));
      panel.hidden = !on;
    });
    if (focus) btn.focus();
  };
  tabs.forEach((btn, idx) => {
    btn.tabIndex = btn.classList.contains('active') ? 0 : -1;
    btn.addEventListener('click', () => activate(btn));
    btn.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') activate(tabs[(idx + 1) % tabs.length], true);
      if (e.key === 'ArrowLeft') activate(tabs[(idx - 1 + tabs.length) % tabs.length], true);
    });
  });
}

/* -------------------------------------------------------------------------
   Diálogos (light dismiss com fallback para navegadores sem `closedby`)
   ------------------------------------------------------------------------- */
function setupDialogs() {
  const dialogs = document.querySelectorAll('dialog.app-dialog');
  if (!('closedBy' in HTMLDialogElement.prototype)) {
    dialogs.forEach((dialog) => {
      dialog.addEventListener('click', (event) => {
        if (event.target !== dialog) return;
        const rect = dialog.getBoundingClientRect();
        const inside = rect.top <= event.clientY && event.clientY <= rect.top + rect.height
          && rect.left <= event.clientX && event.clientX <= rect.left + rect.width;
        if (!inside) dialog.close();
      });
    });
  }
  $('btn-open-help').addEventListener('click', () => $('dlg-help').showModal());
  $('btn-open-settings').addEventListener('click', () => {
    syncSettingsUI();
    $('dlg-settings').showModal();
  });
}

/* -------------------------------------------------------------------------
   Preferências: tema e itens por página
   ------------------------------------------------------------------------- */
function currentThemeValue() {
  return document.querySelector('meta[name="color-scheme"]').content;
}

function applyTheme(value) {
  document.querySelector('meta[name="color-scheme"]').content = value;
  // Escuro é o padrão do portal: só persistimos quando o usuário escolhe outra opção.
  safeStorageSet(STORAGE_THEME, value === 'dark' ? null : value);
}

function syncSettingsUI() {
  const theme = currentThemeValue();
  document.querySelectorAll('input[name="theme"]').forEach((r) => { r.checked = r.value === theme; });
  const sysDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  $('theme-system-label').textContent = `Seguir o sistema (atualmente ${sysDark ? 'escuro' : 'claro'})`;
  $('pref-page-size').value = String(state.pageSize);
}

function setupSettings() {
  document.querySelectorAll('input[name="theme"]').forEach((radio) => {
    radio.addEventListener('change', () => applyTheme(radio.value));
  });
  window.addEventListener('storage', (e) => {
    if (e.key === STORAGE_THEME) {
      document.querySelector('meta[name="color-scheme"]').content = e.newValue ?? 'dark';
    }
  });
  $('pref-page-size').addEventListener('change', (e) => {
    state.pageSize = parseInt(e.target.value, 10) || 30;
    safeStorageSet(STORAGE_PAGE, String(state.pageSize));
    state.page = 1;
    renderTable();
  });
}

/* -------------------------------------------------------------------------
   Filtros
   ------------------------------------------------------------------------- */
function populateFilterOptions(records) {
  const parl = new Map();
  const ufs = new Set();
  const partidos = new Set();
  const cats = new Set();

  records.forEach((r) => {
    const cargo = r.cargo === 'SENADOR DA REPÚBLICA' ? 'Sen.' : 'Dep.';
    parl.set(r.nomeParlamentar, `${r.nomeParlamentar} (${cargo} ${r.siglaPartido}-${r.siglaUF})`);
    ufs.add(r.siglaUF);
    partidos.add(r.siglaPartido);
    cats.add(r.tipoDespesa);
  });

  const sortedParl = Array.from(parl.entries()).sort((a, b) => a[0].localeCompare(b[0], 'pt-BR'));
  $('f-parlamentar').innerHTML = `<option value="all">Todos os parlamentares (${sortedParl.length})</option>`
    + sortedParl.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join('');
  $('dl-busca').innerHTML = sortedParl.map(([v]) => `<option value="${esc(v)}"></option>`).join('');

  const fill = (id, set, label) => {
    $(id).innerHTML = `<option value="all">${label}</option>`
      + Array.from(set).sort((a, b) => a.localeCompare(b, 'pt-BR'))
        .map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
  };
  fill('f-uf', ufs, 'Todos');
  fill('f-partido', partidos, 'Todos');
  fill('f-categoria', cats, 'Todas');
}

function getFilterState() {
  const f = {};
  FILTERS.forEach(({ id, key, type }) => {
    const el = $(id);
    if (type === 'check') f[key] = el.checked;
    else if (type === 'text') f[key] = el.value.trim().toLowerCase();
    else f[key] = el.value;
  });
  return f;
}

function setFilterValue(key, value) {
  const def = FILTERS.find((x) => x.key === key);
  if (!def) return;
  const el = $(def.id);
  if (def.type === 'check') el.checked = Boolean(value);
  else el.value = value;
}

function resetFilters() {
  FILTERS.forEach(({ id, type }) => {
    const el = $(id);
    if (type === 'check') el.checked = false;
    else if (type === 'text') el.value = '';
    else el.value = 'all';
  });
  applyFilters();
}

function matchRecord(r, f) {
  if (f.parlamentar !== 'all' && r.nomeParlamentar !== f.parlamentar) return false;
  if (f.cargo !== 'all' && r.cargo !== f.cargo) return false;
  if (f.regiao !== 'all' && r.regiao !== f.regiao) return false;
  if (f.uf !== 'all' && r.siglaUF !== f.uf) return false;
  if (f.partido !== 'all' && r.siglaPartido !== f.partido) return false;
  if (f.ano !== 'all' && String(r.ano) !== f.ano) return false;
  if (f.mes !== 'all' && String(r.mes) !== f.mes) return false;
  if (f.categoria !== 'all' && r.tipoDespesa !== f.categoria) return false;
  if (f.alerta !== 'all' && String(r.classePredita) !== f.alerta) return false;
  if (f.busca && !r._search.includes(f.busca)) return false;
  return true;
}

function scopeDescription(f) {
  const parts = [];
  if (f.parlamentar !== 'all') parts.push(f.parlamentar);
  if (f.cargo !== 'all') parts.push(f.cargo === 'SENADOR DA REPÚBLICA' ? 'Senado' : 'Câmara');
  if (f.partido !== 'all') parts.push(f.partido);
  if (f.uf !== 'all') parts.push(f.uf);
  else if (f.regiao !== 'all') parts.push(`Região ${f.regiao}`);
  if (f.mes !== 'all') parts.push(MESES_NOMES[Number(f.mes) - 1]);
  if (f.ano !== 'all') parts.push(f.ano);
  if (f.categoria !== 'all') parts.push(f.categoria.toLowerCase());
  if (f.alerta === '1') parts.push('com alerta');
  if (f.alerta === '0') parts.push('dentro da média');
  if (f.busca) parts.push(`"${f.busca}"`);
  return parts.length ? parts.join(' · ') : 'Visão geral · Câmara e Senado';
}

function renderActiveChips(f) {
  const bar = $('active-filters-bar');
  const chips = [];
  const add = (key, label) => chips.push({ key, label });
  if (f.busca) add('busca', `Busca: "${f.busca}"`);
  if (f.parlamentar !== 'all') add('parlamentar', f.parlamentar);
  if (f.cargo !== 'all') add('cargo', f.cargo === 'SENADOR DA REPÚBLICA' ? 'Senado' : 'Câmara');
  if (f.regiao !== 'all') add('regiao', `Região ${f.regiao}`);
  if (f.uf !== 'all') add('uf', `UF ${f.uf}`);
  if (f.partido !== 'all') add('partido', f.partido);
  if (f.ano !== 'all') add('ano', f.ano);
  if (f.mes !== 'all') add('mes', MESES_NOMES[Number(f.mes) - 1]);
  if (f.categoria !== 'all') add('categoria', f.categoria);
  if (f.alerta !== 'all') add('alerta', f.alerta === '1' ? 'Com alerta' : 'Dentro da média');

  if (!chips.length) {
    bar.hidden = true;
    bar.innerHTML = '';
    return;
  }
  bar.hidden = false;
  bar.innerHTML = '<span>Filtros ativos:</span>' + chips.map((c) => `
    <span class="filter-chip">${esc(c.label)}
      <button type="button" class="chip-remove" data-clear="${c.key}" aria-label="Remover filtro ${esc(c.label)}">✕</button>
    </span>`).join('');
}

/* -------------------------------------------------------------------------
   URL compartilhável
   ------------------------------------------------------------------------- */
function syncUrl(f) {
  const params = new URLSearchParams();
  FILTERS.forEach(({ key, param, type }) => {
    const v = f[key];
    if (type === 'check' ? v : (v && v !== 'all')) params.set(param, type === 'check' ? '1' : v);
  });
  const qs = params.toString();
  history.replaceState(null, '', qs ? `?${qs}` : location.pathname);
}

function restoreFromUrl() {
  const params = new URLSearchParams(location.search);
  FILTERS.forEach(({ id, param, type }) => {
    if (!params.has(param)) return;
    const el = $(id);
    const v = params.get(param);
    if (type === 'check') el.checked = v === '1';
    else if (type === 'text') el.value = v;
    else if (Array.from(el.options).some((o) => o.value === v)) el.value = v;
  });
}

/* -------------------------------------------------------------------------
   Perfil, KPIs e resumo inteligente
   ------------------------------------------------------------------------- */
function renderProfile(f, rows) {
  const banner = $('parlamentar-profile-banner');
  if (f.parlamentar === 'all' || !rows.length) {
    banner.hidden = true;
    banner.innerHTML = '';
    return;
  }
  const s = rows[0];
  const cargo = s.cargo === 'SENADOR DA REPÚBLICA' ? 'Senador(a)' : 'Deputado(a) Federal';
  const total = rows.reduce((a, r) => a + r.valorLiquido, 0);
  const alertas = rows.filter((r) => r.classePredita === 1).length;
  const avatar = s.urlFoto && s.urlFoto.startsWith('http') ? s.urlFoto : AVATAR_FALLBACK;

  banner.hidden = false;
  banner.innerHTML = `
    <div class="profile-left">
      <img src="${esc(avatar)}" alt="Foto de ${esc(s.nomeParlamentar)}" class="profile-avatar" loading="lazy" />
      <div>
        <h2 class="profile-name">${esc(s.nomeParlamentar)}</h2>
        <div class="profile-meta">
          <span class="profile-tag">${cargo}</span>
          <span class="profile-tag">${esc(s.siglaPartido)}</span>
          <span class="profile-tag">${esc(s.siglaUF)} · ${esc(s.regiao)}</span>
        </div>
      </div>
    </div>
    <div class="profile-right">
      <span class="kpi-title">No filtro atual</span>
      <strong class="kpi-number" style="display:block">${brl.format(total)}</strong>
      <span class="kpi-sub">${intFmt.format(rows.length)} notas · ${intFmt.format(alertas)} com alerta</span>
    </div>`;
  const img = banner.querySelector('img');
  img.addEventListener('error', () => { img.src = AVATAR_FALLBACK; }, { once: true });
}

function topEntry(rows, keyFn) {
  const m = new Map();
  rows.forEach((r) => m.set(keyFn(r), (m.get(keyFn(r)) || 0) + r.valorLiquido));
  return Array.from(m.entries()).sort((a, b) => b[1] - a[1])[0] || ['—', 0];
}

function renderKpisAndInsight(f, rows) {
  const total = rows.reduce((a, r) => a + r.valorLiquido, 0);
  const n = rows.length;
  const media = n ? total / n : 0;
  const pessoas = new Set(rows.map((r) => r.nomeParlamentar)).size;
  const alertas = rows.filter((r) => r.classePredita === 1);
  const excesso = alertas.reduce((a, r) => a + r.excessoMensal, 0);
  const impacto = alertas.reduce((a, r) => a + r.impacto5Anos, 0);
  const merendas = Math.floor(impacto / 1.5);

  $('kpi-total-gasto').textContent = brl.format(total);
  $('kpi-total-notas').textContent = `${intFmt.format(n)} notas fiscais`;
  $('kpi-media-nota').textContent = brl.format(media);
  $('kpi-total-parlamentares').textContent = `${intFmt.format(pessoas)} parlamentar(es)`;
  $('kpi-total-atipicos').textContent = `${intFmt.format(alertas.length)} notas`;
  $('kpi-excesso-mensal').textContent = n
    ? `${((alertas.length / n) * 100).toFixed(1)}% do recorte · excesso de ${brl.format(excesso)}`
    : '—';
  $('kpi-equiv-merendas').textContent = intFmt.format(merendas);

  // Comparação com o outro ano disponível (mesmos filtros)
  const deltaEl = $('kpi-delta');
  if (f.ano !== 'all') {
    const outro = f.ano === '2026' ? '2025' : '2026';
    const totalOutro = state.all
      .filter((r) => matchRecord(r, { ...f, ano: outro }))
      .reduce((a, r) => a + r.valorLiquido, 0);
    if (totalOutro > 0) {
      const delta = ((total - totalOutro) / totalOutro) * 100;
      deltaEl.hidden = false;
      deltaEl.className = `kpi-delta ${delta > 0 ? 'up' : 'down'}`;
      deltaEl.textContent = `${pctFmt.format(delta)}% vs. ${outro} (${brlCompact.format(totalOutro)})`;
    } else {
      deltaEl.hidden = true;
    }
  } else {
    deltaEl.hidden = true;
  }

  const scope = scopeDescription(f);
  $('ai-context-tag').textContent = scope;
  $('impact-scope-label').textContent = scope;
  $('impact-total-5y').textContent = brl.format(impacto);
  renderImpactCards(impacto);

  const body = $('ai-insight-body');
  if (!n) {
    body.innerHTML = '<p>Nenhuma nota corresponde aos filtros atuais. Remova algum filtro para ver o resumo.</p>';
    return;
  }

  const [topCat, topCatVal] = topEntry(rows, (r) => r.tipoDespesa);
  const [topForn, topFornVal] = topEntry(rows, (r) => r.nomeFornecedor);
  const [topUF] = topEntry(rows, (r) => r.siglaUF);
  const [topPart, topPartVal] = topEntry(rows, (r) => r.siglaPartido);

  const diff = state.mediaGlobal ? ((media - state.mediaGlobal) / state.mediaGlobal) * 100 : 0;
  const comparativo = Math.abs(diff) < 5
    ? `em linha com a média geral (${brl.format(state.mediaGlobal)})`
    : `<strong>${Math.abs(diff).toFixed(1)}% ${diff > 0 ? 'acima' : 'abaixo'}</strong> da média geral (${brl.format(state.mediaGlobal)})`;
  const pctCat = total ? ((topCatVal / total) * 100).toFixed(0) : 0;

  let contexto;
  if (f.parlamentar !== 'all') {
    contexto = `<p><strong>${esc(f.parlamentar)}</strong> registrou <strong>${intFmt.format(n)} notas</strong> somando
      <strong>${brl.format(total)}</strong>. O valor médio por nota está ${comparativo}.
      A maior parte foi para <strong>${esc(topCat.toLowerCase())}</strong> (${pctCat}% do total), e o fornecedor que mais recebeu foi
      <strong>${esc(topForn)}</strong> (${brl.format(topFornVal)}).</p>`;
  } else if (f.partido !== 'all' || f.uf !== 'all' || f.regiao !== 'all' || f.cargo !== 'all') {
    contexto = `<p>Neste recorte (<strong>${esc(scope)}</strong>) há <strong>${intFmt.format(pessoas)} parlamentares</strong> e
      <strong>${intFmt.format(n)} notas</strong>, totalizando <strong>${brl.format(total)}</strong>. O valor médio por nota está ${comparativo}.
      A principal despesa é <strong>${esc(topCat.toLowerCase())}</strong> (${pctCat}%), com maior concentração em
      <strong>${esc(topUF)}</strong> e no partido <strong>${esc(topPart)}</strong>.</p>`;
  } else {
    contexto = `<p>Somando Câmara e Senado, a amostra consultada reúne <strong>${intFmt.format(pessoas)} parlamentares</strong> e
      <strong>${brl.format(total)}</strong> em <strong>${intFmt.format(n)} notas</strong>. A despesa de maior peso é
      <strong>${esc(topCat.toLowerCase())}</strong> (${pctCat}%). O partido com maior volume é <strong>${esc(topPart)}</strong>
      (${brl.format(topPartVal)}) e o estado com maior concentração é <strong>${esc(topUF)}</strong>.</p>`;
  }

  const ia = alertas.length
    ? `<p>A IA sinalizou <strong>${intFmt.format(alertas.length)} notas</strong> bem acima do padrão da própria categoria
      (excesso de <strong>${brl.format(excesso)}</strong>). Se isso se repetisse por 5 anos, o valor pagaria
      <strong>${intFmt.format(merendas)} merendas escolares</strong> ou <strong>${intFmt.format(Math.floor(impacto / 65))} consultas no SUS</strong>.
      Alertas não são acusação — são um convite para conferir a nota.</p>`
    : '<p>Nenhuma nota deste recorte ficou acima do padrão histórico da sua categoria.</p>';

  body.innerHTML = contexto + ia;
}

/* -------------------------------------------------------------------------
   Gráficos
   ------------------------------------------------------------------------- */
function renderMonthly(rows, f) {
  const normal = new Array(12).fill(0);
  const alert = new Array(12).fill(0);
  rows.forEach((r) => {
    const i = Math.min(11, Math.max(0, r.mes - 1));
    if (r.classePredita === 1) alert[i] += r.valorLiquido;
    else normal[i] += r.valorLiquido;
  });
  const totals = normal.map((v, i) => v + alert[i]);
  const max = Math.max(...totals, 1);

  $('chart-monthly').innerHTML = MESES_CURTOS.map((label, i) => {
    const selected = f.mes === String(i + 1);
    return `
      <button type="button" class="month-col${selected ? ' selected' : ''}" data-month="${i + 1}"
        aria-pressed="${selected}" aria-label="${MESES_NOMES[i]}: ${brl.format(totals[i])}"
        title="${MESES_NOMES[i]}: ${brl.format(totals[i])}">
        <span class="month-val-top">${totals[i] ? brlCompact.format(totals[i]) : '—'}</span>
        <span class="month-bars-stack">
          <span class="m-bar-alert" style="block-size:${Math.round((alert[i] / max) * 140)}px"></span>
          <span class="m-bar-normal" style="block-size:${Math.round((normal[i] / max) * 140)}px"></span>
        </span>
        <span class="month-label">${label}</span>
      </button>`;
  }).join('');
}

function renderCategories(rows, f) {
  const m = new Map();
  rows.forEach((r) => m.set(r.tipoDespesa, (m.get(r.tipoDespesa) || 0) + r.valorLiquido));
  const sorted = Array.from(m.entries()).sort((a, b) => b[1] - a[1]).slice(0, 7);
  const total = rows.reduce((a, r) => a + r.valorLiquido, 0) || 1;
  const max = sorted.length ? sorted[0][1] : 1;

  $('chart-categories').innerHTML = sorted.length
    ? sorted.map(([cat, val]) => {
      const selected = f.categoria === cat;
      return `
        <button type="button" class="bar-row${selected ? ' selected' : ''}" data-cat="${esc(cat)}" aria-pressed="${selected}">
          <span class="bar-row-top">
            <span class="bar-row-name">${esc(cat)}</span>
            <span class="bar-row-val">${brl.format(val)} · ${((val / total) * 100).toFixed(1)}%</span>
          </span>
          <span class="bar-track"><span class="bar-fill" style="inline-size:${Math.max(3, (val / max) * 100)}%; display:block"></span></span>
        </button>`;
    }).join('')
    : '<p class="card-subtitle">Sem despesas para exibir.</p>';
}

function renderRanking(rows) {
  const groups = new Map();
  rows.forEach((r) => {
    let key;
    let sub;
    switch (state.rankMode) {
      case 'partido': key = r.siglaPartido; sub = 'Partido'; break;
      case 'uf': key = r.siglaUF; sub = `Região ${r.regiao}`; break;
      case 'regiao': key = r.regiao; sub = 'Região'; break;
      case 'fornecedor': key = r.nomeFornecedor; sub = r.cnpjCpfFornecedor; break;
      default:
        key = r.nomeParlamentar;
        sub = `${r.cargo === 'SENADOR DA REPÚBLICA' ? 'Senador(a)' : 'Deputado(a)'} · ${r.siglaPartido}-${r.siglaUF}`;
    }
    const g = groups.get(key) || { total: 0, count: 0, alerts: 0, sub };
    g.total += r.valorLiquido;
    g.count += 1;
    if (r.classePredita === 1) g.alerts += 1;
    groups.set(key, g);
  });

  const sorted = Array.from(groups.entries()).sort((a, b) => b[1].total - a[1].total).slice(0, 12);
  $('chart-ranking').innerHTML = sorted.length
    ? sorted.map(([key, g], i) => `
      <button type="button" class="rank-card" data-key="${esc(key)}" title="Filtrar por ${esc(key)}">
        <span class="rank-pos">${i + 1}</span>
        <span class="rank-info">
          <span class="rank-title" style="display:block">${esc(key)}</span>
          <span class="rank-sub" style="display:block">${esc(g.sub)} · ${g.count} notas${g.alerts ? ` · ${g.alerts} alertas` : ''}</span>
        </span>
        <span class="rank-amount">${brl.format(g.total)}</span>
      </button>`).join('')
    : '<p class="card-subtitle">Sem dados para o ranking.</p>';
}

/* -------------------------------------------------------------------------
   Tabela (ordenação, paginação, ações por linha)
   ------------------------------------------------------------------------- */
function sortedRows() {
  const { key, dir } = state.sort;
  if (!key) return state.filtered;
  const mult = dir === 'asc' ? 1 : -1;
  const getter = {
    nomeParlamentar: (r) => r.nomeParlamentar,
    data: (r) => r.ano * 100 + r.mes,
    valorLiquido: (r) => r.valorLiquido,
    diffPct: (r) => r.diffPct,
  }[key];
  return [...state.filtered].sort((a, b) => {
    const va = getter(a);
    const vb = getter(b);
    if (typeof va === 'string') return va.localeCompare(vb, 'pt-BR') * mult;
    return (va - vb) * mult;
  });
}

function updateSortHeaders() {
  document.querySelectorAll('#audit-table thead th[data-col]').forEach((th) => {
    if (th.dataset.col === state.sort.key) {
      th.setAttribute('aria-sort', state.sort.dir === 'asc' ? 'ascending' : 'descending');
    } else {
      th.removeAttribute('aria-sort');
    }
  });
}

function renderTable() {
  const rows = sortedRows();
  const totalPages = Math.max(1, Math.ceil(rows.length / state.pageSize));
  state.page = Math.min(state.page, totalPages);
  const start = (state.page - 1) * state.pageSize;
  const pageItems = rows.slice(start, start + state.pageSize);

  $('table-count-label').textContent = rows.length
    ? `Mostrando ${start + 1}–${start + pageItems.length} de ${intFmt.format(rows.length)} notas`
    : 'Nenhuma nota no filtro atual';
  $('page-indicator').textContent = `Página ${state.page} de ${totalPages}`;
  $('btn-prev-page').disabled = state.page <= 1;
  $('btn-next-page').disabled = state.page >= totalPages;
  $('table-empty').hidden = rows.length > 0;
  $('audit-table').hidden = rows.length === 0;
  updateSortHeaders();

  $('audit-tbody').innerHTML = pageItems.map((r) => {
    const cargo = r.cargo === 'SENADOR DA REPÚBLICA' ? 'Sen.' : 'Dep.';
    const badge = r.classePredita === 1
      ? `<span class="badge-atipico">Alerta · ${(r.probabilidadeAtipico * 100).toFixed(0)}%</span>`
      : '<span class="badge-normal">Dentro da média</span>';
    const diffCls = r.diffPct > 0 ? 'diff-up' : 'diff-down';
    const doc = r.urlDocumento && r.urlDocumento.startsWith('http')
      ? `<a href="${esc(r.urlDocumento)}" target="_blank" rel="noopener noreferrer" class="btn-mini" aria-label="Abrir nota oficial (nova aba)">Nota ↗</a>`
      : '';
    return `
      <tr>
        <th scope="row">
          <span class="cell-person">
            <button type="button" class="person-link-btn" data-person="${esc(r.nomeParlamentar)}">${esc(r.nomeParlamentar)}</button>
            <span class="role-pill">${cargo}</span>
          </span>
        </th>
        <td>${esc(r.siglaPartido)} · ${esc(r.siglaUF)}</td>
        <td class="mono">${String(r.mes).padStart(2, '0')}/${r.ano}</td>
        <td>${esc(r.tipoDespesa)}</td>
        <td class="cell-forn">${esc(r.nomeFornecedor)}<small>${esc(r.cnpjCpfFornecedor)}</small></td>
        <td class="mono"><strong>${brl.format(r.valorLiquido)}</strong></td>
        <td class="mono ${diffCls}">${pctFmt.format(r.diffPct)}%</td>
        <td>${badge}</td>
        <td><span class="row-actions">
          <button type="button" class="btn-mini" data-detail="${r._idx}">Detalhes</button>${doc}
        </span></td>
      </tr>`;
  }).join('');
}

/* -------------------------------------------------------------------------
   Diálogo de detalhes da nota
   ------------------------------------------------------------------------- */
function openDetail(idx) {
  const r = state.all[idx];
  if (!r) return;
  const isAlert = r.classePredita === 1;
  const ref = Math.max(r.valorLiquido, r.limiarCategoria, r.mediaAceitavelCategoria, 1);
  const cargo = r.cargo === 'SENADOR DA REPÚBLICA' ? 'Senador(a)' : 'Deputado(a) Federal';
  const vezes = r.mediaAceitavelCategoria > 0 ? r.valorLiquido / r.mediaAceitavelCategoria : 0;

  const explain = isAlert
    ? `<strong>Por que recebeu alerta?</strong> Esta nota de ${brl.format(r.valorLiquido)} é cerca de
       <strong>${vezes.toFixed(1)}×</strong> o valor típico de "${esc(r.tipoDespesa.toLowerCase())}" (${brl.format(r.mediaAceitavelCategoria)}).
       A IA estimou ${(r.probabilidadeAtipico * 100).toFixed(0)}% de chance de ser um gasto fora do padrão.
       Repetido por 5 anos, o excesso somaria ${brl.format(r.impacto5Anos)} — o equivalente a
       ${intFmt.format(Math.floor(r.impacto5Anos / 1.5))} merendas escolares.`
    : `<strong>Dentro do esperado.</strong> O valor está compatível com o padrão de "${esc(r.tipoDespesa.toLowerCase())}"
       (média típica de ${brl.format(r.mediaAceitavelCategoria)}; alerta a partir de ${brl.format(r.limiarCategoria)}).`;

  const doc = r.urlDocumento && r.urlDocumento.startsWith('http')
    ? `<a class="btn-primary" href="${esc(r.urlDocumento)}" target="_blank" rel="noopener noreferrer">Abrir nota oficial ↗</a>`
    : '';

  $('dlg-detail-body').innerHTML = `
    <div class="detail-grid">
      <div class="detail-item"><span>Parlamentar</span><strong>${esc(r.nomeParlamentar)}</strong></div>
      <div class="detail-item"><span>Cargo</span><strong>${cargo} · ${esc(r.siglaPartido)}-${esc(r.siglaUF)}</strong></div>
      <div class="detail-item"><span>Competência</span><strong>${MESES_NOMES[r.mes - 1]} de ${r.ano}</strong></div>
      <div class="detail-item"><span>Despesa</span><strong>${esc(r.tipoDespesa)}</strong></div>
      <div class="detail-item"><span>Fornecedor</span><strong>${esc(r.nomeFornecedor)}</strong></div>
      <div class="detail-item"><span>CNPJ/CPF</span><strong class="mono">${esc(r.cnpjCpfFornecedor)}</strong></div>
    </div>
    <div class="compare-bar" aria-hidden="true">
      <span>Valor da nota: <strong>${brl.format(r.valorLiquido)}</strong></span>
      <div class="compare-track"><div class="compare-fill val${isAlert ? ' alert' : ''}" style="inline-size:${(r.valorLiquido / ref) * 100}%"></div></div>
      <span>Média típica da categoria: <strong>${brl.format(r.mediaAceitavelCategoria)}</strong></span>
      <div class="compare-track"><div class="compare-fill avg" style="inline-size:${(r.mediaAceitavelCategoria / ref) * 100}%"></div></div>
    </div>
    <div class="detail-explain ${isAlert ? 'alert' : 'ok'}">${explain}</div>
    <div class="dialog-actions">
      ${doc}
      <button type="button" class="btn-secondary" data-dlg-person="${esc(r.nomeParlamentar)}">Ver tudo deste parlamentar</button>
      <button type="button" class="btn-ghost" data-dlg-forn="${esc(r.nomeFornecedor)}">Ver notas deste fornecedor</button>
    </div>`;
  $('dlg-detail').showModal();
}

/* -------------------------------------------------------------------------
   Aba 2 — serviços equivalentes
   ------------------------------------------------------------------------- */
function renderImpactCards(total) {
  const c = state.data.custosReferencia;
  const items = [
    ['Educação', c.merenda_escolar_pnae, 'merendas escolares (R$ 1,50 cada)'],
    ['Material escolar', c.kit_escolar_aluno, 'kits escolares anuais (R$ 120)'],
    ['Inclusão digital', c.computador_escola_publica, 'computadores para escolas (R$ 2.800)'],
    ['Saúde (SUS)', c.consulta_atencao_basica, 'consultas na atenção básica (R$ 65)'],
    ['Farmácia básica', c.kit_medicamentos_basicos, 'kits mensais de medicamentos (R$ 45)'],
    ['Segurança pública', c.equipamento_seguranca_uf, 'diárias de patrulhamento (R$ 420)'],
    ['Assistência social', c.cesta_basica_familiar, 'cestas básicas (R$ 750)'],
    ['Renda', c.salario_minimo, 'salários mínimos (R$ 1.518)'],
  ];
  $('impact-cards-grid').innerHTML = items.map(([cat, unit, desc]) => `
    <article class="impact-item">
      <span class="impact-item-cat">${cat}</span>
      <strong class="impact-item-qty">${intFmt.format(Math.floor(total / unit))}</strong>
      <span class="impact-item-desc">${desc}</span>
    </article>`).join('');
}

/* -------------------------------------------------------------------------
   Aba 3 — ficha técnica
   ------------------------------------------------------------------------- */
function setupTechnicalTab(data) {
  const m = data.metrics;
  $('tech-loss').textContent = `${m.lossInitial.toFixed(4)} → ${m.lossFinal.toFixed(4)}`;
  $('tech-acc').textContent = `${m.testAccuracy.toFixed(2)}%`;
  $('tech-prec-rec').textContent = `Precisão ${m.precision.toFixed(1)}% · Recall ${m.recall.toFixed(1)}%`;
  $('tech-confusion').textContent = `VP ${m.confusionMatrix.vp} · VN ${m.confusionMatrix.vn}`;
  $('tech-confusion-fp').textContent = `FP ${m.confusionMatrix.fp} · FN ${m.confusionMatrix.fn}`;

  const opts = (obj) => Object.entries(obj).map(([n, c]) => `<option value="${c}">${esc(n)}</option>`).join('');
  $('sim-categoria').innerHTML = opts(data.mappings.mapaCategoria);
  $('sim-uf').innerHTML = opts(data.mappings.mapaUF);
  $('sim-partido').innerHTML = opts(data.mappings.mapaPartido);

  const update = () => {
    const valor = parseFloat($('sim-valor').value) || 0;
    const cat = parseInt($('sim-categoria').value, 10) || 1;
    const mes = parseInt($('sim-mes').value, 10) || 1;
    const part = parseInt($('sim-partido').value, 10) || 1;
    const uf = parseInt($('sim-uf').value, 10) || 1;
    const res = predictMlpBrowser([valor, cat, mes, part, uf], data.modelParams);
    const stats = data.mappings.categorias[String(cat)] || { media_aceitavel: valor * 0.4 };

    const verdict = $('sim-verdict');
    const bar = $('sim-prob-bar');
    $('sim-prob').textContent = `${(res.probabilidadeAtipico * 100).toFixed(1)}%`;
    bar.style.inlineSize = `${Math.max(3, res.probabilidadeAtipico * 100)}%`;
    const alert = res.classePredita === 1;
    verdict.textContent = alert ? 'Classe 1 — acima do padrão' : 'Classe 0 — dentro da média';
    verdict.className = alert ? 'verdict-alert' : 'verdict-ok';
    bar.className = `prob-bar-fill ${alert ? 'alert' : 'ok'}`;
    $('sim-impact-preview').textContent = alert
      ? `Excesso estimado de ${brl.format(Math.max(valor - stats.media_aceitavel, valor * 0.3))} frente à média da categoria (${brl.format(stats.media_aceitavel)}).`
      : `Compatível com a média da categoria (${brl.format(stats.media_aceitavel)}).`;

    $('neurons-grid').innerHTML = res.a1.map((v, i) => {
      const a = 0.1 + Math.min(1, Math.abs(v)) * 0.35;
      const bg = v >= 0 ? `rgb(106 168 255 / ${a})` : `rgb(255 122 147 / ${a})`;
      return `<div class="neuron-cell" style="background:${bg}">h${i + 1}: ${v.toFixed(2)}</div>`;
    }).join('');
  };

  ['sim-valor', 'sim-categoria', 'sim-mes', 'sim-partido', 'sim-uf'].forEach((id) => $(id).addEventListener('input', update));
  update();
}

/* -------------------------------------------------------------------------
   Exportação CSV e compartilhamento
   ------------------------------------------------------------------------- */
function exportCsv() {
  const rows = sortedRows();
  if (!rows.length) {
    showToast('Não há notas para exportar');
    return;
  }
  const cols = [
    ['Parlamentar', (r) => r.nomeParlamentar],
    ['Cargo', (r) => r.cargo],
    ['Partido', (r) => r.siglaPartido],
    ['UF', (r) => r.siglaUF],
    ['Região', (r) => r.regiao],
    ['Ano', (r) => r.ano],
    ['Mês', (r) => r.mes],
    ['Despesa', (r) => r.tipoDespesa],
    ['Fornecedor', (r) => r.nomeFornecedor],
    ['CNPJ/CPF', (r) => r.cnpjCpfFornecedor],
    ['Valor (R$)', (r) => r.valorLiquido.toFixed(2).replace('.', ',')],
    ['Média da categoria (R$)', (r) => r.mediaAceitavelCategoria.toFixed(2).replace('.', ',')],
    ['Alerta IA', (r) => (r.classePredita === 1 ? 'Sim' : 'Não')],
    ['Probabilidade alerta', (r) => (r.probabilidadeAtipico * 100).toFixed(1).replace('.', ',')],
    ['Link da nota', (r) => r.urlDocumento],
  ];
  const cell = (v) => `"${String(v ?? '').replaceAll('"', '""')}"`;
  const csv = [cols.map(([h]) => cell(h)).join(';')]
    .concat(rows.map((r) => cols.map(([, fn]) => cell(fn(r))).join(';')))
    .join('\r\n');

  const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `olhocidadao_notas_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  showToast(`${intFmt.format(rows.length)} notas exportadas`);
}

async function copyLink() {
  try {
    await navigator.clipboard.writeText(location.href);
    showToast('Link da consulta copiado');
  } catch {
    window.prompt('Copie o link desta consulta:', location.href);
  }
}

/* -------------------------------------------------------------------------
   Orquestração
   ------------------------------------------------------------------------- */
function applyFilters({ updateUrl = true, keepPage = false } = {}) {
  const f = getFilterState();
  state.filtered = state.all.filter((r) => matchRecord(r, f));
  if (!keepPage) state.page = 1;

  renderActiveChips(f);
  renderProfile(f, state.filtered);
  renderKpisAndInsight(f, state.filtered);
  renderMonthly(state.filtered, f);
  renderCategories(state.filtered, f);
  renderRanking(state.filtered);
  renderTable();
  if (updateUrl) syncUrl(f);
}

function filterByPerson(name) {
  setFilterValue('parlamentar', name);
  applyFilters();
  window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
}

function bindEvents() {
  const debounced = debounce(() => applyFilters(), 220);
  $('f-busca').addEventListener('input', debounced);
  FILTERS.filter((x) => x.type !== 'text').forEach(({ id }) => {
    $(id).addEventListener('change', () => applyFilters());
  });

  $('btn-reset-filters').addEventListener('click', resetFilters);
  $('btn-export-csv').addEventListener('click', exportCsv);
  $('btn-copy-link').addEventListener('click', copyLink);

  // Delegação de cliques para elementos renderizados dinamicamente
  document.addEventListener('click', (e) => {
    const t = e.target.closest('button');
    if (!t) return;

    if (t.dataset.person) { filterByPerson(t.dataset.person); return; }
    if (t.dataset.clear) {
      const key = t.dataset.clear;
      setFilterValue(key, key === 'busca' ? '' : 'all');
      applyFilters();
      return;
    }
    if (t.dataset.action === 'reset') { resetFilters(); return; }
    if (t.dataset.detail) { openDetail(Number(t.dataset.detail)); return; }
    if (t.dataset.dlgPerson) { $('dlg-detail').close(); filterByPerson(t.dataset.dlgPerson); return; }
    if (t.dataset.dlgForn) {
      $('dlg-detail').close();
      $('f-busca').value = t.dataset.dlgForn;
      applyFilters();
      return;
    }
    if (t.dataset.month) {
      setFilterValue('mes', $('f-mes').value === t.dataset.month ? 'all' : t.dataset.month);
      applyFilters();
      return;
    }
    if (t.dataset.cat) {
      setFilterValue('categoria', $('f-categoria').value === t.dataset.cat ? 'all' : t.dataset.cat);
      applyFilters();
      return;
    }
    if (t.dataset.key !== undefined && t.classList.contains('rank-card')) {
      const val = t.dataset.key;
      if (state.rankMode === 'fornecedor') {
        $('f-busca').value = $('f-busca').value === val ? '' : val;
      } else {
        const map = { parlamentar: 'f-parlamentar', partido: 'f-partido', uf: 'f-uf', regiao: 'f-regiao' };
        const el = $(map[state.rankMode]);
        el.value = el.value === val ? 'all' : val;
      }
      applyFilters();
      return;
    }
    if (t.classList.contains('th-sort')) {
      const key = t.dataset.sort;
      state.sort = state.sort.key === key
        ? { key, dir: state.sort.dir === 'desc' ? 'asc' : 'desc' }
        : { key, dir: key === 'nomeParlamentar' ? 'asc' : 'desc' };
      state.page = 1;
      renderTable();
    }
  });

  document.querySelectorAll('.seg-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.seg-btn').forEach((b) => {
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-pressed', String(b === btn));
      });
      state.rankMode = btn.dataset.rank;
      renderRanking(state.filtered);
    });
  });

  $('btn-prev-page').addEventListener('click', () => { state.page -= 1; renderTable(); });
  $('btn-next-page').addEventListener('click', () => { state.page += 1; renderTable(); });

  // Atalho "/" para focar a busca
  document.addEventListener('keydown', (e) => {
    if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest('input, textarea, select, [contenteditable]') || document.querySelector('dialog[open]')) return;
    e.preventDefault();
    $('btn-tab-painel').click();
    $('f-busca').focus();
  });

  // Botão "voltar ao topo"
  const backTop = $('btn-back-top');
  window.addEventListener('scroll', () => { backTop.hidden = window.scrollY < 700; }, { passive: true });
  backTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' }));

  window.addEventListener('popstate', () => { restoreFromUrl(); applyFilters({ updateUrl: false }); });
}

async function init() {
  setupTabs();
  setupDialogs();
  setupSettings();
  state.pageSize = parseInt(safeStorageGet(STORAGE_PAGE) || '30', 10) || 30;

  try {
    const res = await fetch('/data/dashboard_data.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();
  } catch (err) {
    $('ai-insight-body').innerHTML = `<p>Não foi possível carregar os dados (${esc(err.message)}). Rode <code>pnpm run train</code> para gerar <code>public/data/dashboard_data.json</code>.</p>`;
    return;
  }

  state.all = (state.data.records || []).map((r, i) => ({
    ...r,
    _idx: i,
    diffPct: r.mediaAceitavelCategoria > 0 ? ((r.valorLiquido - r.mediaAceitavelCategoria) / r.mediaAceitavelCategoria) * 100 : 0,
    _search: `${r.nomeParlamentar} ${r.siglaPartido} ${r.siglaUF} ${r.nomeFornecedor} ${r.cnpjCpfFornecedor} ${r.tipoDespesa}`.toLowerCase(),
  }));
  state.mediaGlobal = state.all.length
    ? state.all.reduce((a, r) => a + r.valorLiquido, 0) / state.all.length
    : 0;

  populateFilterOptions(state.all);
  setupTechnicalTab(state.data);
  restoreFromUrl();
  bindEvents();
  applyFilters({ updateUrl: false });
}

init();
