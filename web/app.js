import { predictMlpBrowser } from './model.js';
import { filterWorks, cityOptions, workMunicipalities, comparisonValue } from './works.js';
import { decodeRecords, loadDashboardData, loadSummary, pickFilterColumns } from './dataset.js';
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
  { id: 'f-obra-uf', key: 'ufObra', param: 'ufObra', defaultValue: 'recorte' },
  { id: 'f-obra-cidade', key: 'cidadeObra', param: 'cidade' },
  { id: 'f-obra-tipo', key: 'tipoObra', param: 'tipoObra', defaultValue: 'oficial' },
  { id: 'f-obra-ano', key: 'anoObra', param: 'anoObra' },
  { id: 'f-obra-setor', key: 'setorObra', param: 'setorObra' },
  { id: 'f-obra-status', key: 'statusObra', param: 'statusObra' },
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
  worksLimit: 18,
  sort: { key: null, dir: 'desc' },
  mediaGlobal: 0,
  worker: null,
  otherYearRows: null,
  noMonthRows: null,
  maxMonthByYear: {},
  people: [],
  searchDirty: false, // texto digitado ainda não confirmado (Enter ou sugestão)
  committedBusca: '',
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
   Autocompletar de parlamentares (busca sem acento e por várias palavras)
   ------------------------------------------------------------------------- */
const norm = (value) => String(value ?? '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();

function buildPeopleIndex() {
  const map = new Map();
  for (const r of state.all) {
    let p = map.get(r.nomeParlamentar);
    if (!p) {
      p = { nome: r.nomeParlamentar, key: norm(r.nomeParlamentar), partido: r.siglaPartido, uf: r.siglaUF,
        cargo: r.cargo, foto: r.urlFoto, n: 0 };
      map.set(r.nomeParlamentar, p);
    }
    p.n += 1;
  }
  state.people = Array.from(map.values());
}

function suggestPeople(query, limit = 8) {
  const tokens = norm(query).split(/\s+/).filter(Boolean);
  if (!tokens.length || norm(query).trim().length < 2) return [];
  const rank = (p) => {
    if (p.key.startsWith(tokens[0])) return 0;
    return p.key.split(' ').some((w) => w.startsWith(tokens[0])) ? 1 : 2;
  };
  return state.people
    .filter((p) => tokens.every((t) => p.key.includes(t)))
    .sort((a, b) => rank(a) - rank(b) || b.n - a.n || a.nome.localeCompare(b.nome, 'pt-BR'))
    .slice(0, limit);
}

function highlightName(name, query) {
  const token = norm(query).split(/\s+/).filter(Boolean)[0];
  const normalized = norm(name);
  const at = token && normalized.length === name.length ? normalized.indexOf(token) : -1;
  if (at < 0) return esc(name);
  return `${esc(name.slice(0, at))}<mark>${esc(name.slice(at, at + token.length))}</mark>${esc(name.slice(at + token.length))}`;
}

const suggest = { items: [], active: -1 };

function closeSuggest() {
  $('search-suggest').hidden = true;
  $('f-busca').setAttribute('aria-expanded', 'false');
  $('f-busca').removeAttribute('aria-activedescendant');
  suggest.active = -1;
}

function setActiveSuggest(index) {
  const options = $('search-suggest').querySelectorAll('.suggest-item');
  if (!options.length) return;
  suggest.active = (index + options.length) % options.length;
  options.forEach((el, i) => el.setAttribute('aria-selected', String(i === suggest.active)));
  const el = options[suggest.active];
  $('f-busca').setAttribute('aria-activedescendant', el.id);
  el.scrollIntoView({ block: 'nearest' });
}

function renderSuggest() {
  const query = $('f-busca').value;
  suggest.items = suggestPeople(query);
  const list = $('search-suggest');
  if (!suggest.items.length) { closeSuggest(); return; }
  list.innerHTML = suggest.items.map((p, i) => {
    const cargo = p.cargo === 'SENADOR DA REPÚBLICA' ? 'Senador(a)' : 'Deputado(a) Federal';
    const foto = p.foto && p.foto.startsWith('http') ? p.foto : AVATAR_FALLBACK;
    return `<li class="suggest-item" role="option" id="suggest-${i}" data-index="${i}" aria-selected="false">
      <img src="${esc(foto)}" alt="" loading="lazy" decoding="async" width="34" height="34" />
      <span><span class="suggest-name">${highlightName(p.nome, query)}</span>
      <span class="suggest-meta">${cargo} · ${esc(p.partido)} · ${esc(p.uf)} · ${intFmt.format(p.n)} notas</span></span></li>`;
  }).join('') + '<li class="suggest-foot" role="presentation">Enter sem seleção busca o termo em notas e fornecedores</li>';
  list.querySelectorAll('img').forEach((img) => img.addEventListener('error', () => { img.src = AVATAR_FALLBACK; }, { once: true }));
  list.hidden = false;
  $('f-busca').setAttribute('aria-expanded', 'true');
  suggest.active = -1;
}

function chooseSuggestion(index) {
  const person = suggest.items[index];
  if (!person) return;
  $('f-busca').value = '';
  state.searchDirty = false;
  closeSuggest();
  setFilterValue('parlamentar', person.nome);
  applyFilters();
  window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
}

function setupSuggest() {
  const input = $('f-busca');
  const list = $('search-suggest');
  input.addEventListener('input', () => {
    // Digitar só atualiza as sugestões; o filtro permanece como estava.
    state.searchDirty = input.value !== state.committedBusca;
    if (!input.value.trim()) {
      closeSuggest();
      // Limpar o campo (ex.: botão X) remove a busca confirmada.
      if (state.committedBusca) { state.searchDirty = false; applyFilters(); }
      return;
    }
    renderSuggest();
  });
  input.addEventListener('focus', () => { if (input.value.trim()) renderSuggest(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (list.hidden || suggest.active < 0)) {
      // Enter sem escolher sugestão: confirma a busca livre por termo.
      e.preventDefault();
      state.searchDirty = false;
      closeSuggest();
      applyFilters();
      return;
    }
    if (list.hidden) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); setActiveSuggest(suggest.active + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActiveSuggest(suggest.active - 1); }
    else if (e.key === 'Enter' && suggest.active >= 0) { e.preventDefault(); chooseSuggestion(suggest.active); }
    else if (e.key === 'Escape') { e.stopPropagation(); closeSuggest(); }
  });
  // mousedown evita que o blur do input feche a lista antes do clique
  list.addEventListener('mousedown', (e) => {
    const item = e.target.closest('.suggest-item');
    if (!item) return;
    e.preventDefault();
    chooseSuggestion(Number(item.dataset.index));
  });
  document.addEventListener('click', (e) => { if (!e.target.closest('.search-field-wrap')) closeSuggest(); });
  input.addEventListener('blur', () => setTimeout(closeSuggest, 120));
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

  const fill = (id, set, label) => {
    $(id).innerHTML = `<option value="all">${label}</option>`
      + Array.from(set).sort((a, b) => a.localeCompare(b, 'pt-BR'))
        .map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
  };
  fill('f-uf', ufs, 'Todos');
  fill('f-partido', partidos, 'Todos');
  fill('f-categoria', cats, 'Todas');

  const obras = state.data?.obrasProjetos || [];
  const setores = new Set(obras.map((o) => o.setor).filter(Boolean));
  fill('f-obra-setor', setores, 'Todos os setores');
  fill('f-obra-status', new Set(obras.map((o) => o.situacao).filter(Boolean)), 'Todas as situações');
  const ufSelect = $('f-obra-uf');
  ufSelect.innerHTML = '<option value="recorte">Estado do recorte parlamentar</option><option value="all">Todos os estados</option>'
    + [...ufs].sort().map((uf) => `<option value="${uf}">${uf}</option>`).join('');
  fill('f-obra-ano', new Set(obras.filter((o) => o.tipoRegistro === 'oficial' && o.ano > 0).map((o) => String(o.ano))), 'Todos os anos de cadastro');
  updateWorkCities();

}

function getFilterState() {
  const f = {};
  FILTERS.forEach(({ id, key, type }) => {
    const el = $(id);
    if (type === 'check') f[key] = el.checked;
    else if (type === 'text') {
      // Enquanto o usuário digita, vale o último texto confirmado.
      if (!state.searchDirty) state.committedBusca = el.value;
      f[key] = norm(state.committedBusca).trim();
    }
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
  if (def.type === 'text') state.searchDirty = false;
}

function resetFilters() {
  FILTERS.forEach(({ id, type, defaultValue }) => {
    const el = $(id);
    if (type === 'check') el.checked = false;
    else if (type === 'text') el.value = '';
    else el.value = defaultValue || 'all';
  });
  state.searchDirty = false;
  applyFilters();
}

/** Texto de busca montado sob demanda (não é mantido em memória para cada nota). */
function searchText(r) {
  return norm(`${r.nomeParlamentar} ${r.siglaPartido} ${r.siglaUF} ${r.nomeFornecedor} ${r.cnpjCpfFornecedor} ${r.tipoDespesa}`);
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
  if (f.busca) {
    const text = searchText(r);
    if (!f.busca.split(/\s+/).every((t) => text.includes(t))) return false;
  }
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
  if (f.alerta === '0') parts.push('padrão esperado');
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
  if (f.alerta !== 'all') add('alerta', f.alerta === '1' ? 'Com alerta' : 'Padrão esperado');

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
    if (id === 'f-obra-cidade') updateWorkCities();
    if (!params.has(param)) return;
    const el = $(id);
    const v = params.get(param);
    if (type === 'check') el.checked = v === '1';
    else if (type === 'text') { el.value = v; state.searchDirty = false; }
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
      <img src="${esc(avatar)}" alt="Foto de ${esc(s.nomeParlamentar)}" class="profile-avatar" loading="lazy" decoding="async" />
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

function targetWorkUF(f, rows = state.filtered) {
  if (f.ufObra && f.ufObra !== 'recorte') return f.ufObra;
  if (f.uf !== 'all') return f.uf;
  if (f.parlamentar !== 'all') {
    const person = rows.find((row) => row.nomeParlamentar === f.parlamentar)
      || state.all.find((row) => row.nomeParlamentar === f.parlamentar);
    if (person) return person.siglaUF;
  }
  return 'all';
}

function updateWorkCities() {
  const f = getFilterState();
  const current = $('f-obra-cidade').value;
  const cities = cityOptions(state.data?.obrasProjetos || [], targetWorkUF(f));
  $('f-obra-cidade').innerHTML = `<option value="all">Todas as cidades (${cities.length})</option>`
    + cities.map((city) => `<option value="${esc(city.codigoIBGE)}">${esc(city.nome)} / ${esc(city.uf)}</option>`).join('');
  if (cities.some((city) => city.codigoIBGE === current)) $('f-obra-cidade').value = current;
}

function getMatchingObras(f, rows) {
  const list = filterWorks(state.data?.obrasProjetos || [], {
    uf: targetWorkUF(f, rows), city: f.cidadeObra, type: f.tipoObra,
    year: f.anoObra, sector: f.setorObra, status: f.statusObra,
  });
  return f.regiao !== 'all' && targetWorkUF(f, rows) === 'all'
    ? list.filter((work) => work.regiao === f.regiao) : list;
}

function custo(key) { return state.data.custosReferencia[key]; }

function renderKpisAndInsight(f, rows) {
  const total = rows.reduce((a, r) => a + r.valorLiquido, 0);
  const n = rows.length;
  const media = n ? total / n : 0;
  const pessoas = new Set(rows.map((r) => r.nomeParlamentar)).size;
  const alertas = rows.filter((r) => r.classePredita === 1);
  const excesso = alertas.reduce((a, r) => a + r.excessoMensal, 0);

  const obrasPeriodo = getMatchingObras(f, rows);
  const obrasCobertas = obrasPeriodo.filter((o) => comparisonValue(o) != null && excesso >= o.valorTotalObra).length;
  const paralisadas = obrasPeriodo.filter((o) => o.situacao.includes('Paralisada'));

  $('kpi-total-gasto').textContent = brl.format(total);
  $('kpi-total-notas').textContent = `${intFmt.format(n)} notas fiscais`;
  $('kpi-media-nota').textContent = brl.format(media);
  $('kpi-total-parlamentares').textContent = `${intFmt.format(pessoas)} parlamentar(es)`;
  $('kpi-total-atipicos').textContent = `${intFmt.format(alertas.length)} notas`;
  $('kpi-excesso-mensal').textContent = n
    ? `${((alertas.length / n) * 100).toFixed(1)}% das notas do recorte`
    : '—';
  $('kpi-excesso-periodo').textContent = brl.format(excesso);
  if (!excesso) {
    $('kpi-obras-cobertas').textContent = 'Sem excesso acima do padrão no período';
  } else if (obrasCobertas > 0) {
    $('kpi-obras-cobertas').textContent = `Cobre o investimento de ${intFmt.format(obrasCobertas)} projetos (comparados individualmente)`;
  } else if (obrasPeriodo.some((o) => comparisonValue(o) != null)) {
    const menorAporte = obrasPeriodo.reduce((minimum, o) => comparisonValue(o) != null ? Math.min(minimum, o.valorTotalObra) : minimum, Infinity);
    const pct = Math.min(99, Math.max(1, (excesso / menorAporte) * 100));
    $('kpi-obras-cobertas').textContent = `Equivale a ${pct.toFixed(0)}% do custo estimado de uma intervenção`;
  } else {
    $('kpi-obras-cobertas').textContent = 'Valor excedente apurado no recorte';
  }

  // Comparação com o outro ano disponível (mesmos filtros)
  const deltaEl = $('kpi-delta');
  if (f.ano !== 'all') {
    const outro = f.ano === '2026' ? '2025' : '2026';
    const totalOutro = (state.otherYearRows
      ?? state.all.filter((r) => matchRecord(r, { ...f, ano: outro })))
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
  const isMensal = f.mes !== 'all' || (excesso > 0 && excesso < 120000);
  $('ai-context-tag').textContent = scope;
  $('impact-scope-label').textContent = scope;
  $('impact-scale-label').textContent = isMensal
    ? 'comparação em escala mensal / custeio direto'
    : 'comparação em escala anual / obras e equipamentos';
  $('impact-total-periodo').textContent = brl.format(excesso);
  renderImpactCards(excesso, isMensal);
  renderObrasGrid(excesso, f, rows);

  const body = $('ai-insight-body');
  if (!n) {
    body.innerHTML = '<p>Nenhuma nota corresponde aos filtros atuais. Remova algum filtro para ver o resumo.</p>';
    return;
  }

  const [topCat, topCatVal] = topEntry(rows, (r) => r.tipoDespesa);
  const [topForn, topFornVal] = topEntry(rows, (r) => r.nomeFornecedor);
  const [topUF] = topEntry(rows, (r) => r.siglaUF);
  const [topPart, topPartVal] = topEntry(rows.filter((r) => r.siglaPartido !== 'N/I'), (r) => r.siglaPartido);

  const diff = state.mediaGlobal ? ((media - state.mediaGlobal) / state.mediaGlobal) * 100 : 0;
  const comparativo = Math.abs(diff) < 5
    ? `em linha com a média geral (${brl.format(state.mediaGlobal)})`
    : `<strong>${Math.abs(diff).toFixed(1)}% ${diff > 0 ? 'acima' : 'abaixo'}</strong> da média geral (${brl.format(state.mediaGlobal)})`;
  const pctCat = total ? ((topCatVal / total) * 100).toFixed(0) : 0;

  let contexto;
  if (f.parlamentar !== 'all') {
    contexto = `<p><strong>${esc(f.parlamentar)}</strong> registrou <strong>${intFmt.format(n)} notas</strong> somando
      <strong>${brl.format(total)}</strong> no período selecionado. O valor médio por nota está ${comparativo}.
      A maior parte foi para <strong>${esc(topCat.toLowerCase())}</strong> (${pctCat}% do total), e o fornecedor que mais recebeu foi
      <strong>${esc(topForn)}</strong> (${brl.format(topFornVal)}).</p>`;
  } else if (f.partido !== 'all' || f.uf !== 'all' || f.regiao !== 'all' || f.cargo !== 'all') {
    contexto = `<p>Neste recorte (<strong>${esc(scope)}</strong>) há <strong>${intFmt.format(pessoas)} parlamentares</strong> e
      <strong>${intFmt.format(n)} notas</strong>, totalizando <strong>${brl.format(total)}</strong>. O valor médio por nota está ${comparativo}.
      A principal despesa é <strong>${esc(topCat.toLowerCase())}</strong> (${pctCat}%), com maior concentração em
      <strong>${esc(topUF)}</strong> e, entre as legendas informadas, no partido <strong>${esc(topPart || "não informado")}</strong>.</p>`;
  } else {
    contexto = `<p>Somando Câmara e Senado neste período, a consulta reúne <strong>${intFmt.format(pessoas)} parlamentares</strong> e
      <strong>${brl.format(total)}</strong> em <strong>${intFmt.format(n)} notas</strong>. A despesa de maior peso é
      <strong>${esc(topCat.toLowerCase())}</strong> (${pctCat}%). Entre as legendas informadas, o maior volume é do partido <strong>${esc(topPart)}</strong>
      (${brl.format(topPartVal)}) e o estado com maior concentração é <strong>${esc(topUF)}</strong>.</p>`;
  }

  let ia;
  if (alertas.length) {
    // Descobre qual UF concentrou o maior volume de excesso nas notas com alerta do filtro atual
    const [ufMaiorExcesso] = topEntry(alertas, (r) => r.siglaUF);
    const ufAlvo = f.uf !== 'all' ? f.uf : (f.parlamentar !== 'all' && rows.length ? rows[0].siglaUF : ufMaiorExcesso);

    // Filtra obras da UF principal do recorte; se "Todos os anos" estiver ativo, prioriza as mais atuais (2026)
    let candidatas = obrasPeriodo.filter((o) => o.uf === ufAlvo);
    if (!candidatas.length) candidatas = obrasPeriodo;
    if (f.ano === 'all') {
      const atuais2026 = candidatas.filter((o) => o.ano === 2026);
      if (atuais2026.length) candidatas = atuais2026;
    }

    // Escolhe de forma determinística a obra cujo aporte/custo melhor se compara ao excesso do período
    // priorizando obras reais do ObrasGov.br ("OG-") e obras paralisadas/em andamento
    const ordenadas = [...candidatas].sort((a, b) => {
      const aOg = String(a.codigoObra || '').startsWith('OG-') ? 1 : 0;
      const bOg = String(b.codigoObra || '').startsWith('OG-') ? 1 : 0;
      if (aOg !== bOg) return bOg - aOg;
      const distA = Math.abs(a.aporteNecessarioPeriodo - excesso);
      const distB = Math.abs(b.aporteNecessarioPeriodo - excesso);
      return distA - distB;
    });

    const obraDestaque = ordenadas.find((o) => comparisonValue(o) != null);
    const trechoObra = obraDestaque
      ? ` No cadastro ${obraDestaque.tipoRegistro === 'oficial' ? 'oficial do ObrasGov' : 'de referências'},
          <em>${esc(obraDestaque.titulo)}</em>, em <strong>${esc(obraDestaque.municipio)}</strong>,
          tem ${obraDestaque.tipoRegistro === 'oficial' ? 'investimento previsto publicado' : 'custo estimado'} de
          <strong>${brl.format(obraDestaque.valorTotalObra)}</strong>.
          O excedente do recorte parlamentar equivale a <strong>${(excesso / obraDestaque.valorTotalObra).toFixed(1)}×</strong>
          esse valor. Esta comparação não vincula a nota parlamentar ao financiamento da obra.`
      : '';
    const multiSetor = isMensal
      ? `Nesse recorte mensal, o valor custearia <strong>${intFmt.format(Math.max(0, Math.floor(excesso / custo('consulta_atencao_basica'))))} atendimentos no SUS</strong> (Saúde),
         <strong>${intFmt.format(Math.max(0, Math.floor(excesso / custo('aluno_mes_fundeb'))))} aluno-mês de referência FUNDEB</strong> (Educação) ou
         <strong>${intFmt.format(Math.max(0, Math.floor(excesso / custo('equipamento_seguranca_uf'))))} diárias de patrulhamento e proteção</strong> (Segurança).`
      : `No acumulado do período, esse montante financiaria <strong>${intFmt.format(Math.max(0, Math.floor(excesso / custo('reforma_ubs'))))} reformas de UBS de referência</strong> (Saúde),
         <strong>${intFmt.format(Math.max(0, Math.floor(excesso / custo('climatizacao_escola'))))} climatizações escolares de referência</strong> (Educação) ou
         <strong>${intFmt.format(Math.max(0, Math.floor(excesso / custo('pavimentacao_vias'))))} trechos de pavimentação de referência</strong> (Infraestrutura).`;
    ia = `<p>A Rede Neural sinalizou <strong>${intFmt.format(alertas.length)} notas</strong> potencialmente atípicas, somando um
      excedente estatístico de <strong>${brl.format(excesso)}</strong> no período. ${multiSetor}${trechoObra}</p>`;
  } else {
    ia = '<p>Nenhuma nota deste recorte ficou acima do padrão estatístico da sua categoria.</p>';
  }

  body.innerHTML = contexto + ia + '<p class="footnote">Base integral de despesas positivas publicadas em 2025–2026; estornos negativos não entram na soma. Alertas são estatísticos e não comprovam irregularidade. Equivalências sociais usam custos de referência.</p>';
}

/* -------------------------------------------------------------------------
   Gráficos
   ------------------------------------------------------------------------- */
function maxMonthForYear(ano) {
  if (ano === 'all') return 12;
  const maxM = state.maxMonthByYear[String(ano)] || 0;
  return maxM > 0 ? maxM : (String(ano) === '2026' ? 10 : 12);
}

function syncMonthSelectForYear(ano) {
  const maxM = maxMonthForYear(ano);
  const mesSelect = $('f-mes');
  if (!mesSelect) return;
  Array.from(mesSelect.options).forEach((opt) => {
    if (opt.value === 'all') return;
    const m = Number(opt.value);
    opt.hidden = m > maxM;
    opt.disabled = m > maxM;
  });
  if (mesSelect.value !== 'all' && Number(mesSelect.value) > maxM) {
    mesSelect.value = 'all';
  }
}

function renderMonthly(rows, f) {
  const maxMonth = maxMonthForYear(f.ano);
  const baseRows = f.mes === 'all'
    ? rows
    : (state.noMonthRows ?? state.all.filter((r) => matchRecord(r, { ...f, mes: 'all' })));

  const normal = new Array(maxMonth).fill(0);
  const alert = new Array(maxMonth).fill(0);
  baseRows.forEach((r) => {
    const i = r.mes - 1;
    if (i < 0 || i >= maxMonth) return;
    if (r.classePredita === 1) alert[i] += r.valorLiquido;
    else normal[i] += r.valorLiquido;
  });
  const totals = normal.map((v, i) => v + alert[i]);
  const max = Math.max(...totals, 1);

  const container = $('chart-monthly');
  container.style.setProperty('--month-cols', String(maxMonth));

  const mesesVisiveis = MESES_CURTOS.slice(0, maxMonth);
  container.innerHTML = mesesVisiveis.map((label, i) => {
    const selected = f.mes === String(i + 1);
    const dimmed = f.mes !== 'all' && !selected;
    const valShort = totals[i]
      ? brlCompact.format(totals[i]).replace(/^R\$\s?/, '')
      : '—';
    return `
      <button type="button" class="month-col${selected ? ' selected' : ''}${dimmed ? ' dimmed' : ''}" data-month="${i + 1}"
        aria-pressed="${selected}" aria-label="${MESES_NOMES[i]}: ${brl.format(totals[i])}"
        title="${MESES_NOMES[i]}: ${brl.format(totals[i])} (Normal: ${brl.format(normal[i])} · Alerta: ${brl.format(alert[i])})">
        <span class="month-val-top">${valShort}</span>
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
      : '<span class="badge-normal">Padrão esperado</span>';
    const diffCls = r.classePredita === 1 ? 'diff-alert' : (r.diffPct > 0 ? 'diff-up' : 'diff-down');
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

  const obraUF = (state.data?.obrasProjetos || []).find((o) => o.uf === r.siglaUF && o.ano === r.ano && o.valorTotalObra > 0)
    || (state.data?.obrasProjetos || []).find((o) => o.uf === r.siglaUF && o.valorTotalObra > 0);
  const refObraTxt = obraUF
    ? ` Em <strong>${esc(obraUF.municipio)}/${esc(obraUF.uf)}</strong> (${r.ano}), esse valor representa
       <strong>${Math.min(100, (r.excessoMensal / obraUF.aporteNecessarioPeriodo) * 100).toFixed(1)}%</strong> do custo de referência da obra/serviço
       <em>"${esc(obraUF.titulo)}"</em>.`
    : '';

  const explain = isAlert
    ? `<strong>Por que recebeu alerta?</strong> Esta nota de ${brl.format(r.valorLiquido)} é cerca de
       <strong>${vezes.toFixed(1)}×</strong> o valor típico de "${esc(r.tipoDespesa.toLowerCase())}" (${brl.format(r.mediaAceitavelCategoria)}).
       O excedente estatístico desta nota é de <strong>${brl.format(r.excessoMensal)}</strong> — no mesmo mês (${MESES_NOMES[r.mes - 1]}/${r.ano}),
       esse valor custearia <strong>${intFmt.format(Math.max(0, Math.floor(r.excessoMensal / custo('consulta_atencao_basica'))))} consultas médicas no SUS</strong>,
       <strong>${intFmt.format(Math.max(0, Math.floor(r.excessoMensal / custo('kit_escolar_aluno'))))} kits escolares anuais</strong> ou
       <strong>${intFmt.format(Math.max(0, Math.floor(r.excessoMensal / custo('cesta_basica_familiar'))))} cestas de assistência de referência</strong> em ${esc(r.siglaUF)}.${refObraTxt}`
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
      <span>Limiar de alerta da IA (Média + 1,2σ): <strong>${brl.format(r.limiarCategoria)}</strong></span>
      <div class="compare-track"><div class="compare-fill threshold" style="inline-size:${(r.limiarCategoria / ref) * 100}%"></div></div>
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
   Aba 2 — Comparativo multi-setor e Obras/Serviços no período
   ------------------------------------------------------------------------- */
function renderImpactCards(totalExcesso, isMensal) {
  const items = isMensal
    ? [
      ['Saúde', custo('consulta_atencao_basica'), 'consultas especializadas', 'Referência de credenciamento; não tarifa universal SUS'],
      ['Educação', custo('aluno_mes_fundeb'), 'aluno-mês', 'Aproximação FUNDEB; varia por rede e modalidade'],
      ['Segurança', custo('equipamento_seguranca_uf'), 'diárias operacionais', 'Estimativa comparativa'],
      ['Assistência social', custo('cesta_basica_familiar'), 'cestas de assistência', 'Estimativa; não cesta DIEESE'],
    ]
    : [
      ['Saúde', custo('reforma_ubs'), 'reformas de UBS', 'Cenário de R$ 150 mil a R$ 350 mil'],
      ['Educação', custo('climatizacao_escola'), 'climatizações de escolas', 'Cenário de R$ 80 mil a R$ 180 mil'],
      ['Urgência e emergência', custo('ambulancia_samu'), 'ambulâncias SAMU', 'Referência aproximada; especificações variam'],
      ['Infraestrutura', custo('pavimentacao_vias'), 'trechos de pavimentação', 'Referência comparativa; extensão e escopo variam'],
    ];

  $('impact-cards-grid').innerHTML = items.map(([cat, unit, desc, sub]) => {
    const qty = totalExcesso / unit;
    const qtyFmt = qty >= 1 ? intFmt.format(Math.floor(qty)) : `${(qty * 100).toFixed(0)}% de 1`;
    return `
      <article class="impact-item">
        <span class="impact-item-cat">${cat}</span>
        <strong class="impact-item-qty">${qtyFmt}</strong>
        <span class="impact-item-desc">${desc} (${brl.format(unit)})</span>
        <span class="impact-item-sub">${sub}</span>
      </article>`;
  }).join('');
}

function renderObrasGrid(totalExcesso, f, rows) {
  const grid = $('obras-grid');
  if (!grid) return;
  const obras = getMatchingObras(f, rows);
  const cityLabel = f.cidadeObra !== 'all' ? $('f-obra-cidade').selectedOptions[0]?.textContent : null;
  $('obras-subtitle').textContent = `${intFmt.format(obras.length)} projetos ${cityLabel ? 'em ' + cityLabel : 'no recorte geográfico'} · situação atual do cadastro. Ano e mês dos gastos não restringem o cadastro de obras.`;
  $('obras-scope-note').textContent = `Comparação com ${brl.format(totalExcesso)} de excedente estatístico das despesas parlamentares no recorte selecionado. O filtro de cidade seleciona obras e não atribui esses gastos ao município.`;
  if (!obras.length) {
    grid.innerHTML = '<p class="card-subtitle" role="status">Nenhum projeto cadastrado corresponde aos filtros. Experimente outro município ou amplie o recorte.</p>';
    $('btn-more-obras').hidden = true;
    return;
  }
  const sorted = [...obras].sort((a, b) => {
    const status = Number(b.situacao === 'Paralisada') - Number(a.situacao === 'Paralisada');
    return status || Number(comparisonValue(a) == null) - Number(comparisonValue(b) == null)
      || String(b.dataCadastro || '').localeCompare(String(a.dataCadastro || ''))
      || (a.valorTotalObra || 0) - (b.valorTotalObra || 0) || a.codigoObra.localeCompare(b.codigoObra);
  });
  grid.innerHTML = sorted.slice(0, state.worksLimit).map((o) => {
    const official = o.tipoRegistro === 'oficial';
    const value = o.valorTotalObra;
    const ratio = comparisonValue(o) != null ? totalExcesso / value : null;
    const pctBar = ratio == null ? 0 : Math.min(100, Math.max(0, ratio * 100));
    const locations = workMunicipalities(o);
    const place = locations.length ? locations.map((city) => `${city.nome} / ${city.uf}`).join('; ') : o.municipio;
    const comparison = ratio == null
      ? (value > 0 ? 'O valor publicado é inferior a R$ 1.000 e não é usado na comparação orçamentária.' : 'A fonte não publicou valor positivo de investimento para esta obra.')
      : `O excedente do recorte equivale a <strong>${(ratio * 100).toFixed(1)}%</strong> do ${official ? 'investimento previsto' : 'custo de referência'} (${ratio.toFixed(1)}×).`;
    return `<article class="obra-card${o.situacao === 'Paralisada' ? ' paralisada' : ''}${ratio >= 1 ? ' coberta' : ''}">
      <div class="obra-top"><span class="obra-badge-setor">${esc(o.setor)}${o.subTipo ? ' · ' + esc(o.subTipo) : ''}</span>
        <span class="obra-badge-status">${esc(o.situacao)}</span></div>
      <h4 class="obra-title">${esc(o.titulo)}</h4>
      <div class="obra-meta">
        <span>📍 Município: <strong>${esc(place)}</strong></span>
        ${o.divergenciaLocalizacao ? `<span>${esc(o.origemMunicipio)}; divergência na geometria da fonte (${esc((o.municipiosGeometria || []).map((city) => city.nome + "/" + city.uf).join("; "))}).</span>` : ""}
        ${o.endereco ? `<span>Endereço cadastrado: ${esc(o.endereco)}</span>` : ''}
        <span>Cadastro: ${o.dataCadastro ? esc(o.dataCadastro.split('-').reverse().join('/')) : official ? 'Não informado' : 'referência ' + o.ano}</span>
        <span>🏛 ${esc(o.orgaoFonte)}</span>
      </div>
      <div class="obra-values"><span>${official ? 'Investimento previsto' : 'Custo estimado'}: <strong class="mono">${value > 0 ? brl.format(value) : 'Não informado'}</strong></span></div>
      ${ratio != null ? `<div class="obra-bar-track" aria-hidden="true"><div class="obra-bar-fill${ratio < 1 ? ' partial' : ''}" style="inline-size:${pctBar}%"></div></div>` : ''}
      <p class="obra-impact-note">${comparison}</p>
      ${official && o.urlFonte?.startsWith('https://cipi.economia.gov.br/') ? `<a href="${esc(o.urlFonte)}" target="_blank" rel="noopener noreferrer">Ver cadastro oficial ↗</a>` : ''}
    </article>`;
  }).join('');
  $('btn-more-obras').hidden = state.worksLimit >= obras.length;
  $('btn-more-obras').textContent = `Mostrar mais projetos (${intFmt.format(Math.min(state.worksLimit, obras.length))} de ${intFmt.format(obras.length)})`;
}

/* -------------------------------------------------------------------------
   Aba 3 — ficha técnica
   ------------------------------------------------------------------------- */
function setupTechnicalTab(data) {
  const m = data.metrics;
  const evaluation = m.evaluation;
  const params = data.modelParams;
  $('tech-architecture').textContent = `${params.n_input} → ${params.n_hidden} → 2`;
  $('tech-training').textContent = evaluation
    ? `${intFmt.format(m.trainRecords)} notas · Adam · época ${evaluation.bestEpoch}` : 'Treino em CPU';
  if (evaluation) {
    const test = evaluation.test;
    const baseline = evaluation.baselineSameTemporalTest;
    $('tech-protocol').textContent = `${intFmt.format(test.records)} notas no teste de jul–out/2026, reservado até a escolha final. Treino: 2025; validação: jan–mar; calibração: abr–mai; limiar: jun/2026.`;
    $('tech-comparison').textContent = `Comparação no mesmo teste: precisão ${(baseline.precision * 100).toFixed(2)}% → ${(test.precision * 100).toFixed(2)}%; F1 ${(baseline.f1 * 100).toFixed(2)}% → ${(test.f1 * 100).toFixed(2)}%. Falsos positivos: ${intFmt.format(baseline.confusionMatrix.fp)} → ${intFmt.format(test.confusionMatrix.fp)}.`;
    $('tech-calibration').textContent = `Brier: ${test.brier.toFixed(4)} · Erro de calibração (ECE): ${(test.ece * 100).toFixed(2)} p.p. · Limiar automático: ${(evaluation.decisionThreshold * 100).toFixed(2)}%. Score referente ao alvo estatístico, não à probabilidade de corrupção.`;
  }
  $('tech-loss').textContent = `${m.lossInitial.toFixed(4)} → ${m.lossFinal.toFixed(4)}`;
  $('tech-acc').textContent = `${m.testAccuracy.toFixed(2)}%`;
  $('tech-prec-rec').textContent = `Precisão ${m.precision.toFixed(1)}% · Recall ${m.recall.toFixed(1)}%`;
  $('tech-confusion').textContent = `VP ${m.confusionMatrix.vp} · VN ${m.confusionMatrix.vn}`;
  $('tech-confusion-fp').textContent = `FP ${m.confusionMatrix.fp} · FN ${m.confusionMatrix.fn}`;

  const opts = (obj) => Object.entries(obj).map(([n, c]) => `<option value="${c}">${esc(n || 'Categoria não informada na fonte')}</option>`).join('');
  $('sim-categoria').innerHTML = opts(data.mappings.mapaCategoria);
  const defaultCategory = Object.entries(data.mappings.mapaCategoria).find(([name]) => name.trim());
  if (defaultCategory) $('sim-categoria').value = String(defaultCategory[1]);
  $('sim-form').addEventListener('submit', (event) => event.preventDefault());

  const update = () => {
    const valor = parseFloat($('sim-valor').value) || 0;
    const cat = parseInt($('sim-categoria').value, 10) || 1;
    const res = predictMlpBrowser({ valorLiquido: valor, codTipoDespesa: cat }, data.modelParams);
    const stats = data.mappings.categorias[String(cat)] || { media_aceitavel: valor * 0.4 };

    const verdict = $('sim-verdict');
    const bar = $('sim-prob-bar');
    $('sim-prob').textContent = `${(res.probabilidadeAtipico * 100).toFixed(1)}%`;
    bar.style.inlineSize = `${Math.max(3, res.probabilidadeAtipico * 100)}%`;
    const alert = res.classePredita === 1;
    verdict.textContent = alert ? 'Classe 1 — acima do padrão' : 'Classe 0 — sem alerta estatístico';
    verdict.className = alert ? 'verdict-alert' : 'verdict-ok';
    bar.className = `prob-bar-fill ${alert ? 'alert' : 'ok'}`;
    $('sim-impact-preview').textContent = alert
      ? `Excesso estimado de ${brl.format(Math.max(valor - stats.media_aceitavel, 0))} frente à média da categoria (${brl.format(stats.media_aceitavel)}).`
      : `Sem alerta pelo limiar calibrado. Média histórica da categoria (${brl.format(stats.media_aceitavel)}).`;

    $('neurons-grid').innerHTML = res.a1.map((v, i) => {
      const a = 0.1 + Math.min(1, Math.abs(v)) * 0.35;
      const bg = v >= 0 ? `rgb(106 168 255 / ${a})` : `rgb(255 122 147 / ${a})`;
      return `<div class="neuron-cell" style="background:${bg}">h${i + 1}: ${v.toFixed(2)}</div>`;
    }).join('');
  };

  ['sim-valor', 'sim-categoria'].forEach((id) => $(id).addEventListener('input', update));
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
    ['Score calibrado de atipicidade', (r) => (r.probabilidadeAtipico * 100).toFixed(1).replace('.', ',')],
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
/* Filtragem em Web Worker: a thread da interface não congela com 300 mil notas. */
const pendingFilters = new Map();
let workerReady = null;
let requestId = 0;
let applySeq = 0;

function setupFilterWorker(columns) {
  try {
    const worker = new Worker(new URL('./filter.worker.js', import.meta.url), { type: 'module' });
    workerReady = new Promise((resolve, reject) => {
      worker.onmessage = (event) => {
        const msg = event.data;
        if (msg.type === 'ready') resolve();
        else if (msg.type === 'result') {
          pendingFilters.get(msg.id)?.(msg);
          pendingFilters.delete(msg.id);
        }
      };
      worker.onerror = (event) => {
        reject(new Error(event.message || 'Falha no worker'));
        pendingFilters.forEach((done) => done(null));
        pendingFilters.clear();
      };
    });
    workerReady.catch(() => {});
    worker.postMessage({ type: 'init', cols: columns });
    state.worker = worker;
  } catch (error) {
    console.info('Web Worker indisponível, filtrando na thread principal:', error.message);
    state.worker = null;
  }
}

async function filterRows(f) {
  const otherYear = f.ano !== 'all' ? (f.ano === '2026' ? '2025' : '2026') : null;
  if (state.worker) {
    try {
      await workerReady;
      const id = ++requestId;
      const res = await new Promise((resolve) => {
        pendingFilters.set(id, resolve);
        state.worker.postMessage({ type: 'filter', id, f, otherYear });
      });
      if (!res) throw new Error('worker sem resposta');
      const pick = (indexes) => (indexes ? Array.from(indexes, (i) => state.all[i]) : null);
      return { filtered: pick(res.filtered), otherYear: pick(res.otherYear), noMonth: pick(res.noMonth) };
    } catch (error) {
      console.info('Filtro no worker falhou; usando a thread principal:', error.message);
      state.worker = null;
    }
  }
  return { filtered: state.all.filter((r) => matchRecord(r, f)), otherYear: null, noMonth: null };
}

function setBusy(busy) {
  document.querySelector('.kpi-row')?.classList.toggle('is-loading', busy);
  document.querySelector('.ai-insight-card')?.classList.toggle('is-loading', busy);
}

async function applyFilters({ updateUrl = true, keepPage = false } = {}) {
  syncMonthSelectForYear($('f-ano')?.value || 'all');
  updateWorkCities();
  const f = getFilterState();
  const seq = ++applySeq;
  const slowTimer = setTimeout(() => setBusy(true), 150);
  const rows = await filterRows(f);
  clearTimeout(slowTimer);
  if (seq !== applySeq) return; // chegou uma consulta mais recente
  setBusy(false);

  state.worksLimit = 18;
  state.filtered = rows.filtered;
  state.otherYearRows = rows.otherYear;
  state.noMonthRows = rows.noMonth;
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
  FILTERS.filter((x) => x.type !== 'text').forEach(({ id }) => {
    $(id).addEventListener('change', () => applyFilters());
  });

  $('btn-reset-filters').addEventListener('click', resetFilters);
  $('btn-export-csv').addEventListener('click', exportCsv);
  $('btn-copy-link').addEventListener('click', copyLink);
  $('btn-more-obras').addEventListener('click', () => {
    state.worksLimit += 18;
    const excess = state.filtered.filter((r) => r.classePredita === 1).reduce((sum, r) => sum + r.excessoMensal, 0);
    renderObrasGrid(excess, getFilterState(), state.filtered);
  });

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
      state.searchDirty = false;
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
        $('f-busca').value = state.committedBusca === val ? '' : val;
        state.searchDirty = false;
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

/** Mostra os totais gerais (resumo_geral.json) enquanto as notas detalhadas carregam. */
function showQuickSummary(summary) {
  const t = summary?.totais;
  if (!t || state.all.length || location.search) return;
  $('kpi-total-gasto').textContent = brl.format(t.total);
  $('kpi-total-notas').textContent = `${intFmt.format(t.notas)} notas fiscais`;
  $('kpi-media-nota').textContent = brl.format(t.notas ? t.total / t.notas : 0);
  $('kpi-total-parlamentares').textContent = `${intFmt.format(t.parlamentares)} parlamentar(es)`;
  $('kpi-total-atipicos').textContent = `${intFmt.format(t.alertas)} notas`;
  $('kpi-excesso-mensal').textContent = `${((t.alertas / Math.max(t.notas, 1)) * 100).toFixed(1)}% das notas`;
  $('kpi-excesso-periodo').textContent = brl.format(t.excesso);
  $('kpi-obras-cobertas').textContent = 'Visão geral · detalhando o recorte…';
  document.querySelector('.kpi-row')?.classList.remove('is-loading');
  $('ai-insight-body').innerHTML = `<p>Carregando as ${intFmt.format(t.notas)} notas fiscais para habilitar filtros e comparações…</p>`;
}

async function init() {
  setupTabs();
  setupDialogs();
  setupSettings();
  state.pageSize = parseInt(safeStorageGet(STORAGE_PAGE) || '30', 10) || 30;

  loadSummary().then(showQuickSummary);
  try {
    state.data = await loadDashboardData();
  } catch (err) {
    setBusy(false);
    $('ai-insight-body').innerHTML = `<p>Não foi possível carregar os dados (${esc(err.message)}). Execute <code>pnpm setup</code> para verificar e restaurar os arquivos publicados.</p>`;
    return;
  }

  state.all = decodeRecords(state.data);
  setupFilterWorker(pickFilterColumns(state.data));
  delete state.data.recordColumns;
  delete state.data.records;
  state.all.forEach((r, i) => {
    r._idx = i;
    r.diffPct = r.mediaAceitavelCategoria > 0 ? ((r.valorLiquido - r.mediaAceitavelCategoria) / r.mediaAceitavelCategoria) * 100 : 0;
    const year = String(r.ano);
    if (!(state.maxMonthByYear[year] >= r.mes)) state.maxMonthByYear[year] = r.mes;
  });
  state.mediaGlobal = state.all.length
    ? state.all.reduce((a, r) => a + r.valorLiquido, 0) / state.all.length
    : 0;

  populateFilterOptions(state.all);
  buildPeopleIndex();
  setupSuggest();
  setupTechnicalTab(state.data);
  restoreFromUrl();
  bindEvents();
  applyFilters({ updateUrl: false });
}

init();
