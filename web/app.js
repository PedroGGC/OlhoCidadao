const brlFormatter = new Intl.NumberFormat('pt-BR', {
  style: 'currency',
  currency: 'BRL',
});

const intFormatter = new Intl.NumberFormat('pt-BR', {
  maximumFractionDigits: 0,
});

let dashboardData = null;

/**
 * Executa o Forward Pass exato da Rede Neural NnModel no navegador
 * utilizando os parâmetros treinados (mean, std, W1, b1, W2, b2) e ativação np.tanh
 */
function predictMlpBrowser(features, modelParams) {
  const { mean, std, W1, b1, W2, b2 } = modelParams;

  // 1. Normalização Z-Score: (X - mean) / (std + 1e-8)
  const xNorm = features.map((val, i) => (val - mean[i]) / (std[i] + 1e-8));

  // 2. Camada oculta com Tangente Hiperbólica: a1 = tanh(xNorm · W1 + b1)
  const nHidden = b1.length;
  const a1 = new Array(nHidden).fill(0);
  for (let j = 0; j < nHidden; j++) {
    let z = b1[j];
    for (let i = 0; i < xNorm.length; i++) {
      z += xNorm[i] * W1[i][j];
    }
    a1[j] = Math.tanh(z);
  }

  // 3. Camada de saída + Softmax numericamente estável
  const nOutput = b2.length;
  const z2 = new Array(nOutput).fill(0);
  for (let k = 0; k < nOutput; k++) {
    let sum = b2[k];
    for (let j = 0; j < nHidden; j++) {
      sum += a1[j] * W2[j][k];
    }
    z2[k] = sum;
  }

  const maxZ2 = Math.max(...z2);
  const expScores = z2.map((v) => Math.exp(v - maxZ2));
  const sumExp = expScores.reduce((acc, v) => acc + v, 0);
  const probs = expScores.map((v) => v / sumExp);

  const classePredita = probs[1] >= 0.5 ? 1 : 0;
  return {
    classePredita,
    probabilidadeAtipico: probs[1],
    a1,
  };
}

function renderKPIs(metrics) {
  document.getElementById('kpi-acc').textContent = `${metrics.testAccuracy.toFixed(2)}%`;
  document.getElementById('kpi-recall').textContent =
    `Recall Atípicos: ${metrics.recall.toFixed(2)}% · Precisão: ${metrics.precision.toFixed(2)}%`;
  document.getElementById('kpi-loss').textContent =
    `${metrics.lossInitial.toFixed(4)} → ${metrics.lossFinal.toFixed(4)}`;
  document.getElementById('kpi-impacto').textContent = brlFormatter.format(metrics.impact5YearsTotal);
  document.getElementById('kpi-suspeitos').textContent =
    `${metrics.suspiciousCount} notas suspeitas em ${metrics.testRecords} avaliadas`;

  const merendas = Math.floor(metrics.impact5YearsTotal / 1.5);
  document.getElementById('kpi-merendas').textContent = `${intFormatter.format(merendas)} refeições`;

  const cm = metrics.confusionMatrix;
  document.getElementById('confusion-badge').textContent =
    `VP: ${cm.vp} | VN: ${cm.vn} | FP: ${cm.fp} | FN: ${cm.fn}`;
}

function renderImpactCards(total5Years, custos) {
  const items = [
    {
      cat: 'Educação Básica (PNAE/FNDE)',
      qty: intFormatter.format(Math.floor(total5Years / custos.merenda_escolar_pnae)),
      desc: 'Merendas escolares completas (a R$ 1,50/refeição)',
    },
    {
      cat: 'Material Escolar (MEC)',
      qty: intFormatter.format(Math.floor(total5Years / custos.kit_escolar_aluno)),
      desc: 'Kits escolares anuais completos (a R$ 120,00/aluno)',
    },
    {
      cat: 'Inclusão Digital Escolar',
      qty: intFormatter.format(Math.floor(total5Years / custos.computador_escola_publica)),
      desc: 'Computadores para laboratórios públicos (a R$ 2.800,00)',
    },
    {
      cat: 'Saúde Primária (SUS)',
      qty: intFormatter.format(Math.floor(total5Years / custos.consulta_atencao_basica)),
      desc: 'Consultas médicas de atenção básica (a R$ 65,00)',
    },
    {
      cat: 'Assistência Farmacêutica',
      qty: intFormatter.format(Math.floor(total5Years / custos.kit_medicamentos_basicos)),
      desc: 'Kits mensais da Farmácia Básica (a R$ 45,00)',
    },
    {
      cat: 'Segurança Pública Estadual',
      qty: intFormatter.format(Math.floor(total5Years / custos.equipamento_seguranca_uf)),
      desc: 'Diárias operacionais de patrulhamento (a R$ 420,00)',
    },
    {
      cat: 'Segurança Alimentar (DIEESE)',
      qty: intFormatter.format(Math.floor(total5Years / custos.cesta_basica_familiar)),
      desc: 'Cestas básicas familiares (a R$ 750,00)',
    },
    {
      cat: 'Renda de Referência',
      qty: intFormatter.format(Math.floor(total5Years / custos.salario_minimo)),
      desc: 'Salários mínimos equivalentes (a R$ 1.518,00)',
    },
  ];

  const container = document.getElementById('impact-cards-grid');
  container.innerHTML = items
    .map(
      (item) => `
      <article class="impact-item">
        <span class="impact-item-cat">${item.cat}</span>
        <strong class="impact-item-qty">${item.qty}</strong>
        <span class="impact-item-desc">${item.desc}</span>
      </article>
    `
    )
    .join('');
}

function setupSimulator(data) {
  const catSelect = document.getElementById('sim-categoria');
  const ufSelect = document.getElementById('sim-uf');
  const partidoSelect = document.getElementById('sim-partido');
  const valorInput = document.getElementById('sim-valor');
  const mesInput = document.getElementById('sim-mes');

  const mapaCat = data.mappings.mapaCategoria;
  catSelect.innerHTML = Object.entries(mapaCat)
    .map(([nome, cod]) => `<option value="${cod}">${nome}</option>`)
    .join('');

  const mapaUF = data.mappings.mapaUF;
  ufSelect.innerHTML = Object.entries(mapaUF)
    .map(([uf, cod]) => `<option value="${cod}">${uf}</option>`)
    .join('');

  const mapaPartido = data.mappings.mapaPartido;
  partidoSelect.innerHTML = Object.entries(mapaPartido)
    .map(([partido, cod]) => `<option value="${cod}">${partido}</option>`)
    .join('');

  const updateSimulation = () => {
    const valor = parseFloat(valorInput.value) || 0;
    const codCat = parseInt(catSelect.value, 10) || 1;
    const mes = parseInt(mesInput.value, 10) || 1;
    const codPartido = parseInt(partidoSelect.value, 10) || 1;
    const codUF = parseInt(ufSelect.value, 10) || 1;

    const result = predictMlpBrowser(
      [valor, codCat, mes, codPartido, codUF],
      data.modelParams
    );

    const catStats = data.mappings.categorias[String(codCat)] || {
      media_aceitavel: valor * 0.4,
      limiar_atipico: valor * 0.8,
    };

    const verdictEl = document.getElementById('sim-verdict');
    const probEl = document.getElementById('sim-prob');
    const barEl = document.getElementById('sim-prob-bar');
    const previewEl = document.getElementById('sim-impact-preview');

    const pct = (result.probabilidadeAtipico * 100).toFixed(1);
    probEl.textContent = `${pct}%`;
    barEl.style.inlineSize = `${Math.max(3, result.probabilidadeAtipico * 100)}%`;

    if (result.classePredita === 1) {
      verdictEl.textContent = 'CLASSE 1 — Gasto Atípico / Suspeito de Excesso';
      verdictEl.style.color = '#f43f5e';
      barEl.style.backgroundColor = '#f43f5e';

      const excessoMensal = Math.max(valor - catStats.media_aceitavel, valor * 0.3);
      const impacto5Anos = excessoMensal * 60;
      const merendas = Math.floor(impacto5Anos / 1.5);
      const consultas = Math.floor(impacto5Anos / 65.0);

      previewEl.innerHTML = `
        <strong>Projeção de Impacto Social (60 meses):</strong> Excesso mensal de
        <strong>${brlFormatter.format(excessoMensal)}</strong> sobre a média normal da categoria
        (${brlFormatter.format(catStats.media_aceitavel)}). Em 5 anos, acumula
        <strong>${brlFormatter.format(impacto5Anos)}</strong> — o equivalente a
        <strong>${intFormatter.format(merendas)} merendas escolares (PNAE)</strong> ou
        <strong>${intFormatter.format(consultas)} consultas médicas no SUS</strong>.
      `;
    } else {
      verdictEl.textContent = 'CLASSE 0 — Gasto Dentro do Padrão Normal da Categoria';
      verdictEl.style.color = '#10b981';
      barEl.style.backgroundColor = '#10b981';
      previewEl.innerHTML = `
        Valor dentro do comportamento histórico esperado para a categoria (média aceitável:
        <strong>${brlFormatter.format(catStats.media_aceitavel)}</strong> | limiar estatístico:
        <strong>${brlFormatter.format(catStats.limiar_atipico)}</strong>).
      `;
    }

    const neuronsGrid = document.getElementById('neurons-grid');
    neuronsGrid.innerHTML = result.a1
      .map((val, idx) => {
        const intensity = Math.min(1, Math.abs(val));
        const bg =
          val >= 0
            ? `rgba(56, 189, 248, ${0.12 + intensity * 0.35})`
            : `rgba(244, 63, 94, ${0.12 + intensity * 0.35})`;
        return `<div class="neuron-cell" style="background:${bg}" title="Neurônio Oculto #${idx + 1}">
          h${idx + 1}: ${val.toFixed(2)}
        </div>`;
      })
      .join('');
  };

  [valorInput, catSelect, ufSelect, partidoSelect, mesInput].forEach((el) =>
    el.addEventListener('input', updateSimulation)
  );
  updateSimulation();
}

function setupTable(records) {
  const ufFilter = document.getElementById('filter-uf');
  const statusFilter = document.getElementById('filter-status');
  const searchInput = document.getElementById('filter-search');
  const tbody = document.getElementById('audit-tbody');

  const ufs = Array.from(new Set(records.map((r) => r.siglaUF))).sort();
  ufFilter.innerHTML =
    '<option value="all">Todas as UFs</option>' +
    ufs.map((uf) => `<option value="${uf}">${uf}</option>`).join('');

  const renderRows = () => {
    const statusVal = statusFilter.value;
    const ufVal = ufFilter.value;
    const query = searchInput.value.trim().toLowerCase();

    const filtered = records.filter((r) => {
      if (statusVal !== 'all' && String(r.classePredita) !== statusVal) return false;
      if (ufVal !== 'all' && r.siglaUF !== ufVal) return false;
      if (query) {
        const hay = `${r.nomeParlamentar} ${r.siglaPartido} ${r.nomeFornecedor} ${r.cnpjCpfFornecedor} ${r.tipoDespesa}`.toLowerCase();
        if (!hay.includes(query)) return false;
      }
      return true;
    });

    tbody.innerHTML = filtered
      .map((r) => {
        const merendas = Math.floor(r.impacto5Anos / 1.5);
        const badge =
          r.classePredita === 1
            ? `<span class="badge-atipico">Atípico (${(r.probabilidadeAtipico * 100).toFixed(1)}%)</span>`
            : `<span class="badge-normal">Normal (${((1 - r.probabilidadeAtipico) * 100).toFixed(1)}%)</span>`;

        const docLink = r.urlDocumento && r.urlDocumento.startsWith('http')
          ? `<a href="${r.urlDocumento}" target="_blank" rel="noopener noreferrer" class="invoice-link">Abrir PDF ↗</a>`
          : 'N/D';

        return `
          <tr>
            <th scope="row">${r.nomeParlamentar} (${r.siglaPartido}-${r.siglaUF})</th>
            <td>${r.tipoDespesa}</td>
            <td>${r.nomeFornecedor} <span class="mono">(${r.cnpjCpfFornecedor})</span></td>
            <td class="mono"><strong>${brlFormatter.format(r.valorLiquido)}</strong></td>
            <td class="mono">${brlFormatter.format(r.mediaAceitavelCategoria)}</td>
            <td class="mono">${r.classePredita === 1 ? brlFormatter.format(r.impacto5Anos) : '—'}</td>
            <td>${r.classePredita === 1 ? `${intFormatter.format(merendas)} merendas` : 'Dentro do padrão'}</td>
            <td>${badge}</td>
            <td>${docLink}</td>
          </tr>
        `;
      })
      .join('');
  };

  [ufFilter, statusFilter, searchInput].forEach((el) =>
    el.addEventListener('input', renderRows)
  );
  renderRows();
}

async function init() {
  const response = await fetch('/data/dashboard_data.json');
  dashboardData = await response.json();

  renderKPIs(dashboardData.metrics);
  renderImpactCards(dashboardData.metrics.impact5YearsTotal, dashboardData.custosReferencia);
  setupSimulator(dashboardData);
  setupTable(dashboardData.records);
}

init();
