export function workMunicipalities(work) {
  return Array.isArray(work.municipios) ? work.municipios : [];
}

export function comparisonValue(work) {
  const value = work.valorTotalObra;
  // O cadastro contém valores simbólicos como R$ 0,01. Continua exibindo o
  // valor publicado, mas evita tratá-lo como orçamento de uma intervenção.
  const minimum = work.tipoRegistro === 'oficial' ? 1000 : 0;
  return Number.isFinite(value) && value > minimum ? value : null;
}

export function matchesWorkState(work, uf) {
  return uf === 'all' || work.uf === uf || workMunicipalities(work).some((city) => city.uf === uf);
}

export function filterWorks(works, { uf = 'all', city = 'all', type = 'oficial', year = 'all', sector = 'all', status = 'all' } = {}) {
  return works.filter((work) => {
    const official = work.tipoRegistro === 'oficial';
    if (type !== 'all' && (type === 'oficial') !== official) return false;
    if (!matchesWorkState(work, uf)) return false;
    if (city !== 'all' && !workMunicipalities(work).some((item) => item.codigoIBGE === city && (uf === 'all' || item.uf === uf))) return false;
    if (year !== 'all' && String(work.ano) !== year) return false;
    if (sector !== 'all' && work.setor !== sector) return false;
    if (status !== 'all' && work.situacao !== status) return false;
    return true;
  });
}

export function cityOptions(works, uf = 'all') {
  const cities = new Map();
  for (const work of works) {
    if (work.tipoRegistro !== 'oficial') continue;
    for (const city of workMunicipalities(work)) {
      if (uf === 'all' || city.uf === uf) cities.set(city.codigoIBGE, city);
    }
  }
  return [...cities.values()].sort((a, b) => a.nome.localeCompare(b.nome, 'pt-BR') || a.uf.localeCompare(b.uf));
}
