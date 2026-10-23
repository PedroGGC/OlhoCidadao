import test from 'node:test';
import assert from 'node:assert/strict';
import { cityOptions, filterWorks, comparisonValue } from '../web/works.js';

const works = [
  { codigoObra: 'OG-1', tipoRegistro: 'oficial', uf: 'MA', ano: 2022, municipios: [
    { codigoIBGE: '2111300', nome: 'São Luís', uf: 'MA' },
    { codigoIBGE: '2111201', nome: 'São José de Ribamar', uf: 'MA' }], valorTotalObra: 500000 },
  { codigoObra: 'OG-2', tipoRegistro: 'oficial', uf: 'MA', ano: 2026, municipios: [] },
  { codigoObra: 'OG-3', tipoRegistro: 'oficial', uf: 'PI', ano: 2025, municipios: [
    { codigoIBGE: '2211001', nome: 'Teresina', uf: 'PI' }] },
  { codigoObra: 'REF-1', tipoRegistro: 'referencia', uf: 'MA', ano: 2026, municipios: [] },
];

test('city filters include multicipality works once and exclude unknown locations and estimates', () => {
  assert.deepEqual(filterWorks(works, { uf: 'MA', city: '2111300' }).map((work) => work.codigoObra), ['OG-1']);
  assert.deepEqual(filterWorks(works, { uf: 'MA', city: '2111201' }).map((work) => work.codigoObra), ['OG-1']);
  assert.equal(filterWorks(works, { uf: 'PI', city: '2111300' }).length, 0);
});

test('city options follow state and exclude generic cost references', () => {
  assert.equal(cityOptions(works, 'MA').length, 2);
  assert.equal(cityOptions(works, 'PI')[0].nome, 'Teresina');
  assert.equal(cityOptions(works, 'SP').length, 0);
});

test('all cadastro years remain available independently from parliamentary years', () => {
  assert.deepEqual(filterWorks(works, { uf: 'MA' }).map((work) => work.codigoObra), ['OG-1', 'OG-2']);
  assert.deepEqual(filterWorks(works, { uf: 'MA', year: '2026' }).map((work) => work.codigoObra), ['OG-2']);
});


test('symbolic official values remain visible but cannot inflate comparisons', () => {
  assert.equal(comparisonValue({ tipoRegistro: 'oficial', valorTotalObra: 0.01 }), null);
  assert.equal(comparisonValue({ tipoRegistro: 'oficial', valorTotalObra: null }), null);
  assert.equal(comparisonValue({ tipoRegistro: 'oficial', valorTotalObra: 250000 }), 250000);
});
