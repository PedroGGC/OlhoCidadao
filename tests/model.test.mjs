import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { predictMlpBrowser } from '../web/model.js';

const fixture = JSON.parse(readFileSync(new URL('../outputs/model_parity.json', import.meta.url), 'utf8'));

test('inferência no navegador coincide com NumPy em notas, fronteiras e extremos', () => {
  for (const row of fixture.cases) {
    const result = predictMlpBrowser(row, fixture.modelParams);
    assert.equal(result.classePredita, row.classePredita);
    assert.ok(Math.abs(result.probabilidadeAtipico - row.probabilidadeAtipico) < 1e-10,
      `Score divergente em ${JSON.stringify(row)}`);
    assert.ok(result.a1.every(Number.isFinite));
  }
});

test('partido, UF e mês não alteram uma despesa idêntica', () => {
  const expense = fixture.cases[0];
  const first = predictMlpBrowser({ ...expense, siglaUF: 'MA', siglaPartido: 'X', mes: 1 }, fixture.modelParams);
  const second = predictMlpBrowser({ ...expense, siglaUF: 'SP', siglaPartido: 'Y', mes: 12 }, fixture.modelParams);
  assert.deepEqual(first, second);
});
