"""Concilia dumps, CSVs e payload sem rede: python scripts/validate_dataset.py."""
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / 'data'
camara = pd.read_csv(data / 'cota_parlamentar_camara_completa.csv', dtype=str, keep_default_na=False)
senado = pd.read_csv(data / 'despesas_senadores_ceaps_completo.csv', dtype=str, keep_default_na=False)
raw = pd.read_csv(data / 'raw_despesas_camara.csv', low_memory=False)
processed = pd.read_csv(data / 'processed_dataset.csv', low_memory=False)
payload_path = root / 'public/data/dashboard_data.json'
payload = json.loads(payload_path.read_text(encoding='utf-8'))

assert len(camara) + len(senado) == len(raw) == len(processed) == payload['recordCount']
assert len(processed) == payload['metrics']['webRecords']
assert processed.siglaUF.nunique() == 27
assert (processed.valorLiquido > 0).all()
assert np.isfinite(processed[['valorLiquido', 'probabilidadeAtipico', 'limiarCategoria']].to_numpy()).all()
assert processed.probabilidadeAtipico.between(0, 1).all()
assert set(processed.classePredita.unique()) <= {0, 1}
assert payload['metrics']['trainRecords'] <= 5000
assert payload['metrics']['sampling'] == 'none'

def cents(series):
    return np.rint(np.asarray(series, dtype=float) * 100).astype(np.int64)

source_cents = int(cents(camara.vlrLiquido).sum() + cents(senado.valorReembolsado).sum())
assert source_cents == int(cents(raw.valorLiquido).sum()) == int(cents(processed.valorLiquido).sum())
columns = payload['recordColumns']
for name, column in columns.items():
    assert len(column['values']) == len(processed), name
    if 'dictionary' in column:
        assert min(column['values']) >= 0 and max(column['values']) < len(column['dictionary'])
np.testing.assert_array_equal(cents(columns['valorLiquido']['values']), cents(processed.valorLiquido))
np.testing.assert_array_equal(columns['classePredita']['values'], processed.classePredita)
np.testing.assert_allclose(columns['probabilidadeAtipico']['values'], processed.probabilidadeAtipico, atol=0.000051)
uf_column = columns['siglaUF']
assert [uf_column['dictionary'][i] for i in uf_column['values']] == processed.siglaUF.tolist()
with gzip.open(str(payload_path) + '.gz', 'rb') as source:
    assert hashlib.sha256(source.read()).digest() == hashlib.sha256(payload_path.read_bytes()).digest()

summary = processed.groupby(['ano', 'siglaUF']).agg(registros=('valorLiquido', 'size'),
                                                  valorPositivoBRL=('valorLiquido', 'sum')).reset_index()
summary['valorPositivoBRL'] = summary['valorPositivoBRL'].round(2)
report = {
    'validatedAtUTC': datetime.now(timezone.utc).isoformat(),
    'camaraRecords': len(camara), 'senadoRecords': len(senado), 'totalRecords': len(processed),
    'totalPositiveBRL': source_cents / 100, 'states': 27,
    'minimumRecordsPerStateBothYears': int(processed.groupby('siglaUF').size().min()),
    'senatorsWithExpensesBothYears': int(senado.codSenador.nunique()),
    'trainingRecords': payload['metrics']['trainRecords'], 'testRecords': payload['metrics']['testRecords'],
    'jsonBytes': payload_path.stat().st_size, 'gzipBytes': Path(str(payload_path) + '.gz').stat().st_size,
    'coverage': summary.to_dict(orient='records'),
    'sha256': {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in [
        'cota_parlamentar_camara_completa.csv', 'despesas_senadores_ceaps_completo.csv',
        'raw_despesas_camara.csv', 'processed_dataset.csv']},
}
(root / 'outputs/data_quality_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in report.items() if key not in ['coverage', 'sha256']}, indent=2))
print(summary[summary.siglaUF.eq('MA')].to_string(index=False))
