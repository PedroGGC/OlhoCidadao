"""Diagnóstico offline: não treina nem altera os artefatos usados pelo portal."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.nn_model import NnModel

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'data/processed_dataset.csv', low_memory=False)
payload = json.loads((ROOT / 'public/data/dashboard_data.json').read_text(encoding='utf-8'))
params = payload['modelParams']
published_metrics = payload['metrics']
del payload


def metrics(y, prediction, probabilities=None):
    tp = int(((prediction == 1) & (y == 1)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result = dict(n=len(y), positives=int(y.sum()), alerts=int(prediction.sum()), tp=tp, fp=fp, tn=tn, fn=fn,
                  accuracy=float(np.mean(prediction == y)), precision=precision, recall=recall,
                  f1=2 * precision * recall / max(1e-15, precision + recall))
    if probabilities is not None:
        result['brier'] = float(np.mean((probabilities - y) ** 2))
        bins = []
        for left in np.arange(0, 1, 0.1):
            mask = (probabilities >= left) & (probabilities < left + 0.1)
            if left > 0.89:
                mask = (probabilities >= left) & (probabilities <= 1)
            if mask.any():
                bins.append(dict(lower=round(float(left), 1), n=int(mask.sum()),
                                 mean_score=float(probabilities[mask].mean()), positive_rate=float(y[mask].mean())))
        result['calibration_bins'] = bins
        result['ece10'] = sum(b['n'] * abs(b['mean_score'] - b['positive_rate']) for b in bins) / len(y)
    return result


features = ['valorLiquido', 'codTipoDespesa', 'mes', 'codPartido', 'codUF']
X = df[features].to_numpy(float)
y = df.target.to_numpy(int)
test_idx = np.random.default_rng(42).choice(len(df), min(2000, len(df) // 5), replace=False)
test = df.iloc[test_idx]
xtest = X[test_idx]
yt = y[test_idx]
pt = test.probabilidadeAtipico.to_numpy(float)
pred = test.classePredita.to_numpy(int)
model = NnModel(np.zeros((1, 5)), np.zeros(1, dtype=int), n_hidden=params['n_hidden'], activation=params['activation'])
for name in ['W1', 'W2']:
    setattr(model, name, np.asarray(params[name]))
for name in ['b1', 'b2']:
    setattr(model, name, np.asarray(params[name]).reshape(1, -1))
mean, std = np.asarray(params['mean']), np.asarray(params['std'])
forward = lambda x: model.predict((x - mean) / (std + 1e-8))
reproduced, scores = forward(xtest)
assert np.array_equal(reproduced, pred)
assert np.allclose(scores, pt)
report = {'published_metrics': published_metrics,
          'holdout_current': metrics(yt, pred, pt),
          'holdout_always_normal': metrics(yt, np.zeros_like(yt)),
          'holdout_direct_label_rule': metrics(yt, (test.valorLiquido.to_numpy() > test.limiarCategoria.to_numpy()).astype(int)),
          'full_dataset': metrics(y, df.classePredita.to_numpy(int)),
          'thresholds_exploratory_not_validation': {str(t): metrics(yt, (pt >= t).astype(int)) for t in [0.5, 0.7, 0.8, 0.9, 0.95]},
          'holdout_by_house': {}, 'holdout_by_year': {}}
for column, key in [('cargoPublico', 'holdout_by_house'), ('ano', 'holdout_by_year')]:
    for group in test[column].unique():
        mask = test[column].to_numpy() == group
        report[key][str(group)] = metrics(yt[mask], pred[mask])

# Os códigos de UF/partido não fazem parte da regra que gera os rótulos.
# Variações abaixo são uma análise de sensibilidade, não uma conclusão causal.
for field, feature_index in [('codUF', 4), ('codPartido', 3)]:
    alternatives = []
    probabilities = []
    for code in sorted(df[field].unique()):
        changed = xtest.copy()
        changed[:, feature_index] = code
        alternative, probs = forward(changed)
        alternatives.append(alternative)
        probabilities.append(probs)
    alternatives = np.array(alternatives)
    probabilities = np.array(probabilities)
    report['sensitivity_' + field] = {
        'holdout_records_changing_class_under_any_code': int((alternatives.max(axis=0) != alternatives.min(axis=0)).sum()),
        'mean_score_range': float(np.ptp(probabilities, axis=0).mean()),
        'maximum_score_range': float(np.ptp(probabilities, axis=0).max()),
    }

# Verificação numérica do gradiente da loss média implementada.
rng = np.random.default_rng(7)
small_x = rng.normal(size=(16, 5))
small_y = np.tile([0, 1], 8)
check = NnModel(small_x, small_y, n_hidden=4, lr=1e-4)
original = check.W1.copy()
step = 1e-6
check.W1[0, 0] = original[0, 0] + step
positive_loss = check.loss(check.forward(), small_y)
check.W1[0, 0] = original[0, 0] - step
negative_loss = check.loss(check.forward(), small_y)
numeric = (positive_loss - negative_loss) / (2 * step)
check.W1 = original.copy()
check.forward()
check.backpropagation()
actual = (original[0, 0] - check.W1[0, 0]) / check.lr
report['gradient_check'] = dict(n=16, numerical_gradient=float(numeric), applied_gradient=float(actual),
                                ratio=float(actual / numeric))

report['test_parliamentarians_overlapping_other_rows'] = int(test.idDeputado.isin(df.drop(index=test_idx).idDeputado).sum())
report['label_prevalence'] = float(y.mean())
report['rule_targets_verified'] = bool(np.array_equal(y, (df.valorLiquido > df.limiarCategoria).astype(int)))
(OUT / 'metricas_ia.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
compact = {key: value for key, value in report.items() if key not in ['published_metrics', 'holdout_current', 'thresholds_exploratory_not_validation']}
compact['holdout_current'] = {key: value for key, value in report['holdout_current'].items() if key != 'calibration_bins'}
print(json.dumps(compact, ensure_ascii=False, indent=2))
