"""Avaliação automática e temporal; nenhuma etapa depende de revisão humana.

O alvo é atipicidade estatística, não fraude. Referências são congeladas em
2025; meses posteriores têm papéis separados antes de qualquer treinamento.
"""
import numpy as np
import pandas as pd


def temporal_partitions(df):
    period = df['ano'].to_numpy(int) * 100 + df['mes'].to_numpy(int)
    masks = {
        'reference': period <= 202512,
        'validation': (period >= 202601) & (period <= 202603),
        'calibration': (period >= 202604) & (period <= 202605),
        'threshold': period == 202606,
        'test': period >= 202607,
    }
    parts = {name: np.flatnonzero(mask) for name, mask in masks.items()}
    if any(len(indices) == 0 for indices in parts.values()):
        raise ValueError('Avaliação exige 2025 e os períodos de validação, calibração, limiar e teste de 2026.')
    return parts


def historical_reference(df):
    """Não usa valores futuros nem o alvo como entrada da rede."""
    reference = df.iloc[temporal_partitions(df)['reference']]

    def stats(group):
        values = group['valorLiquido'].to_numpy(float)
        mean, std = float(values.mean()), float(values.std())
        threshold = mean + 1.2 * std
        return {'media_total': mean, 'desvio_padrao': std,
                'limiar_atipico': threshold,
                'media_aceitavel': float(values[values <= threshold].mean()),
                'referenceRecords': len(values)}

    fallback = stats(reference)
    known = {int(code): dict(stats(group), nome=str(group.tipoDespesa.iloc[0]))
             for code, group in reference.groupby('codTipoDespesa')}
    result = {int(code): known.get(int(code), dict(fallback, nome=str(group.tipoDespesa.iloc[0]),
                                                  fallbackReference=True))
              for code, group in df.groupby('codTipoDespesa')}
    df = df.copy()
    df['limiarCategoria'] = df.codTipoDespesa.map({k: v['limiar_atipico'] for k, v in result.items()})
    df['mediaAceitavelCategoria'] = df.codTipoDespesa.map({k: v['media_aceitavel'] for k, v in result.items()})
    df['target'] = (df.valorLiquido > df.limiarCategoria).astype(int)
    return df, result


class ExpenseFeatures:
    """Log do valor, contexto histórico e categorias one-hot (sem ordem fictícia).

    Não recebe target, limiar, partido, UF ou identidade do parlamentar.
    """
    def fit(self, reference):
        self.categories = sorted(int(x) for x in reference.codTipoDespesa.unique())
        logs = np.log1p(reference.valorLiquido.to_numpy(float))
        self.fallback = [float(logs.mean()), max(float(logs.std()), 0.1)]
        self.context = {}
        for code, group in reference.groupby('codTipoDespesa'):
            values = np.log1p(group.valorLiquido.to_numpy(float))
            self.context[int(code)] = [float(values.mean()), max(float(values.std()), 0.1)]
        return self

    def transform(self, frame):
        return self.transform_values(frame.valorLiquido.to_numpy(float), frame.codTipoDespesa.to_numpy(int))

    def transform_values(self, values, codes):
        logs = np.log1p(np.maximum(np.asarray(values, dtype=float), 0))
        codes = np.asarray(codes, dtype=int)
        means = np.array([self.context.get(int(c), self.fallback)[0] for c in codes])
        scales = np.array([self.context.get(int(c), self.fallback)[1] for c in codes])
        onehot = np.column_stack([codes == c for c in self.categories] + [~np.isin(codes, self.categories)])
        return np.column_stack([logs, (logs - means) / scales, onehot]).astype(float)

    def export(self):
        return {'version': 2, 'categories': self.categories, 'context': self.context,
                'fallback': self.fallback,
                'featureNames': ['log1p_valor', 'log_valor_padronizado_categoria'] +
                                [f'categoria_{c}' for c in self.categories] + ['categoria_desconhecida']}

    @classmethod
    def from_config(cls, config):
        if config.get('version') != 2:
            raise ValueError('Versão desconhecida da transformação de entradas.')
        result = cls()
        result.categories = list(config['categories'])
        result.context = {int(code): value for code, value in config['context'].items()}
        result.fallback = list(config['fallback'])
        return result


def sigmoid(logits):
    logits = np.asarray(logits, dtype=float)
    return np.exp(-np.logaddexp(0, -logits))


def fit_calibration(logits, labels):
    """Platt scaling: Newton com busca de passo, em período exclusivo e prevalência natural."""
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels, dtype=float)
    if len(np.unique(labels)) != 2:
        raise ValueError('Calibração requer ambas as classes.')
    design = np.column_stack([logits, np.ones(len(logits))])
    weights = np.array([1.0, 0.0])
    penalty = np.diag([1e-5, 1e-8])

    def objective(w):
        z = design @ w
        return float(np.mean(np.logaddexp(0, z) - labels * z) + .5 * w @ penalty @ w)

    for _ in range(80):
        p = sigmoid(design @ weights)
        gradient = design.T @ (p - labels) / len(labels) + penalty @ weights
        hessian = (design.T * (p * (1 - p))) @ design / len(labels) + penalty
        step = np.linalg.solve(hessian, gradient)
        rate, current = 1.0, objective(weights)
        while rate > 1e-8:
            proposal = weights - rate * step
            if proposal[0] > 0 and objective(proposal) <= current:
                break
            rate *= .5
        if rate <= 1e-8:
            break
        weights = proposal
        if np.max(np.abs(rate * step)) < 1e-7:
            break
    return {'slope': float(weights[0]), 'intercept': float(weights[1])}


def calibrate(logits, calibration):
    return sigmoid(calibration['slope'] * np.asarray(logits) + calibration['intercept'])


def select_threshold(labels, probabilities, minimum_recall=.90):
    """Maximiza F1 sob piso de recall; usa somente junho, nunca o teste final."""
    labels, probabilities = np.asarray(labels, int), np.asarray(probabilities, float)
    if not np.any(labels == 1):
        raise ValueError('Seleção de limiar requer exemplos positivos.')
    order = np.argsort(-probabilities, kind='stable')
    scores, truth = probabilities[order], labels[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = np.cumsum(truth)[ends]
    precision, recall = tp / (ends + 1), tp / labels.sum()
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    eligible = np.flatnonzero(recall >= minimum_recall)
    best = eligible[np.argmax(f1[eligible])]
    return float(scores[ends[best]])


def _wilson(success, total):
    if total == 0:
        return [0., 0.]
    p, z = success / total, 1.96
    center = p + z*z / (2*total)
    spread = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total))
    return [float((center-spread)/(1+z*z/total)), float((center+spread)/(1+z*z/total))]


def classification_metrics(labels, probabilities, threshold=.5):
    labels, probabilities = np.asarray(labels, int), np.asarray(probabilities, float)
    predictions = probabilities >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))
    precision, recall = tp / max(tp+fp, 1), tp / max(tp+fn, 1)
    ece, bins = 0., []
    for i in range(10):
        mask = (probabilities >= i/10) & ((probabilities < (i+1)/10) if i < 9 else (probabilities <= 1))
        if mask.any():
            confidence, frequency = float(probabilities[mask].mean()), float(labels[mask].mean())
            ece += float(mask.mean()) * abs(confidence - frequency)
            bins.append({'count': int(mask.sum()), 'meanScore': confidence, 'positiveRate': frequency})
    order = np.argsort(-probabilities, kind='stable')
    sorted_p, truth = probabilities[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(labels)-1]
    cumulative = np.cumsum(truth)[ends]
    ap = np.sum(np.diff(np.r_[0, cumulative]) / max(int(labels.sum()), 1) * cumulative / (ends+1))
    return {'records': len(labels), 'positives': int(labels.sum()), 'alerts': tp+fp,
            'accuracy': (tp+tn)/len(labels), 'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/max(precision+recall, 1e-12),
            'balancedAccuracy': (recall + tn/max(tn+fp, 1))/2,
            'averagePrecision': float(ap), 'brier': float(np.mean((probabilities-labels)**2)),
            'ece': ece, 'precisionCI95': _wilson(tp, tp+fp), 'recallCI95': _wilson(tp, tp+fn),
            'confusionMatrix': {'vp': tp, 'vn': tn, 'fp': fp, 'fn': fn}, 'calibrationBins': bins}
