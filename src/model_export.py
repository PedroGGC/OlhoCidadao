"""Contrato do modelo e casos para testar paridade NumPy/JavaScript."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_evaluation import ExpenseFeatures
from src.nn_model import NnModel


def model_parameters(nn, mean, std, transform):
    return {'n_input': nn.n_input, 'n_hidden': nn.n_hidden, 'n_output': nn.n_output,
            'lr': nn.lr, 'activation': nn.activation, 'optimizer': nn.optimizer,
            'featureTransform': transform, 'calibration': nn.calibration,
            'decisionThreshold': nn.decision_threshold,
            'mean': mean.tolist(), 'std': std.tolist(),
            'W1': nn.W1.tolist(), 'b1': nn.b1[0].tolist(),
            'W2': nn.W2.tolist(), 'b2': nn.b2[0].tolist()}


def load_model(params):
    """Restaura apenas parâmetros numéricos, sem desserializar objetos executáveis."""
    model = NnModel(np.zeros((1, params['n_input'])), np.array([0]),
                    n_hidden=params['n_hidden'], activation=params['activation'], lr=params['lr'])
    for name in ('W1', 'b1', 'W2', 'b2'):
        value = np.asarray(params[name], dtype=float).reshape(getattr(model, name).shape)
        if not np.isfinite(value).all():
            raise ValueError('Pesos não finitos.')
        setattr(model, name, value)
    model.optimizer = params['optimizer']
    model.calibration = dict(params['calibration'])
    model.decision_threshold = params['decisionThreshold']
    return model


def export_parity_cases(nn, params, df, path='outputs/model_parity.json'):
    selected = df.sample(n=min(64, len(df)), random_state=17)
    rows = selected[['valorLiquido', 'codTipoDespesa']].to_dict('records')
    # Inclui valores próximos à fronteira, extremos e categoria desconhecida.
    for code, group in df.groupby('codTipoDespesa'):
        threshold = float(group.limiarCategoria.iloc[0])
        rows.extend({'valorLiquido': threshold*factor, 'codTipoDespesa': int(code)}
                    for factor in (0., .99, 1., 1.01, 10.))
    rows.append({'valorLiquido': 100., 'codTipoDespesa': 999999})
    frame = pd.DataFrame(rows)
    features = ExpenseFeatures.from_config(params['featureTransform']).transform(frame)
    features = (features-np.asarray(params['mean']))/(np.asarray(params['std'])+1e-8)
    classes, probabilities = nn.predict(features)
    for row, label, score in zip(rows, classes, probabilities):
        row.update(classePredita=int(label), probabilidadeAtipico=float(score))
    artifact = {'modelParams': params, 'cases': rows, 'source': 'Inferência NumPy; inclui notas e casos artificiais de paridade.'}
    Path(path).write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
