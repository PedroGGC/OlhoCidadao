"""Verifica o modelo congelado e gera casos NumPy para os testes JavaScript."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd

from src.model_evaluation import ExpenseFeatures, temporal_partitions, classification_metrics
from src.model_export import load_model, export_parity_cases


def main():
    payload = json.loads((root / 'public/data/dashboard_data.json').read_text(encoding='utf-8'))
    params = payload['modelParams']
    df = pd.read_csv(root / 'data/processed_dataset.csv', low_memory=False)
    model = load_model(params)
    features = ExpenseFeatures.from_config(params['featureTransform']).transform(df)
    features = (features-np.asarray(params['mean']))/(np.asarray(params['std'])+1e-8)
    predictions, probabilities = model.predict(features)
    np.testing.assert_array_equal(predictions, df.classePredita)
    np.testing.assert_allclose(probabilities, df.probabilidadeAtipico, atol=1e-10, rtol=1e-10)
    report = json.loads((root / 'outputs/model_evaluation.json').read_text(encoding='utf-8'))
    assert report == payload['metrics']['evaluation']
    test = temporal_partitions(df)['test']
    measured = classification_metrics(df.target.to_numpy()[test], probabilities[test], model.decision_threshold)
    assert measured['confusionMatrix'] == report['test']['confusionMatrix']
    for metric in ('accuracy', 'precision', 'recall', 'f1', 'averagePrecision', 'brier', 'ece'):
        np.testing.assert_allclose(measured[metric], report['test'][metric], atol=1e-12)
    export_parity_cases(model, params, df, path=str(root / 'outputs/model_parity.json'))
    print(f'OK: {len(df)} inferencias e {len(test)} notas de teste conciliadas com pesos, CSV e dashboard.')
    print(f'Precisao={measured["precision"]:.4%}; recall={measured["recall"]:.4%}; F1={measured["f1"]:.6f}')


if __name__ == '__main__':
    main()
