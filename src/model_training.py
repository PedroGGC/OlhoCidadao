"""Treino e seleção automáticos com papéis temporais fixados antes do fit."""
import numpy as np

from src.data_fetcher import normalize_zscore
from src.model_evaluation import (ExpenseFeatures, temporal_partitions, classification_metrics,
                                  fit_calibration, calibrate, select_threshold)
from src.nn_model import NnModel


def train_and_evaluate(df, legacy_inputs, training_sampler, seed=42):
    parts = temporal_partitions(df)
    train = training_sampler(df, parts['reference'], seed=seed)
    rng = np.random.default_rng(seed)
    validation = rng.choice(parts['validation'], min(5000, len(parts['validation'])), replace=False)
    y = df.target.to_numpy(int)
    transformer = ExpenseFeatures().fit(df.iloc[parts['reference']])
    X_raw = transformer.transform(df)
    _, mean, std = normalize_zscore(X_raw[train])
    # One-hot permanece binário, inclusive para categorias desconhecidas no treino.
    mean[2:], std[2:] = 0., 1.
    X, _, _ = normalize_zscore(X_raw, mean, std)

    # Comparação justa: arquitetura anterior retreinada nos mesmos 5.000 registros,
    # sem lhe dar despesas futuras. lr=.4 reproduz o antigo passo efetivo (.05*8).
    old = legacy_inputs.copy()
    medians = df.iloc[parts['reference']].groupby('codTipoDespesa').valorLiquido.median().sort_values()
    rank = {int(code): i+1 for i, code in enumerate(medians.index)}
    old[:, 1] = [rank.get(int(code), 0) for code in old[:, 1]]
    _, old_mean, old_std = normalize_zscore(old[train])
    old, _, _ = normalize_zscore(old, old_mean, old_std)
    baseline = NnModel(old[train], y[train], n_hidden=16, lr=.4, seed=seed)
    baseline.fit(epochs=2000, X_val=old[validation], y_val=y[validation],
                 patience=150, save_curve=False, verbose_step=500)

    candidates = []
    best_model, best_score = None, -1.
    for hidden in (16, 32):
        model = NnModel(X[train], y[train], n_hidden=hidden, lr=.01,
                        optimizer='adam', weight_decay=1e-5, seed=seed)
        model.fit(epochs=1200, X_val=X[validation], y_val=y[validation],
                  patience=100, save_curve=False, verbose_step=400)
        _, scores = model.predict(X[validation])
        metrics = classification_metrics(y[validation], scores)
        candidates.append({'hidden': hidden, 'bestEpoch': model.best_epoch,
                           'epochsRun': model.epochs_run, 'validation': metrics})
        if metrics['averagePrecision'] > best_score:
            best_model, best_score = model, metrics['averagePrecision']
    model = best_model

    calibration_logits = model.logits(X[parts['calibration']])
    model.calibration = fit_calibration(calibration_logits, y[parts['calibration']])
    tuning_scores = calibrate(model.logits(X[parts['threshold']]), model.calibration)
    model.decision_threshold = select_threshold(y[parts['threshold']], tuning_scores, minimum_recall=.90)
    tuning_metrics = classification_metrics(y[parts['threshold']], tuning_scores, model.decision_threshold)

    # Tudo está congelado antes de acessar julho em diante. Nenhum ajuste após isto.
    test = parts['test']
    logits = model.logits(X[test])
    scores = calibrate(logits, model.calibration)
    test_metrics = classification_metrics(y[test], scores, model.decision_threshold)
    _, baseline_scores = baseline.predict(old[test])
    baseline_metrics = classification_metrics(y[test], baseline_scores)
    raw_metrics = classification_metrics(y[test], 1/(1+np.exp(-np.clip(logits, -700, 700))))
    slices = {}
    test_frame = df.iloc[test].reset_index(drop=True)
    for column in ('siglaUF', 'cargoPublico', 'tipoDespesa'):
        slices[column] = {str(name): classification_metrics(y[test][group.index], scores[group.index],
                                                           model.decision_threshold)
                          for name, group in test_frame.groupby(column)}

    # Sensibilidade controlada e determinística, não representa fraude real rotulada.
    normal = test[y[test] == 0]
    probe = df.iloc[rng.choice(normal, min(2000, len(normal)), replace=False)].copy()
    probe['valorLiquido'] = probe.limiarCategoria * 2
    probe_X = transformer.transform(probe)
    probe_X, _, _ = normalize_zscore(probe_X, mean, std)
    probe_predictions, _ = model.predict(probe_X)

    periods = {}
    for name, indices in parts.items():
        period = df.iloc[indices].ano * 100 + df.iloc[indices].mes
        periods[name] = {'start': int(period.min()), 'end': int(period.max()), 'records': len(indices)}
    report = {
        'protocolVersion': 2, 'seed': seed, 'humanReview': False,
        'target': 'Valor > média de 2025 da categoria + 1,2 × desvio de 2025; não é fraude comprovada.',
        'protocol': '2025: referências/treino; jan-mar: parada/arquitetura; abr-mai: calibração; '
                    'jun: limiar F1 com recall mínimo 90%; jul em diante: teste final sem ajustes.',
        'periods': periods, 'trainingRecords': len(train), 'validationSampleRecords': len(validation),
        'featureTransform': transformer.export(), 'candidates': candidates,
        'selectionMetric': 'averagePrecision na validação jan-mar',
        'selectedHidden': model.n_hidden, 'bestEpoch': model.best_epoch, 'epochsRun': model.epochs_run,
        'calibration': model.calibration, 'decisionThreshold': model.decision_threshold,
        'thresholdValidation': tuning_metrics, 'test': test_metrics,
        'baselineSameTemporalTest': baseline_metrics,
        'uncalibratedTest': raw_metrics,
        'allNormalBaselineAccuracy': float(np.mean(y[test] == 0)),
        'testSlices': slices,
        'syntheticStressTest': {'records': len(probe), 'valueMultiplierOfHistoricalThreshold': 2,
                                'detectionRate': float(probe_predictions.mean()),
                                'scope': 'Valores artificialmente aumentados; teste de sensibilidade, não fraude real.'},
    }
    model._save_learning_curve(n_train=len(train), n_val=len(validation))
    return model, X, mean, std, transformer.export(), report
