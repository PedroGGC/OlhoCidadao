import unittest

import numpy as np
import pandas as pd

from src.model_evaluation import (ExpenseFeatures, historical_reference, temporal_partitions,
                                  fit_calibration, calibrate, select_threshold, classification_metrics)
from src.nn_model import NnModel


class AutonomousModelTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{'ano': year, 'mes': month, 'valorLiquido': value,
                              'codTipoDespesa': code, 'tipoDespesa': str(code)}
                             for year, month in [(2025, 1), (2026, 2), (2026, 4), (2026, 6), (2026, 8)]
                             for code in [1, 2] for value in [10., 20., 25., 100.]])

    def test_future_changes_neither_historical_targets_nor_reference_features(self):
        original = self.frame()
        changed = original.copy()
        changed.loc[changed.ano == 2026, 'valorLiquido'] *= 100000
        first, first_stats = historical_reference(original)
        second, second_stats = historical_reference(changed)
        self.assertEqual(first_stats, second_stats)
        past = temporal_partitions(first)['reference']
        np.testing.assert_array_equal(first.iloc[past].target, second.iloc[past].target)
        a = ExpenseFeatures().fit(first.iloc[past])
        b = ExpenseFeatures().fit(second.iloc[past])
        self.assertEqual(a.export(), b.export())

    def test_temporal_roles_are_exclusive_and_ordered(self):
        df = self.frame()
        parts = temporal_partitions(df)
        seen, previous = set(), 0
        periods = df.ano.to_numpy()*100+df.mes.to_numpy()
        for indices in parts.values():
            self.assertFalse(seen.intersection(indices))
            seen.update(indices)
            self.assertGreater(periods[indices].min(), previous)
            previous = periods[indices].max()
        self.assertEqual(len(seen), len(df))

    def test_gradients_match_finite_differences_for_all_activations(self):
        rng = np.random.default_rng(9)
        X, y = rng.normal(size=(12, 3)), np.arange(12) % 2
        for activation in ('tanh', 'sigmoid', 'leaky_relu'):
            model = NnModel(X, y, n_hidden=4, lr=.001, activation=activation, weight_decay=.002)
            numerical = {}
            for name in ('W1', 'b1', 'W2', 'b2'):
                parameter = getattr(model, name)
                gradient = np.empty_like(parameter)
                for index in np.ndindex(parameter.shape):
                    value = parameter[index]
                    parameter[index] = value+1e-6
                    high = model.loss(model.forward())
                    parameter[index] = value-1e-6
                    low = model.loss(model.forward())
                    parameter[index] = value
                    gradient[index] = (high-low)/2e-6
                numerical[name] = gradient
            before = {name: getattr(model, name).copy() for name in numerical}
            model.forward()
            model.backpropagation()
            for name in numerical:
                np.testing.assert_allclose((before[name]-getattr(model, name))/model.lr,
                                           numerical[name], atol=1e-7, rtol=1e-5)

    def test_calibration_corrects_prior_shift_on_an_independent_population(self):
        rng = np.random.default_rng(17)
        logits = rng.normal(size=12000)*2
        truth_probability = 1/(1+np.exp(-(logits-2)))
        truth = (rng.random(len(logits)) < truth_probability).astype(int)
        calibration = fit_calibration(logits[:6000], truth[:6000])
        raw = 1/(1+np.exp(-logits[6000:]))
        corrected = calibrate(logits[6000:], calibration)
        self.assertLess(np.mean((corrected-truth[6000:])**2), np.mean((raw-truth[6000:])**2))
        self.assertGreater(calibration['slope'], 0)

    def test_threshold_respects_recall_and_does_not_split_tied_scores(self):
        y = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        p = np.array([.9, .9, .8, .7, .6, .4, .3, .2])
        threshold = select_threshold(y, p, minimum_recall=.9)
        metrics = classification_metrics(y, p, threshold)
        self.assertGreaterEqual(metrics['recall'], .9)
        self.assertEqual(metrics['confusionMatrix'], {'vp': 3, 'vn': 3, 'fp': 2, 'fn': 0})

    def test_protected_attributes_are_not_model_inputs_and_unknown_category_is_finite(self):
        df = self.frame()
        transform = ExpenseFeatures().fit(df[df.ano == 2025])
        a = df.assign(siglaUF='MA', siglaPartido='X')
        b = df.assign(siglaUF='SP', siglaPartido='Y')
        np.testing.assert_array_equal(transform.transform(a), transform.transform(b))
        unknown = transform.transform_values([100.], [999])
        self.assertTrue(np.isfinite(unknown).all())
        self.assertEqual(unknown[0, -1], 1)
        self.assertEqual(unknown[0, 2:-1].sum(), 0)

    def test_early_stopping_restores_selected_weights(self):
        X = np.array([[-1.], [-.5], [.5], [1.]])
        y = np.array([0, 0, 1, 1])
        model = NnModel(X, y, optimizer='adam', lr=.05)
        model.fit(epochs=100, X_val=X, y_val=1-y, patience=4, save_curve=False, verbose_step=100)
        self.assertLess(model.epochs_run, 100)
        self.assertAlmostEqual(model.loss(model.forward(X), 1-y), model.val_loss_history[-1])


if __name__ == '__main__':
    unittest.main()
