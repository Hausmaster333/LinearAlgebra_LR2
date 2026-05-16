import unittest

import numpy as np

from src.data import standardize_train_test, stratified_kfold_indices
from src.metrics import accuracy, classification_metrics, roc_auc_score
from src.perceptron import Perceptron


class CoreTests(unittest.TestCase):
    def test_sigmoid_range(self):
        values = Perceptron.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))
        self.assertAlmostEqual(values[1], 0.5)

    def test_bce_is_finite_near_edges(self):
        model = Perceptron(input_dim=2)
        loss = model.compute_loss(np.array([0, 1]), np.array([1e-20, 1.0 - 1e-20]))
        self.assertTrue(np.isfinite(loss))

    def test_standardization_uses_train_statistics(self):
        X_train = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        X_test = np.array([[7.0, 14.0]])
        Xs_train, Xs_test, mean, std = standardize_train_test(X_train, X_test)
        self.assertTrue(np.allclose(Xs_train.mean(axis=0), 0.0))
        self.assertTrue(np.allclose(Xs_test, (X_test - mean) / std))

    def test_metrics_manual_values(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        values = classification_metrics(y_true, y_pred, np.array([0.9, 0.4, 0.8, 0.1]))
        self.assertAlmostEqual(values["accuracy"], 0.5)
        self.assertAlmostEqual(values["precision"], 0.5)
        self.assertAlmostEqual(values["recall"], 0.5)
        self.assertAlmostEqual(values["f1"], 0.5)
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.5)
        self.assertTrue(0.0 <= roc_auc_score(y_true, np.array([0.9, 0.4, 0.8, 0.1])) <= 1.0)

    def test_stratified_kfold_has_both_classes(self):
        y = np.array([0] * 20 + [1] * 20)
        folds = stratified_kfold_indices(y, n_splits=5, random_state=1)
        self.assertEqual(len(folds), 5)
        for _, val_idx in folds:
            self.assertEqual(set(y[val_idx]), {0, 1})

    def test_perceptron_learns_simple_boundary(self):
        X = np.array([[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]])
        y = np.array([0, 0, 1, 1])
        model = Perceptron(input_dim=2, random_state=1)
        model.fit(X, y, X, y, epochs=80, lr=0.1, batch_size=2)
        self.assertGreaterEqual(accuracy(y, model.predict(X)), 0.99)


if __name__ == "__main__":
    unittest.main()
