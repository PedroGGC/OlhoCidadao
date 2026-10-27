"""MLP construída do zero em NumPy, incluindo gradientes e otimizador Adam."""
import os
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NnModel:
    def __init__(self, X, y, n_hidden=16, lr=.05, activation='tanh', seed=42,
                 optimizer='sgd', weight_decay=0.):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.num_examples, self.n_input = self.X.shape
        self.n_hidden, self.n_output = n_hidden, 2
        self.input_neurons, self.hidden_neurons, self.output_neurons = self.n_input, n_hidden, 2
        self.lr, self.activation = lr, activation.lower()
        if self.activation not in ('tanh', 'leaky_relu') or optimizer not in ('sgd', 'adam'):
            raise ValueError('Ativação ou otimizador inválido.')
        self.optimizer, self.weight_decay = optimizer, weight_decay
        rng = np.random.default_rng(seed)
        limit1, limit2 = np.sqrt(6/(self.n_input+n_hidden)), np.sqrt(6/(n_hidden+2))
        self.W1 = rng.uniform(-limit1, limit1, (self.n_input, n_hidden))
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.uniform(-limit2, limit2, (n_hidden, 2))
        self.b2 = np.zeros((1, 2))
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.calibration = {'slope': 1., 'intercept': 0.}
        self.decision_threshold = .5
        self._adam_m, self._adam_v, self._step = {}, {}, 0

    def _activate_hidden(self, z):
        return np.where(z > 0, z, .01*z) if self.activation == 'leaky_relu' else np.tanh(z)

    def _activate_hidden_derivative(self, a, z):
        return np.where(z > 0, 1., .01) if self.activation == 'leaky_relu' else 1-a*a

    @staticmethod
    def _softmax(z):
        scores = np.exp(z - z.max(axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)

    def forward(self, X=None):
        X = self.X if X is None else X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.f1 = self._activate_hidden(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_pred = self.f2 = self._softmax(self.z2)
        return self.y_pred

    def loss(self, y_pred, y_true=None):
        y_true = self.y if y_true is None else y_true
        data_loss = -np.log(np.clip(y_pred[np.arange(len(y_true)), y_true], 1e-12, 1.)).mean()
        return float(data_loss + .5*self.weight_decay*(np.sum(self.W1**2)+np.sum(self.W2**2)))

    def backpropagation(self, X=None, y=None):
        X, y = (self.X if X is None else X), (self.y if y is None else y)
        delta2 = self.y_pred.copy()
        delta2[np.arange(len(y)), y] -= 1
        # Derivada exata da perda MÉDIA: não multiplicar implicitamente lr por oito.
        delta2 /= len(y)
        delta1 = (delta2 @ self.W2.T) * self._activate_hidden_derivative(self.a1, self.z1)
        gradients = {'W1': X.T @ delta1 + self.weight_decay*self.W1,
                     'b1': delta1.sum(axis=0, keepdims=True),
                     'W2': self.a1.T @ delta2 + self.weight_decay*self.W2,
                     'b2': delta2.sum(axis=0, keepdims=True)}
        self._step += 1
        for name, gradient in gradients.items():
            if self.optimizer == 'adam':
                m = .9*self._adam_m.get(name, np.zeros_like(gradient)) + .1*gradient
                v = .999*self._adam_v.get(name, np.zeros_like(gradient)) + .001*gradient**2
                self._adam_m[name], self._adam_v[name] = m, v
                gradient = (m/(1-.9**self._step)) / (np.sqrt(v/(1-.999**self._step))+1e-8)
            setattr(self, name, getattr(self, name)-self.lr*gradient)

    def fit(self, epochs=1200, verbose_step=250, X_val=None, y_val=None,
            patience=100, min_delta=1e-5, save_curve=True) -> Tuple[List[float], List[float]]:
        """Treina apenas self.X; validação escolhe parada. Teste nunca entra aqui."""
        if (X_val is None) != (y_val is None):
            raise ValueError('Informe X_val e y_val juntos.')
        self.loss_history, self.accuracy_history = [], []
        self.val_loss_history, self.val_accuracy_history = [], []
        best_loss, best_state, best_epoch, stale = float('inf'), None, 0, 0
        print(f'[NnModel] CPU NumPy | treino={len(self.y)} | validação={0 if y_val is None else len(y_val)} '
              f'| {self.n_input} -> {self.n_hidden} -> 2 | {self.optimizer} lr={self.lr}')
        for epoch in range(1, epochs+1):
            self.forward()
            self.backpropagation()
            train_prob = self.forward()
            train_loss = self.loss(train_prob)
            self.loss_history.append(train_loss)
            self.accuracy_history.append(float(np.mean(train_prob.argmax(axis=1) == self.y)))
            monitored = train_loss
            if X_val is not None:
                val_prob = self.forward(X_val)
                monitored = self.loss(val_prob, y_val)
                self.val_loss_history.append(monitored)
                self.val_accuracy_history.append(float(np.mean(val_prob.argmax(axis=1) == y_val)))
            if monitored < best_loss-min_delta:
                best_loss, best_epoch, stale = monitored, epoch, 0
                best_state = {name: getattr(self, name).copy() for name in ('W1', 'b1', 'W2', 'b2')}
            else:
                stale += 1
            if epoch == 1 or epoch % verbose_step == 0:
                print(f'  época={epoch} loss={train_loss:.5f} loss_val={monitored:.5f}')
            if X_val is not None and stale >= patience:
                break
        # Recupera os pesos escolhidos pela validação, não os da última época.
        for name, value in best_state.items():
            setattr(self, name, value)
        self.best_epoch, self.epochs_run = best_epoch, epoch
        self.loss_history = self.loss_history[:best_epoch]
        self.accuracy_history = self.accuracy_history[:best_epoch]
        self.val_loss_history = self.val_loss_history[:best_epoch]
        self.val_accuracy_history = self.val_accuracy_history[:best_epoch]
        print(f'[NnModel] melhor época={best_epoch}; executadas={epoch}; validação={best_loss:.5f}')
        if save_curve:
            self._save_learning_curve(n_train=len(self.y), n_val=0 if y_val is None else len(y_val))
        return self.loss_history, self.accuracy_history

    def _save_learning_curve(self, output_path=os.path.join('outputs', 'loss_curve.png'), n_train=0, n_val=0):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, facecolor='#161c26')
        epochs = np.arange(1, len(self.loss_history)+1)
        series = [(self.loss_history, self.val_loss_history, 'Entropia cruzada'),
                  (np.array(self.accuracy_history)*100, np.array(self.val_accuracy_history)*100, 'Acurácia (%)')]
        for ax, (train, val, title) in zip(axes, series):
            ax.set_facecolor('#1a202b')
            ax.tick_params(colors='#c2cbd8')
            ax.plot(epochs, train, color='#6aa8ff', label=f'Treino ({n_train:,})')
            if len(val):
                ax.plot(epochs, val, '--', color='#34d3bd', label=f'Validação ({n_val:,})')
            ax.set(title=title, xlabel='Época')
            ax.title.set_color('#e7ecf3')
            ax.xaxis.label.set_color('#c2cbd8')
            ax.grid(alpha=.15)
            ax.legend()
        fig.suptitle(f'OlhoCidadão · NumPy {self.n_input} → {self.n_hidden} tanh → 2 · '
                     f'parada na época {self.best_epoch}\nValidação temporal; teste final fora do treinamento', color='#e7ecf3')
        fig.tight_layout()
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)

    def logits(self, X):
        self.forward(X)
        return self.z2[:, 1] - self.z2[:, 0]

    def predict(self, X):
        from src.model_evaluation import calibrate
        probabilities = calibrate(self.logits(X), self.calibration)
        return (probabilities >= self.decision_threshold).astype(int), probabilities
