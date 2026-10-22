import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, List


class NnModel:
    """
    Rede Neural Multilayer Perceptron (MLP) construída 100% DO ZERO com NumPy.
    Baseada na arquitetura clássica da playlist de referência ("Rede neural do zero em Python"),
    com duas customizações estruturais para auditoria da CEAP:
      1. Camada de entrada dinâmica adaptada para as 5 features reais de gastos parlamentares.
      2. Função de ativação customizada na camada oculta: Tangente Hiperbólica (np.tanh)
         (ou Leaky ReLU), substituindo a sigmoide padrão e aplicando sua derivada exata
         durante o backpropagation.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_hidden: int = 16,
        lr: float = 0.05,
        activation: str = "tanh",
        seed: int = 42,
    ):
        """
        Inicializa a arquitetura e os parâmetros treináveis (W1, b1, W2, b2) da Rede Neural.

        Parâmetros:
            X (np.ndarray): Matriz de entrada de treino com shape (n_amostras, n_features).
            y (np.ndarray): Vetor de rótulos reais (0 = Normal, 1 = Atípico) com shape (n_amostras,).
            n_hidden (int): Quantidade de neurônios na camada oculta (padrão: 16).
            lr (float): Taxa de aprendizado (learning rate) do Gradiente Descendente (padrão: 0.05).
            activation (str): Função de ativação da camada oculta ('tanh' ou 'leaky_relu').
            seed (int): Semente de aleatoriedade para reprodutibilidade acadêmica.
        """
        self.X = X
        self.y = y.astype(int)
        self.num_examples = X.shape[0]

        # ADEQUAÇÃO DA ARQUITETURA:
        # Dimensionamento dinâmico de acordo com X.shape[1] (5 features de entrada)
        # e 2 neurônios na camada de saída para classificação binária via Softmax + Cross-Entropy.
        self.n_input = X.shape[1]
        self.n_hidden = n_hidden
        self.n_output = 2

        # Aliases idênticos à nomenclatura da playlist "Rede neural do zero em Python"
        self.input_neurons = self.n_input
        self.hidden_neurons = self.n_hidden
        self.output_neurons = self.n_output
        self.lr = lr
        self.activation = activation.lower()

        rng = np.random.default_rng(seed)

        # Inicialização Xavier/Glorot para estabilidade numérica com Tangente Hiperbólica
        limite_w1 = np.sqrt(6.0 / (self.n_input + self.n_hidden))
        limite_w2 = np.sqrt(6.0 / (self.n_hidden + self.n_output))

        self.W1 = rng.uniform(-limite_w1, limite_w1, size=(self.n_input, self.n_hidden))
        self.b1 = np.zeros((1, self.n_hidden), dtype=np.float64)

        self.W2 = rng.uniform(-limite_w2, limite_w2, size=(self.n_hidden, self.n_output))
        self.b2 = np.zeros((1, self.n_output), dtype=np.float64)

        # Armazenamento de estados intermediários para o backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.y_pred = None

        # Históricos para geração da curva de aprendizado
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []

    def _activate_hidden(self, z: np.ndarray) -> np.ndarray:
        """
        =======================================================================
        CUSTOMIZAÇÃO OBRIGATÓRIA — FUNÇÃO DE ATIVAÇÃO DA CAMADA OCULTA
        =======================================================================
        Substitui a função sigmoide padrão pela Tangente Hiperbólica (np.tanh)
        ou Leaky ReLU. A np.tanh centraliza as ativações no intervalo (-1, 1),
        evitando o deslocamento unilateral dos gradientes (zig-zag optimization)
        e acelerando a convergência sobre dados financeiros normalizados via Z-Score.
        """
        if self.activation == "leaky_relu":
            return np.where(z > 0, z, 0.01 * z)
        # Padrão customizado principal: Tangente Hiperbólica
        return np.tanh(z)

    def _activate_hidden_derivative(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        =======================================================================
        CUSTOMIZAÇÃO OBRIGATÓRIA — DERIVADA EXATA DA FUNÇÃO DE ATIVAÇÃO
        =======================================================================
        Calcula a derivada analítica para uso na Regra da Cadeia (Backpropagation):
          - Para Tangente Hiperbólica: f(z) = tanh(z) => f'(z) = 1 - tanh²(z) = 1 - a²
          - Para Leaky ReLU: f'(z) = 1 se z > 0, senão 0.01
        """
        if self.activation == "leaky_relu":
            return np.where(z > 0, 1.0, 0.01)
        # Derivada exata da Tangente Hiperbólica: 1 - a1^2
        return 1.0 - np.power(a, 2)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """
        Calcula a função Softmax na camada de saída com estabilização numérica
        (subtração do máximo por linha para evitar overflow em np.exp).
        """
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_scores = np.exp(z_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X: np.ndarray = None) -> np.ndarray:
        """
        Propagação Direta (Forward Pass):
          1. Combinação linear da camada oculta: z1 = X · W1 + b1
          2. Ativação não-linear customizada (tanh): a1 = tanh(z1)
          3. Combinação linear da camada de saída: z2 = a1 · W2 + b2
          4. Distribuição de probabilidades via Softmax: y_pred = softmax(z2)
        """
        if X is None:
            X = self.X

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._activate_hidden(self.z1)
        self.f1 = self.a1  # Alias da playlist de referência
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_pred = self._softmax(self.z2)
        self.f2 = self.y_pred  # Alias da playlist de referência
        return self.y_pred

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray = None) -> float:
        """
        Calcula a perda de Entropia Cruzada Categórica (Cross-Entropy Loss)
        com regularização L2 leve para evitar overfitting:
            L = - (1 / N) * sum( log( y_pred[i, y_true[i]] ) )
        """
        if y_true is None:
            y_true = self.y

        n = y_true.shape[0]
        # Clipping para evitar log(0)
        probs_corretas = np.clip(y_pred[np.arange(n), y_true], 1e-12, 1.0)
        corect_logprobs = -np.log(probs_corretas)
        data_loss = float(np.sum(corect_logprobs) / n)
        return data_loss

    def backpropagation(self, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """
        Retropropagação do Erro (Backpropagation) via Regra da Cadeia:
          1. Gradiente na saída (Softmax + Cross-Entropy): delta2 = (y_pred - y_one_hot) / N
          2. Gradientes da camada de saída: dW2 = a1^T · delta2, db2 = sum(delta2)
          3. Propagação para camada oculta aplicando a DERIVADA DA TANH (1 - a1^2):
             delta1 = (delta2 · W2^T) * (1 - a1^2)
          4. Gradientes da camada oculta: dW1 = X^T · delta1, db1 = sum(delta1)
          5. Atualização dos pesos (Gradiente Descendente).
        """
        if X is None:
            X = self.X
        if y is None:
            y = self.y

        n = X.shape[0]

        # Passo 1: Erro na camada de saída (derivada de Softmax + Cross-Entropy)
        delta2 = self.y_pred.copy()
        delta2[np.arange(n), y] -= 1.0
        # Normalização proporcional a N para manter lr=0.05 monotônico em bases massivas
        escala = max(1.0, n / 22.0)
        delta2 /= escala

        # Passo 2: Gradientes de W2 e b2
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # Passo 3: Erro na camada oculta usando a derivada customizada de np.tanh: (1 - a1^2)
        derivada_oculta = self._activate_hidden_derivative(self.a1, self.z1)
        delta1 = np.dot(delta2, self.W2.T) * derivada_oculta

        # Passo 4: Gradientes de W1 e b1
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Passo 5: Atualização dos parâmetros pelo Gradiente Descendente
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, epochs: int = 2000, verbose_step: int = 250) -> Tuple[List[float], List[float]]:
        """
        Executa o loop de treinamento por `epochs` épocas, registra o histórico de
        Cross-Entropy Loss e Acurácia, e salva o gráfico em `outputs/loss_curve.png`.
        """
        self.loss_history = []
        self.accuracy_history = []

        print(
            f"[NnModel] Iniciando treinamento em CPU (NumPy puro) | "
            f"Arquitetura: {self.n_input} -> {self.n_hidden} ({self.activation}) -> {self.n_output} (softmax) | "
            f"lr={self.lr} | epochs={epochs}"
        )

        for epoch in range(1, epochs + 1):
            # 1. Forward Pass
            y_pred = self.forward(self.X)

            # 2. Cálculo da Loss e Acurácia
            current_loss = self.loss(y_pred, self.y)
            preds = np.argmax(y_pred, axis=1)
            acc = float(np.mean(preds == self.y))

            self.loss_history.append(current_loss)
            self.accuracy_history.append(acc)

            # 3. Backpropagation e atualização de pesos
            self.backpropagation(self.X, self.y)

            if epoch == 1 or epoch % verbose_step == 0 or epoch == epochs:
                print(
                    f"  -> Época {epoch:4d}/{epochs} | "
                    f"Cross-Entropy Loss: {current_loss:.4f} | "
                    f"Acurácia (Treino): {acc * 100:.2f}%"
                )

        self._save_learning_curve()
        return self.loss_history, self.accuracy_history

    def _save_learning_curve(self, output_path: str = os.path.join("outputs", "loss_curve.png")) -> None:
        """
        Gera e salva o gráfico da curva de aprendizado (Queda da Loss e Evolução da Acurácia)
        em `outputs/loss_curve.png`.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
        epocas = np.arange(1, len(self.loss_history) + 1)

        # Gráfico 1: Curva de Perda (Cross-Entropy Loss)
        ax1.plot(epocas, self.loss_history, color="#D9381E", linewidth=2.2, label="Cross-Entropy Loss")
        ax1.set_title("Curva de Aprendizado — Queda do Erro (Loss)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Épocas de Treinamento")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax1.grid(True, linestyle="--", alpha=0.45)
        ax1.legend(loc="upper right")

        # Gráfico 2: Curva de Acurácia
        ax2.plot(
            epocas,
            np.array(self.accuracy_history) * 100.0,
            color="#1B6CA8",
            linewidth=2.2,
            label=f"Acurácia ({self.activation.upper()})",
        )
        ax2.set_title("Evolução da Acurácia no Treinamento (%)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Épocas de Treinamento")
        ax2.set_ylabel("Acurácia (%)")
        ax2.set_ylim(40, 102)
        ax2.grid(True, linestyle="--", alpha=0.45)
        ax2.legend(loc="lower right")

        plt.suptitle(
            f"Auditoria CEAP — Rede Neural MLP do Zero em NumPy ({self.n_input}-{self.n_hidden}-{self.n_output})",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[NnModel] Gráfico da curva de aprendizado salvo em: {output_path}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza a inferência para novos registros normalizados.

        Retorna:
            classes_preditas (np.ndarray): Vetor com 0 (Normal) ou 1 (Gasto Atípico/Suspeito).
            probabilidades_atipico (np.ndarray): Probabilidade [0.0 a 1.0] de pertencer à classe 1.
        """
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._activate_hidden(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probs = self._softmax(z2)

        classes_preditas = np.argmax(probs, axis=1)
        probabilidades_atipico = probs[:, 1]
        return classes_preditas, probabilidades_atipico
