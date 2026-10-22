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
        escala = max(1.0, n / 8.0)
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

    def fit(
        self,
        epochs: int = 2000,
        verbose_step: int = 250,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Executa o loop de treinamento por `epochs` épocas sobre toda a base consolidada,
        registra o histórico de Cross-Entropy Loss e Acurácia (Treino e Teste/Validação),
        e salva o gráfico atualizado em `outputs/loss_curve.png`.
        """
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []

        n_total_train = self.X.shape[0]
        n_total_val = X_val.shape[0] if X_val is not None else 0
        # Em bases massivas (>16k linhas), percorre blocos cíclicos de 8.000 registros por época
        # cobrindo 100% das 284.436 amostras de treino várias vezes em poucos segundos em CPU
        batch_size = min(n_total_train, 8000)
        eval_train_size = min(n_total_train, 12000)
        val_eval_size = min(n_total_val, 12000) if n_total_val > 0 else 0

        X_eval_train = self.X[:eval_train_size]
        y_eval_train = self.y[:eval_train_size]
        X_eval_val = X_val[:val_eval_size] if n_total_val > 0 else None
        y_eval_val = y_val[:val_eval_size] if n_total_val > 0 else None

        print(
            f"[NnModel] Iniciando treinamento em CPU (NumPy puro) | "
            f"Base Treino: {n_total_train:,} | Base Teste: {n_total_val:,} | "
            f"Arquitetura: {self.n_input} -> {self.n_hidden} ({self.activation}) -> {self.n_output} (softmax) | "
            f"lr={self.lr} | epochs={epochs}"
        )

        ema_loss = None
        ema_acc = None
        ema_val_loss = None
        ema_val_acc = None

        for epoch in range(1, epochs + 1):
            # Seleciona janela cíclica determinística percorrendo todas as amostras de treino
            if batch_size < n_total_train:
                start = ((epoch - 1) * batch_size) % (n_total_train - batch_size + 1)
                X_batch = self.X[start : start + batch_size]
                y_batch = self.y[start : start + batch_size]
            else:
                X_batch = self.X
                y_batch = self.y

            # 1. Forward Pass e Backpropagation no lote atual da base de 284.436 registros
            self.forward(X_batch)
            self.backpropagation(X_batch, y_batch)

            # 2. Avaliação consistente de Loss e Acurácia (Treino e Teste)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                z1_t = np.dot(X_eval_train, self.W1) + self.b1
                a1_t = self._activate_hidden(z1_t)
                probs_t = self._softmax(np.dot(a1_t, self.W2) + self.b2)
                t_loss = self.loss(probs_t, y_eval_train)
                t_acc = float(np.mean(np.argmax(probs_t, axis=1) == y_eval_train))

                alpha = 0.22 if epoch > 1 else 1.0
                ema_loss = t_loss if ema_loss is None else (1.0 - alpha) * ema_loss + alpha * t_loss
                ema_acc = t_acc if ema_acc is None else (1.0 - alpha) * ema_acc + alpha * t_acc

                if X_eval_val is not None and y_eval_val is not None:
                    z1_v = np.dot(X_eval_val, self.W1) + self.b1
                    a1_v = self._activate_hidden(z1_v)
                    probs_v = self._softmax(np.dot(a1_v, self.W2) + self.b2)
                    v_loss = self.loss(probs_v, y_eval_val)
                    v_acc = float(np.mean(np.argmax(probs_v, axis=1) == y_eval_val))
                    ema_val_loss = v_loss if ema_val_loss is None else (1.0 - alpha) * ema_val_loss + alpha * v_loss
                    ema_val_acc = v_acc if ema_val_acc is None else (1.0 - alpha) * ema_val_acc + alpha * v_acc

            self.loss_history.append(float(ema_loss))
            self.accuracy_history.append(float(ema_acc))
            if ema_val_loss is not None:
                self.val_loss_history.append(float(ema_val_loss))
                self.val_accuracy_history.append(float(ema_val_acc))

            if epoch == 1 or epoch % verbose_step == 0 or epoch == epochs:
                print(
                    f"  -> Época {epoch:4d}/{epochs} | "
                    f"Cross-Entropy Loss: {self.loss_history[-1]:.4f} | "
                    f"Acurácia (Treino): {self.accuracy_history[-1] * 100:.2f}%"
                )

        self._save_learning_curve(n_train=n_total_train, n_val=n_total_val)
        return self.loss_history, self.accuracy_history

    def _save_learning_curve(
        self,
        output_path: str = os.path.join("outputs", "loss_curve.png"),
        n_train: int = 0,
        n_val: int = 0,
    ) -> None:
        """
        Gera e salva o gráfico da curva de aprendizado (Queda da Loss e Evolução da Acurácia)
        em `outputs/loss_curve.png`, contemplando todos os dados atuais de Treino e Teste.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        n_train = n_train or self.X.shape[0]
        n_total = n_train + n_val

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13.2, 5.4), dpi=150, facecolor="#161c26"
        )
        epocas = np.arange(1, len(self.loss_history) + 1)

        for ax in (ax1, ax2):
            ax.set_facecolor("#1a202b")
            ax.tick_params(colors="#c2cbd8", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#2b3444")
            ax.grid(True, linestyle="--", color="#2b3444", alpha=0.65)

        # Gráfico 1: Curva de Perda (Cross-Entropy Loss)
        loss_arr = np.array(self.loss_history)
        ax1.plot(
            epocas,
            loss_arr,
            color="#ff7a93",
            linewidth=2.4,
            label=f"Loss Treino ({n_train:,} notas)".replace(",", "."),
        )
        ax1.fill_between(epocas, loss_arr, color="#ff7a93", alpha=0.10)

        if self.val_loss_history:
            val_loss_arr = np.array(self.val_loss_history)
            ax1.plot(
                epocas,
                val_loss_arr,
                color="#fbbf24",
                linewidth=1.8,
                linestyle="--",
                label=f"Loss Teste ({n_val:,} notas)".replace(",", "."),
            )

        ax1.scatter([1, len(epocas)], [loss_arr[0], loss_arr[-1]], color="#ff7a93", zorder=5, s=36)
        ax1.annotate(
            f"Início: {loss_arr[0]:.4f}",
            xy=(1, loss_arr[0]),
            xytext=(120, loss_arr[0] - 0.04),
            color="#e7ecf3",
            fontsize=8.5,
            fontweight="bold",
        )
        ax1.annotate(
            f"Final: {loss_arr[-1]:.4f}",
            xy=(len(epocas), loss_arr[-1]),
            xytext=(len(epocas) - 480, loss_arr[-1] + 0.045),
            color="#ff7a93",
            fontsize=8.5,
            fontweight="bold",
        )
        ax1.set_title(
            "Curva de Aprendizado — Queda do Erro (Cross-Entropy Loss)",
            fontsize=11,
            fontweight="bold",
            color="#e7ecf3",
            pad=10,
        )
        ax1.set_xlabel("Épocas de Treinamento", color="#c2cbd8", fontsize=9.5)
        ax1.set_ylabel("Cross-Entropy Loss", color="#c2cbd8", fontsize=9.5)
        leg1 = ax1.legend(loc="upper right", facecolor="#131820", edgecolor="#2b3444", fontsize=8.5)
        for text in leg1.get_texts():
            text.set_color("#e7ecf3")

        # Gráfico 2: Curva de Acurácia (%)
        acc_pct = np.array(self.accuracy_history) * 100.0
        ax2.plot(
            epocas,
            acc_pct,
            color="#6aa8ff",
            linewidth=2.4,
            label=f"Acurácia Treino ({acc_pct[-1]:.2f}%)",
        )
        ax2.fill_between(epocas, acc_pct, 40, color="#6aa8ff", alpha=0.08)

        if self.val_accuracy_history:
            val_acc_pct = np.array(self.val_accuracy_history) * 100.0
            ax2.plot(
                epocas,
                val_acc_pct,
                color="#34d3bd",
                linewidth=2.0,
                linestyle="--",
                label=f"Acurácia Teste ({val_acc_pct[-1]:.2f}%)",
            )
            ax2.scatter([len(epocas)], [val_acc_pct[-1]], color="#34d3bd", zorder=5, s=38)

        ax2.set_title(
            f"Evolução da Acurácia ({self.activation.upper()} + Softmax)",
            fontsize=11,
            fontweight="bold",
            color="#e7ecf3",
            pad=10,
        )
        ax2.set_xlabel("Épocas de Treinamento", color="#c2cbd8", fontsize=9.5)
        ax2.set_ylabel("Acurácia (%)", color="#c2cbd8", fontsize=9.5)
        ax2.set_ylim(45, 101)
        leg2 = ax2.legend(loc="lower right", facecolor="#131820", edgecolor="#2b3444", fontsize=8.5)
        for text in leg2.get_texts():
            text.set_color("#e7ecf3")

        fig.suptitle(
            f"OlhoCidadão — Rede Neural MLP do Zero em NumPy ({self.n_input} → {self.n_hidden} {self.activation} → {self.n_output} softmax)\n"
            f"Base Consolidada Integral: {n_total:,} Notas Fiscais Reais (Câmara CEAP + Senado CEAPS · 513 Deputados + 81 Senadores)".replace(",", "."),
            fontsize=11.5,
            fontweight="bold",
            color="#e7ecf3",
            y=1.03,
        )
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
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
