import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
    y_train_oh = to_categorical(y_train, 10)
    y_test_oh = to_categorical(y_test, 10)
    return x_train, y_train, y_train_oh, x_test, y_test, y_test_oh


# ============================================
# Feature Extraction
# ============================================
def extract_features(x):
    return x


# ============================================
# Neural Network FROM SCRATCH
# ============================================
class NeuralNetworkFromScratch:
    def __init__(self, layers, activation, lr):
        self.layers = layers
        self.lr = lr
        self.activation_name = activation
        self.batch_size = 64

        self.weights = [
            np.random.randn(layers[i], layers[i + 1]) * 0.01
            for i in range(len(layers) - 1)
        ]
        self.biases = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]

    def activation(self, x):
        if self.activation_name == "relu":
            return np.maximum(0, x)
        if self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        return np.tanh(x)

    def activation_deriv(self, x):
        if self.activation_name == "relu":
            return (x > 0).astype(float)
        if self.activation_name == "sigmoid":
            s = self.activation(x)
            return s * (1 - s)
        return 1 - np.tanh(x) ** 2

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x):
        self.z = []
        self.a = [x]
        for i in range(len(self.weights) - 1):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            self.z.append(z)
            self.a.append(self.activation(z))
        z = self.a[-1] @ self.weights[-1] + self.biases[-1]
        self.z.append(z)
        self.a.append(self.softmax(z))
        return self.a[-1]

    def backward(self, y):
        dz = self.a[-1] - y
        for i in reversed(range(len(self.weights))):
            dw = self.a[i].T @ dz / self.a[i].shape[0]
            db = np.sum(dz, axis=0, keepdims=True) / self.a[i].shape[0]
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db
            if i != 0:
                dz = (dz @ self.weights[i].T) * self.activation_deriv(self.z[i - 1])

    def train(self, x, y, epochs=6):
        loss_hist, acc_hist = [], []
        n_samples = x.shape[0]

        for epoch in range(epochs):

            idx = np.random.permutation(n_samples)
            x_shuffled = x[idx]
            y_shuffled = y[idx]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                xb = x_shuffled[start:end]
                yb = y_shuffled[start:end]
                self.forward(xb)
                self.backward(yb)

            out = self.forward(x)
            loss = -np.mean(np.sum(y * np.log(out + 1e-8), axis=1))
            preds = np.argmax(out, axis=1)
            acc = np.mean(preds == np.argmax(y, axis=1))
            loss_hist.append(loss)
            acc_hist.append(acc)

        return loss_hist, acc_hist

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)


# ============================================
# TensorFlow Model Builder (full hyperparam range)
# ============================================
def build_tf_model(input_dim, params):
    model = Sequential()
    for i in range(params["hidden_layers"]):
        neurons = params["neurons"]
        act = params["activation"]
        if i == 0:
            model.add(Dense(neurons, activation=act, input_shape=(input_dim,)))
        else:
            model.add(Dense(neurons, activation=act))
    model.add(Dense(10, activation="softmax"))
    opt_dict = {
        "SGD": SGD(params["lr"]),
        "Adam": Adam(params["lr"]),
        "RMSProp": RMSprop(params["lr"]),
        "Adagrad": Adagrad(params["lr"]),
    }
    model.compile(
        optimizer=opt_dict[params["optimizer"]],
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================
# Random Search Optimization
# ============================================
def random_search(x, y, x_val, y_val, trials=3):
    best_acc = 0
    best_params = None
    for _ in range(trials):
        params = {
            "hidden_layers": random.randint(1, 5),
            "neurons": random.randint(32, 512),
            "activation": random.choice(["relu", "tanh", "sigmoid"]),
            "lr": 10 ** random.uniform(-5, -1),
            "batch": random.choice([16, 32, 64, 128]),
            "optimizer": random.choice(["SGD", "Adam", "RMSProp", "Adagrad"]),
            "epochs": random.randint(3, 20),
        }
        model = build_tf_model(x.shape[1], params)
        hist = model.fit(
            x,
            y,
            epochs=params["epochs"],
            batch_size=params["batch"],
            validation_data=(x_val, y_val),
            verbose=0,
        )
        acc = max(hist.history["val_accuracy"])
        if acc > best_acc:
            best_acc = acc
            best_params = params
    return best_params, best_acc


# ============================================
# Genetic Algorithm Optimization
# ============================================
def genetic_algorithm(x, y, x_val, y_val, pop_size=3, generations=2):
    population = []
    for _ in range(pop_size):
        population.append(
            {
                "hidden_layers": random.randint(1, 5),
                "neurons": random.randint(32, 512),
                "activation": random.choice(["relu", "tanh", "sigmoid"]),
                "lr": 10 ** random.uniform(-5, -1),
                "batch": random.choice([16, 32, 64, 128]),
                "optimizer": random.choice(["SGD", "Adam", "RMSProp", "Adagrad"]),
                "epochs": random.randint(3, 20),
            }
        )
    best = None
    best_acc = 0
    for _ in range(generations):
        scores = []
        for p in population:
            model = build_tf_model(x.shape[1], p)
            hist = model.fit(
                x,
                y,
                epochs=p["epochs"],
                batch_size=p["batch"],
                validation_data=(x_val, y_val),
                verbose=0,
            )
            acc = max(hist.history["val_accuracy"])
            scores.append((acc, p))
            if acc > best_acc:
                best_acc = acc
                best = p
        scores.sort(reverse=True, key=lambda x: x[0])

        population = [scores[0][1], scores[1][1]]

        while len(population) < pop_size:
            parent = random.choice(population)
            child = parent.copy()
            child["neurons"] = random.randint(32, 512)
            population.append(child)
    return best, best_acc


# ============================================
# Plot Results
# ============================================
def plot_results(loss, acc):
    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Loss")
    plt.plot(acc, label="Accuracy")
    plt.title("From Scratch NN Training")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


# ============================================
# Main
# ============================================
def main():
    x_train, y_train, y_train_oh, x_test, y_test, y_test_oh = load_and_preprocess_data()
    x_train = extract_features(x_train)
    x_test = extract_features(x_test)
    x_val = x_train[:10000]
    y_val = y_train_oh[:10000]
    x_train2 = x_train[10000:]
    y_train2 = y_train_oh[10000:]

    print("\n================ From Scratch NN ================")

    nn = NeuralNetworkFromScratch([784, 128, 64, 10], "relu", 0.01)
    loss, acc = nn.train(x_train2, y_train2, epochs=6)
    plot_results(loss, acc)
    preds = nn.predict(x_test)
    acc_scratch = np.mean(preds == y_test)
    print(f"Scratch NN Accuracy: {acc_scratch:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    print("\n================ Random Search TF ================")
    rs_params, rs_acc = random_search(x_train2, y_train2, x_val, y_val)
    print("Best Hyperparameters:", rs_params)
    print(f"Validation Accuracy: {rs_acc:.4f}")

    print("\n================ Genetic Algorithm TF ================")
    ga_params, ga_acc = genetic_algorithm(x_train2, y_train2, x_val, y_val)
    print("Best Hyperparameters:", ga_params)
    print(f"Validation Accuracy: {ga_acc:.4f}")

    print("\n================ SUMMARY COMPARISON ================")
    print(f"{'Model':<25} {'Accuracy':<10}")
    print(f"{'From Scratch NN':<25} {acc_scratch:<10.4f}")
    print(f"{'Random Search TF':<25} {rs_acc:<10.4f}")
    print(f"{'Genetic Algorithm TF':<25} {ga_acc:<10.4f}")
    print("===================================================")


if __name__ == "__main__":
    main()
