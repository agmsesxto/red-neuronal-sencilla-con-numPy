import numpy as np

np.set_printoptions(
    threshold=np.inf,
    suppress=True,
)

i = int(0)

# ---- Funciones de activación ----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - x**2

def predict(x):
    z1 = x @ W1 + B1
    a1 = tanh(z1)
    z2 = a1 @ W2 + B2
    a2 = tanh(z2)
    z3 = a2 @ W3 + B3
    a3 = sigmoid(z3)
    return(a3)

# ---- Datos de entrenamiento ----
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# ---- Definir red ----
np.random.seed(42)

W1 = np.random.randn(2, 10)
B1 = np.zeros((1, 10))
W2 = np.random.randn(10, 10)
B2 = np.zeros((1, 10))
W3 = np.random.randn(10, 1)
B3 = np.zeros((1, 1))

# ---- Entrenamiento ----
lr = 0.1
epochs = 10000

while i <= epochs:
    # ---- Forward pass ----
    z1 = X @ W1 + B1
    a1 = tanh(z1)

    z2 = a1 @ W2 + B2
    a2 = tanh(z2)

    z3 = a2 @ W3 + B3
    a3 = sigmoid(z3) # salida final

    # ---- Backpropagation ----
    error = y - a3
    d3 = error * sigmoid_deriv(a3)

    error2 = d3 @ W3.T
    d2 = error2 * tanh_deriv(a2)

    error1 = d2 @ W2.T
    d1 = error1 * tanh_deriv(a1)

    # ---- Actualización de los pesos ----
    W3 += a2.T @ d3 * lr
    W2 += a1.T @ d2 * lr
    W1 += X.T @ d1 * lr
	# ---- Actualización de los bias ----
    B3 = np.sum(d3, axis=0, keepdims=True) * lr
    B2 = np.sum(d2, axis=0, keepdims=True) * lr
    B1 = np.sum(d1, axis=0, keepdims=True) * lr

    porcentaje = (i * 100) / epochs
    if porcentaje.is_integer():
        print(int(porcentaje))
    i = i + 1
i = 0

print("\nResultados:\n")
print(predict(X))

# ---- Guardar modelo entrenado ----
input("\nPresiona enter para guardar.\n")
np.save("model.npy", { "W1": W1, "B1": B1, "W2": W2, "B2": B2, "W3": W3, "B3": B3})
