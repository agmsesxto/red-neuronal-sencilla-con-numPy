import numpy as np

np.set_printoptions(
    threshold=np.inf,
    suppress=True,
)

# ---- Definir red y cargar pesos ----
model = np.load("model.npy", allow_pickle=True).item()
W1 = model["W1"]
B1 = model["B1"]
W2 = model["W2"]
B2 = model["B2"]
W3 = model["W3"]
B3 = model["B3"]


# ---- Funciones de acivación ----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def predict(x):
    z1 = x @ W1 + B1
    a1 = tanh(z1)
    z2 = a1 @ W2 + B2
    a2 = tanh(z2)
    z3 = a2 @ W3 + B3
    a3 = sigmoid(z3)
    return(a3)

X = np.array([[0, 0]])

while 1 == 1:
    X[0, 0] = int(input("Primer numero: "))
    X[0, 1] = int(input("Segundo numero: "))
    print(predict(X))
