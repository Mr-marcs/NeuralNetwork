import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, lr=0.1, epochs=10**4, seed=42,tol=10**-6):
        self.lr = lr
        self.epochs = epochs
        self.initialize_parameters()
        self.tol = tol
        np.random.seed(seed)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.back_prop()

    def initialize_parameters(self):
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.b1 = np.random.randn()
        self.b2 = np.random.randn()
        self.variables = [self.w1, self.w2, self.b1, self.b2]

    def forward(self, target):
        a1 = self.sigmoid(self.w1 * target + self.b1)
        a2 = self.sigmoid(self.w2 * a1 + self.b2)
        return a2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grad_sig(self):
        a1 = self.sigmoid(self.w1 * self.X + self.b1)
        a2 = self.sigmoid(self.w2 * a1 + self.b2)

        delw2 = -2 * np.sum((self.y - a2) * a2 * (1 - a2) * a1)
        delb2 = -2 * np.sum((self.y - a2) * a2 * (1 - a2))

        delw1 = -2 * np.sum((self.y - a2) * a2 * (1 - a2) * self.w2 * a1 * (1 - a1) * self.X)
        delb1 = -2 * np.sum((self.y - a2) * a2 * (1 - a2) * self.w2 * a1 * (1 - a1))
        return delw1, delw2, delb1, delb2

    def back_prop(self):
        count = 0
        for _ in range(self.epochs):
            count += 1

            # Cálculo dos gradientes
            delw1, delw2, delb1, delb2 = self.grad_sig()

            # Atualização dos pesos e vieses
            self.w1 -= self.lr * delw1
            self.w2 -= self.lr * delw2
            self.b1 -= self.lr * delb1
            self.b2 -= self.lr * delb2
            MyGrad = [self.w1, self.w2, self.b1, self.b2]

            err = np.sum((np.array(self.variables) - np.array(MyGrad)) ** 2)
            if err < self.tol:
                break
            self.variables = MyGrad
        print(err)
        
    def predict_sig(self, value):
        return self.forward(value)

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self.forward(x)
            predictions.append(prediction)
        return predictions

    def plot_sigmoid(self):
        x = []
        y = []
        targets = np.linspace(-1, 1, 1000)
        for target in targets:
            x.append(target)
            y.append(self.forward(target))

        plt.plot(x, y)
        plt.title('Sigmoid da Equação')
        plt.xlabel('target')
        plt.ylabel('sigmoid(a6)')
        plt.grid(True)
        plt.show()

def PCA(X):
    X = X - X.mean()
    cov = np.cov(X.T) 
    eigval, eigvec = np.linalg.eig(cov)
    indexes = np.argsort(eigval)[::-1]
    eigvec_sorted = eigvec[:, indexes]
    componentes_principais = eigvec_sorted[:, :1]
    dados_projetados = np.dot(X, componentes_principais)  
    return dados_projetados

iris = load_iris()
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df["target"] = iris.target
df = df[df["target"].isin([0,1])]

df_f = df.drop(columns=["target"]) - df.drop(columns=["target"]).mean()
dp = PCA(df.drop(columns=["target"]))
df_dp = pd.DataFrame({'PC': dp[:, 0], 'target': df["target"]})
df_dp["PC"] = df_dp["PC"].apply(lambda x: x/10)

model3 = NeuralNetwork(lr=0.001,epochs=10**4,tol=10**-9)
model3.fit(df_dp["PC"],df_dp["target"])

print(model3.predict_sig(0.150379))