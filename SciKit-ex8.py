import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier

#Processo de classificação da superficie de decisão da multi-classe 

arquivo = "CSV/Dados_Iris.csv"

colunas = ['sepala-comprimento','sepala-largura','petala-comprimento','petala-largura','classe']

nomes = ['setosa','versicolor','virginica']

dados = pd.read_csv(arquivo, names=colunas)

#convertendo a classe de strings para valores numéricos
classe = dados['classe']
encoder = LabelEncoder()
encoder.fit(classe)
encoded_classe = encoder.transform(classe)

#armazenando os dados da sépala na variável X como um arranjo NumPy
Xs = np.array(dados[['sepala-comprimento','sepala-largura']])

#armazenando os dados da pétala na variável X como um arranjo NumPy
Xp = np.array(dados[['petala-comprimento','petala-largura']])

#armazenando os dados da classe convertida num arranjo NumPy
Y = np.array(encoded_classe,dtype=int)

cores = "bry"

#misturando
idx_s = np.arange(Xs.shape[0])
idx_p = np.arange(Xp.shape[0])

np.random.seed(13)

np.random.shuffle(idx_s)
Xs = Xs[idx_s]
Y = Y[idx_s]

np.random.shuffle(idx_p)
Xp = Xp[idx_p]

#Padronizando sépala
media_s = Xs.mean(axis=0)
desvio_padrao_s = Xs.std(axis=0)
Xs = (Xs - media_s) / desvio_padrao_s

#Padronizando pétala
media_p = Xp.mean(axis=0)
desvio_padrao_p = Xp.std(axis=0)
Xp = (Xp - media_p) / desvio_padrao_p

#Passo da malha
h = 0.02 

clf_s = SGDClassifier(alpha=0.001, max_iter=100)
clf_s.fit(Xs, Y)

clf_p = SGDClassifier(alpha=0.001, max_iter=100)
clf_p.fit(Xp, Y)

#Criando a malha
x_min_s = Xs[:, 0].min() - 1
x_max_s = Xs[:, 0].max() + 1

x_min_p = Xp[:, 0].min() - 1
x_max_p = Xp[:, 0].max() + 1

y_min_s = Xs[:, 1].min() - 1
y_max_s = Xs[:, 1].max() + 1

y_min_p = Xp[:, 1].min() - 1
y_max_p = Xp[:, 1].max() + 1

xx_s, yy_s = np.meshgrid(np.arange(x_min_s, x_max_s, h),np.arange(y_min_s, y_max_s, h))
xx_p, yy_p = np.meshgrid(np.arange(x_min_p, x_max_p, h),np.arange(y_min_p, y_max_p, h))

#Graficando a fronteira de decisão
Zs = clf_s.predict(np.c_[xx_s.ravel(), yy_s.ravel()])
Zp = clf_p.predict(np.c_[xx_p.ravel(), yy_p.ravel()])

Zs = Zs.reshape(xx_s.shape)
Zp = Zp.reshape(xx_p.shape)

plt.figure(1, figsize=(8, 6))

cs_s = plt.contourf(xx_s, yy_s, Zs, cmap=plt.cm.Paired)
plt.axis('tight')

#Graficando os pontos de treino
for i, color in zip(clf_s.classes_, cores):
    idx_s = np.where(Y == i)
    plt.scatter(Xs[idx_s, 0], Xs[idx_s, 1], c=color, label=nomes[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
plt.title("Superfície de decisão da multi-classe SGD Sépala")
plt.axis('tight')

#Graficando as três classes num classificador
xmin_s, xmax_s = plt.xlim()
ymin_s, ymax_s = plt.ylim()

coef_s = clf_s.coef_
intercept_s = clf_s.intercept_

#Função para graficar um hiper-plano
def plot_hyperplano(c, cor):
    def linha(x0):
        return (-(x0 * coef_s[c, 0]) - intercept_s[c]) / coef_s[c, 1]

    plt.plot([xmin_s, xmax_s], [linha(xmin_s), linha(xmax_s)],
             ls="--", color=cor)
             
for i, color in zip(clf_s.classes_, cores):
    plot_hyperplano(i, color)
plt.legend()
plt.show()
savefig("Figs/SGD-Sepala.png",dpi=100)

plt.figure(2, figsize=(8, 6))

cs_p = plt.contourf(xx_p, yy_p, Zp, cmap=plt.cm.Paired)
plt.axis('tight')

#Graficando os pontos de treino
for i, color in zip(clf_p.classes_, cores):
    idx_p = np.where(Y == i)
    plt.scatter(Xp[idx_p, 0], Xp[idx_p, 1], c=color, label=nomes[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
plt.title("Superfície de decisão da multi-classe SGD Pétala")
plt.axis('tight')

#Graficando as três classes num classificador
xmin_p, xmax_p = plt.xlim()
ymin_p, ymax_p = plt.ylim()

coef_p = clf_p.coef_
intercept_p = clf_p.intercept_

#Função para graficar um hiper-plano
def plot_hyperplano2(c, cor):
    def linha2(x0):
        return (-(x0 * coef_p[c, 0]) - intercept_p[c]) / coef_p[c, 1]

    plt.plot([xmin_p, xmax_p], [linha2(xmin_p), linha2(xmax_p)],
             ls="--", color=cor)
             
for i, color in zip(clf_p.classes_, cores):
    plot_hyperplano2(i, color)
plt.legend()
plt.show()
savefig("Figs/SGD-Petala.png",dpi=100)