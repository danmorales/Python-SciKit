import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors

#Processo de classificação do vizinho mais próximo (NNC)

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

#Passo da malha
h = 0.02

#Criando mapa de cores
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#número de vizinhos
n_vizinhos = 15

#definindo pesos
peso1 = 'uniform'
peso2 = 'distance'

#Criando objeto de classificação de vizinhos para cada peso da sépala
clf_s1 = neighbors.KNeighborsClassifier(n_vizinhos, weights=peso1)
clf_s1.fit(Xs, Y)

clf_s2 = neighbors.KNeighborsClassifier(n_vizinhos, weights=peso2)
clf_s2.fit(Xs, Y)

#Criando objeto de classificação de vizinhos para cada peso da pétala
clf_p1 = neighbors.KNeighborsClassifier(n_vizinhos, weights=peso1)
clf_p1.fit(Xp, Y)

clf_p2 = neighbors.KNeighborsClassifier(n_vizinhos, weights=peso2)
clf_p2.fit(Xp, Y)

#Graficando fronteira de decisão

x_min_s = Xs[:, 0].min() - 1
x_max_s = Xs[:, 0].max() + 1

y_min_s = Xs[:, 1].min() - 1
y_max_s = Xs[:, 1].max() + 1

x_min_p = Xp[:, 0].min() - 1
x_max_p = Xp[:, 0].max() + 1

y_min_p = Xp[:, 1].min() - 1
y_max_p = Xp[:, 1].max() + 1

xx_s, yy_s = np.meshgrid(np.arange(x_min_s, x_max_s, h), np.arange(y_min_s, y_max_s, h))
xx_p, yy_p = np.meshgrid(np.arange(x_min_p, x_max_p, h), np.arange(y_min_p, y_max_p, h))

#Colocando os resultados da sépala num arranjo de cores
Zs1 = clf_s1.predict(np.c_[xx_s.ravel(), yy_s.ravel()])
Zs2 = clf_s2.predict(np.c_[xx_s.ravel(), yy_s.ravel()])

Zp1 = clf_p1.predict(np.c_[xx_p.ravel(), yy_p.ravel()])
Zp2 = clf_p2.predict(np.c_[xx_p.ravel(), yy_p.ravel()])

#Colocando os resultados num gráfico de cores
Zs1 = Zs1.reshape(xx_s.shape)
Zs2 = Zs2.reshape(xx_s.shape)

Zp1 = Zp1.reshape(xx_p.shape)
Zp2 = Zp2.reshape(xx_p.shape)

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.pcolormesh(xx_s, yy_s, Zs1, cmap=cmap_light)

#Inserindo os pontos de treinamento
plt.scatter(Xs[:, 0], Xs[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx_s.min(), xx_s.max())
plt.ylim(yy_s.min(), yy_s.max())
plt.title("Classificação das 3 classes (k = %i, peso = '%s')" % (n_vizinhos, peso1))

plt.subplot(1,2,2)
plt.pcolormesh(xx_s, yy_s, Zs2, cmap=cmap_light)

#Inserindo os pontos de treinamento
plt.scatter(Xs[:, 0], Xs[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx_s.min(), xx_s.max())
plt.ylim(yy_s.min(), yy_s.max())
plt.title("Classificação das 3 classes (k = %i, peso = '%s')" % (n_vizinhos, peso2))
plt.tight_layout()
plt.show()
savefig("Figs/NNC-Sepala.png",dpi=100)

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.pcolormesh(xx_p, yy_p, Zp1, cmap=cmap_light)

#Inserindo os pontos de treinamento
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx_p.min(), xx_p.max())
plt.ylim(yy_p.min(), yy_p.max())
plt.title("Classificação das 3 classes (k = %i, peso = '%s')" % (n_vizinhos, peso1))

plt.subplot(1,2,2)
plt.pcolormesh(xx_p, yy_p, Zp2, cmap=cmap_light)

#Inserindo os pontos de treinamento
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx_p.min(), xx_p.max())
plt.ylim(yy_p.min(), yy_p.max())
plt.title("Classificação das 3 classes (k = %i, peso = '%s')" % (n_vizinhos, peso2))
plt.tight_layout()
plt.show()
savefig("Figs/NNC-Petala.png",dpi=100)