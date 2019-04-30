import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestCentroid

#Processo de classificação do centróide mais próximo (NCC)

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

#Definindo o passo da malha
h = 0.02

#Criando mapa de cores
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

shrinkage1 = None
shrinkage2 = 0.2

#Criando os objetos de classificação da sépala para cada peso
clf_s1 = NearestCentroid(shrink_threshold=shrinkage1)
clf_s2 = NearestCentroid(shrink_threshold=shrinkage2)

#Criando os objetos de classificação da pétala para cada peso
clf_p1 = NearestCentroid(shrink_threshold=shrinkage1)
clf_p2 = NearestCentroid(shrink_threshold=shrinkage2)

#Ajustando os dados para sépala
clf_s1.fit(Xs, Y)
y_pred_s1 = clf_s1.predict(Xs)

clf_s2.fit(Xs, Y)
y_pred_s2 = clf_s2.predict(Xs)

#Ajustando os dados para pétala
clf_p1.fit(Xp, Y)
y_pred_p1 = clf_p1.predict(Xp)

clf_p2.fit(Xp, Y)
y_pred_p2 = clf_p2.predict(Xp)

print("shrinkage sépala 1 = ",shrinkage1, " Média = ",np.mean(Y == y_pred_s1))
print("shrinkage sépala 2 = ",shrinkage2, " Média = ",np.mean(Y == y_pred_s2))

#Graficando os limites da fronteira de decisão
#Sépala
x_min_s = Xs[:, 0].min() - 1
x_max_s = Xs[:, 0].max() + 1
y_min_s = Xs[:, 1].min() - 1
y_max_s = Xs[:, 1].max() + 1
#Pétala
x_min_p = Xp[:, 0].min() - 1
x_max_p = Xp[:, 0].max() + 1
y_min_p = Xp[:, 1].min() - 1
y_max_p = Xp[:, 1].max() + 1

xx_s, yy_s = np.meshgrid(np.arange(x_min_s, x_max_s, h), np.arange(y_min_s, y_max_s, h))
xx_p, yy_p = np.meshgrid(np.arange(x_min_p, x_max_p, h), np.arange(y_min_p, y_max_p, h))

Zs1 = clf_s1.predict(np.c_[xx_s.ravel(), yy_s.ravel()])
Zs2 = clf_s2.predict(np.c_[xx_s.ravel(), yy_s.ravel()])

Zp1 = clf_p1.predict(np.c_[xx_p.ravel(), yy_p.ravel()])
Zp2 = clf_p2.predict(np.c_[xx_p.ravel(), yy_p.ravel()])

#Inserindo resultados num gráfico de cores
Zs1 = Zs1.reshape(xx_s.shape)
Zs2 = Zs2.reshape(xx_s.shape)

Zp1 = Zp1.reshape(xx_p.shape)
Zp2 = Zp2.reshape(xx_p.shape)

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.pcolormesh(xx_s, yy_s, Zs1, cmap=cmap_light)
#Graficando os pontos de treino
plt.scatter(Xs[:, 0], Xs[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)
plt.title("Classificação das 3 classes (Enconlhimento=%r)" % shrinkage1)
plt.axis('tight')

plt.subplot(1,2,2)
plt.pcolormesh(xx_s, yy_s, Zs2, cmap=cmap_light)
#Graficando os pontos de treino
plt.scatter(Xs[:, 0], Xs[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)
plt.title("Classificação das 3 classes (Enconlhimento=%r)" % shrinkage2)
plt.axis('tight')
plt.tight_layout()
plt.show()
savefig("Figs/NCC-Sepala.png",dpi=100)

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.pcolormesh(xx_p, yy_p, Zp1, cmap=cmap_light)
#Graficando os pontos de treino
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)
plt.title("Classificação das 3 classes (Enconlhimento=%r)" % shrinkage1)
plt.axis('tight')

plt.subplot(1,2,2)
plt.pcolormesh(xx_p, yy_p, Zp2, cmap=cmap_light)
#Graficando os pontos de treino
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)
plt.title("Classificação das 3 classes (Enconlhimento=%r)" % shrinkage2)
plt.axis('tight')
plt.tight_layout()
plt.show()
savefig("Figs/NCC-Petala.png",dpi=100)