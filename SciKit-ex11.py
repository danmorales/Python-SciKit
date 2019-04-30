import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

def my_kernel(X, Y):
    M = np.array([[2, 0], [0, 1.0]])
    K1 = np.dot(X, M)
    KM = np.dot(K1, Y.T)
    return KM
 
#SVM com kernel definido pelo usuário 
   
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

#Criando o objeto de Support Vector Machine
#Sépala
clf_s = svm.SVC(kernel=my_kernel)
#Pétala
clf_p = svm.SVC(kernel=my_kernel)

#ajustando os dados
#Sépala
clf_s.fit(Xs, Y)
#Pétala
clf_p.fit(Xp, Y)

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

Zs = clf_s.predict(np.c_[xx_s.ravel(), yy_s.ravel()])
Zp = clf_p.predict(np.c_[xx_p.ravel(), yy_p.ravel()])

Zs = Zs.reshape(xx_s.shape)
Zp = Zp.reshape(xx_p.shape)

#Colocando os resultados num gráfico de cores
#Sépala
plt.figure(figsize=(10, 6))
plt.pcolormesh(xx_s, yy_s, Zs, cmap=plt.cm.Paired)

#Graficando os pontos de treinamento
plt.scatter(Xs[:, 0], Xs[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('Classificação das três classes com Kernel definido pelo usuário (Sépala)')
plt.axis('tight')
plt.tight_layout()
plt.show()
savefig("Figs/SVM-Sepala-CustomKernel.png",dpi=100)

#Pétala
plt.figure(figsize=(10, 6))
plt.pcolormesh(xx_p, yy_p, Zp, cmap=plt.cm.Paired)

#Graficando os pontos de treinamento
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('Classificação das três classe com Kernel definido pelo usuário (Pétala)')
plt.axis('tight')
plt.tight_layout()
plt.show()
savefig("Figs/SVM-Petala-CustomKernel.png",dpi=100)