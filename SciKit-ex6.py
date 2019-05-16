import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

#Processo de classificação Gaussiana

arquivo = "CSV/Dados_Iris.csv"

colunas = ['sepala-comprimento','sepala-largura','petala-comprimento','petala-largura','classe']

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

#tamanho do passo na malha
h = 0.02 

#definindo o Kernel isotrópico
kernel = 1.0 * RBF([1.0])
gpc_isotropico_s = GaussianProcessClassifier(kernel=kernel).fit(Xs, Y)
gpc_isotropico_p = GaussianProcessClassifier(kernel=kernel).fit(Xp, Y)

#definindo o Kernel anisotrópico
kernel = 1.0 * RBF([1.0, 1.0])
gpc_anisotropico_s = GaussianProcessClassifier(kernel=kernel).fit(Xs, Y)
gpc_anisotropico_p = GaussianProcessClassifier(kernel=kernel).fit(Xp, Y)

#Criando a malha para graficar
x_min_s = Xs[:, 0].min() - 1
x_max_s = Xs[:, 0].max() + 1

x_min_p = Xp[:, 0].min() - 1
x_max_p = Xp[:, 0].max() + 1

y_min_s = Xs[:, 1].min() - 1
y_max_s = Xs[:, 1].max() + 1

y_min_p = Xp[:, 1].min() - 1
y_max_p = Xp[:, 1].max() + 1

xx_s, yy_s = np.meshgrid(np.arange(x_min_s, x_max_s, h), np.arange(y_min_s, y_max_s, h))
                     
xx_p, yy_p = np.meshgrid(np.arange(x_min_p, x_max_p, h), np.arange(y_min_p, y_max_p, h))

titulos = ["RBF isotrópica", "RBF anisotrópica"]

plt.figure(figsize=(10, 5))

for i, clf in enumerate((gpc_isotropico_s, gpc_anisotropico_s)):
	#Graficando as probabilidade previstas e definindo uma cor para cada ponto na malha
    
    plt.subplot(1, 2, i + 1)

    Z = clf.predict_proba(np.c_[xx_s.ravel(), yy_s.ravel()])

    #Colocando os resultados no gráfico
    Z = Z.reshape((xx_s.shape[0], xx_s.shape[1], 3))
    plt.imshow(Z, extent=(x_min_s, x_max_s, y_min_s, y_max_s), origin="lower")

    # Graficando os pontos de treinamento
    plt.scatter(Xs[:, 0], Xs[:, 1], c=np.array(["r", "g", "b"])[Y], edgecolors=(0, 0, 0))
    plt.xlabel('Comprimento Sépala')
    plt.ylabel('Largura Sépala')
    plt.xlim(xx_s.min(), xx_s.max())
    plt.ylim(yy_s.min(), yy_s.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("%s, LML: %.3f" % (titulos[i], clf.log_marginal_likelihood(clf.kernel_.theta)))

plt.tight_layout()
plt.show()
savefig("Figs/Iris-GPC-Sepala.png",dpi=100)

plt.figure(figsize=(10, 5))

for i, clf in enumerate((gpc_isotropico_p, gpc_anisotropico_p)):
	#Graficando as probabilidade previstas e definindo uma cor para cada ponto na malha
    
    plt.subplot(1, 2, i + 1)

    Z = clf.predict_proba(np.c_[xx_p.ravel(), yy_p.ravel()])

    #Colocando os resultados no gráfico
    Z = Z.reshape((xx_p.shape[0], xx_p.shape[1], 3))
    plt.imshow(Z, extent=(x_min_p, x_max_p, y_min_p, y_max_p), origin="lower")

    # Graficando os pontos de treinamento
    plt.scatter(Xp[:, 0], Xp[:, 1], c=np.array(["r", "g", "b"])[Y], edgecolors=(0, 0, 0))
    plt.xlabel('Comprimento Pétala')
    plt.ylabel('Largura Pétala')
    plt.xlim(xx_p.min(), xx_p.max())
    plt.ylim(yy_p.min(), yy_p.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("%s, LML: %.3f" % (titulos[i], clf.log_marginal_likelihood(clf.kernel_.theta)))

plt.tight_layout()
plt.show()
savefig("Figs/Iris-GPC-Petala.png",dpi=100)