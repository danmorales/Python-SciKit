import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

#Processo de classificação de três classe utilizando Regressão logística

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

#definindo o objeto de regressão logistica para sépala
logreg_s = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

#definindo o objeto de regressão logistica para pétala
logreg_p = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

#ajustando os dados da sépala
logreg_s.fit(Xs, Y)

#ajustando os dados da pétala
logreg_p.fit(Xp, Y)

#Graficando a fronteira de decisão onde cada ponto terá um cor diferente na malha
x_min_s = Xs[:, 0].min() - 0.5
x_max_s = Xs[:, 0].max() + 0.5
y_min_s = Xs[:, 1].min() - 0.5
y_max_s = Xs[:, 1].max() + 0.5

x_min_p = Xp[:, 0].min() - 0.5
x_max_p = Xp[:, 0].max() + 0.5
y_min_p = Xp[:, 1].min() - 0.5
y_max_p = Xp[:, 1].max() + 0.5

#Tamanho do passo na malha
h = 0.02

xx_s, yy_s = np.meshgrid(np.arange(x_min_s, x_max_s, h), np.arange(y_min_s, y_max_s, h))
xx_p, yy_p = np.meshgrid(np.arange(x_min_p, x_max_p, h), np.arange(y_min_p, y_max_p, h))

Zs = logreg_s.predict(np.c_[xx_s.ravel(), yy_s.ravel()])
Zp = logreg_p.predict(np.c_[xx_p.ravel(), yy_p.ravel()])

# Inserindo os resultados da sépala num gráfico de cores
Zs = Zs.reshape(xx_s.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx_s, yy_s, Zs, cmap=plt.cm.Paired)

# Graficando os pontos de treinamento
plt.scatter(Xs[:, 0], Xs[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Comprimento sépala')
plt.ylabel('Largura sépala')
plt.xlim(xx_s.min(), xx_s.max())
plt.ylim(yy_s.min(), yy_s.max())
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()
savefig("Figs/Iris-LogReg3Classe-Sepala.png",dpi=100)

# Inserindo os resultados da pétala num gráfico de cores
Zp = Zp.reshape(xx_p.shape)
plt.figure(2, figsize=(4, 3))
plt.pcolormesh(xx_p, yy_p, Zp, cmap=plt.cm.Paired)

# Graficando os pontos de treinamento
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Comprimento pétala')
plt.ylabel('Largura pétala')
plt.xlim(xx_p.min(), xx_p.max())
plt.ylim(yy_p.min(), yy_p.max())
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()
savefig("Figs/Iris-LogReg3Classe-Petala.png",dpi=100)
