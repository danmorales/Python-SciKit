import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Linear Discriminant Analysis 

arquivo = "CSV/Dados_Iris.csv"

colunas = ['sepala-comprimento','sepala-largura','petala-comprimento','petala-largura','classe']

dados = pd.read_csv(arquivo, names=colunas)

#convertendo a classe de strings para valores numéricos
classe = dados['classe']
encoder = LabelEncoder()
encoder.fit(classe)
encoded_classe = encoder.transform(classe)

#armazenando os dados da sépala na variável X
X = dados[['sepala-comprimento','sepala-largura','petala-comprimento','petala-largura']]
Y = encoded_classe

nomes = ['setosa','versicolor','virginica']

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, Y).transform(X)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, nomes in zip(colors, [0, 1, 2], nomes):
    plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], alpha=.8, color=color, label=nomes)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA do arranjo Iris')
plt.show()
savefig("Figs/Iris-LDA.png",dpi=100)

