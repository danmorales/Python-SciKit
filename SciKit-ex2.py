import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

arquivo = "CSV/Dados_Iris.csv"

colunas = ['sepala-comprimento','sepala-largura','petala-comprimento','petala-largura','classe']

dados = pd.read_csv(arquivo, names=colunas)

#separando as informações da sépala apenas em variáveis
sc = dados['sepala-comprimento']
sl = dados['sepala-largura']

#separando as informações da pétala apenas em variáveis
pc = dados['petala-comprimento']
pl = dados['petala-largura']

#convertendo a classe de strings para valores numéricos
classe = dados['classe']
encoder = LabelEncoder()
encoder.fit(classe)
encoded_classe = encoder.transform(classe)

print("\n")
print("Graficando os dados")
#graficando sépala e colorindo por classe
plt.figure(1)
plt.scatter(sc,sl, c=encoded_classe, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Comprimento da sépala')
plt.ylabel('Largura da sépala')
plt.title('Sépala')
plt.tight_layout()
plt.show()
savefig("Figs/Sepala.png",dpi=100)

#graficando pétala e colorindo por classe
plt.figure(2)
plt.scatter(pc,pl, c=encoded_classe, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Comprimento da pétala')
plt.ylabel('Largura da pétala')
plt.title('Pétala')
plt.tight_layout()
plt.show()
savefig("Figs/Petala.png",dpi=100)