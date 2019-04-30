import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arquivo = "CSV/Dados_Iris.csv"

colunas = ['sepala-comprimento','sepala-largura','petala-comprimento','petala-largura','classe']

dados = pd.read_csv(arquivo, names=colunas)

shape = dados.shape
print("Exibindo dados da amostra")
print("Quantidade de linhas = ",shape[0])
print("Quantidade de colunas = ",shape[1])

print("\n")
print("Exibindo 5 primeiras linhas")
print(dados.head(5))

print("\n")
print("Exibindo 5 últimas linhas")
print(dados.tail(5))

print("\n")
print("Resumo estatístico dos dados")
print(dados.describe())

print("\n")
print("Distribuição de classes")
print(dados.groupby('classe').size())

print("\n")
print("Graficando os dados")

dados.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.tight_layout()
plt.show()
savefig("Figs/BoxPlot.png",dpi=100)

print("\n")
print("Histograma os dados")
dados.hist()
plt.tight_layout()
plt.show()
savefig("Figs/HistPlot.png",dpi=100)
