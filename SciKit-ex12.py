import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

arquivo = "CSV/Dados_Iris.csv"

colunas = ['sepala-comprimento','sepala-largura','petala-comprimento','petala-largura','classe']

nomes = ['setosa','versicolor','virginica']

dados = pd.read_csv(arquivo, names=colunas)

#convertendo a classe de strings para valores numéricos
classe = dados['classe']
encoder = LabelEncoder()
encoder.fit(classe)
encoded_classe = encoder.transform(classe)

#armazenando os dados da sépala e pétala na variável X como um arranjo NumPy
X = np.array(dados[['sepala-comprimento','sepala-largura','petala-comprimento','petala-largura']])

#armazenando os dados da classe convertida num arranjo NumPy
Y = np.array(encoded_classe,dtype=int)

#Separando os dados numa combinação de treino e teste
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y, test_size=0.25, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y, test_size=0.25, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y, test_size=0.25, random_state=0, shuffle=False)

print("Tamanho do X de treinamento = ",X_train.size)
print("Tamanho do X de teste = ",X_test.size)
print("Tamanho do Y de treinamento = ",y_train.size)
print("Tamanho do Y de teste = ",y_test.size)

plt.figure(figsize=(10, 6))
plt.scatter(X_train1[:,0],y_train1,color='black')
plt.scatter(X_train2[:,0],y_train2,color='red')
plt.scatter(X_train3[:,0],y_train3,color='blue')
plt.show()