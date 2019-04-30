import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#Determinando a matriz de confusão

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Matriz de confusão normalizada'
        else:
            title = 'Matriz de confusão não normalizada'

    #Calculando a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    #Utilizando os nomes que aparecem nos dados
    classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão não normalizada')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='Nome verdadeiro',xlabel='Nome previsto')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

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

# Separando o arranjo numa amostra de teste e numa amostra de treino
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.25)

#Definindo um classificador que utiliza um modelo que é bem regularizados, ou seja, um valor de C bem baixo.
classificador = svm.SVC(kernel='linear', C=0.01)

#Fazendo ajuste dados e realizando previsão dos resultados
y_pred = classificador.fit(X_train, y_train).predict(X_test)

#Definindo a precisão como tendo duas casas decimais
np.set_printoptions(precision=2)

#Graficando a matriz de confusão não normalizada
plot_confusion_matrix(y_test, y_pred, classes=classe, title='Matriz de confusão não normalizada')
plt.tight_layout()
plt.show()
savefig("Figs/MatrizConfusaoNaoNormalizada.png",dpi=100)

#Graficando a matriz de confusão normalizada
plot_confusion_matrix(y_test, y_pred, classes=classe, normalize=True, title='Matriz de confusão normalizada')
plt.tight_layout()
plt.show()
savefig("Figs/MatrizConfusaoNormalizada.png",dpi=100)


