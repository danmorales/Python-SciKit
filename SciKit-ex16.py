import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

#KNN

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.25)

scoring = 'accuracy'

KNN = KNeighborsClassifier()
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(KNN, X_train, Y_train, cv=kfold, scoring=scoring)

msg = "%s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())
print("\n")
print("Método: média (desvio padrão)")
print(msg)

print("\n")
print("Ajustando os dados utilizando os dados de treino")
KNN.fit(X_train, Y_train)

print("\n")
print("Realizando previsões utilizando os dados de teste")
predictions = KNN.predict(X_test)

print("\n")
print("Score")
score = KNN.score(X_test, Y_test)
print(score)

print("\n")
print("Score de precisão")
precisao = accuracy_score(Y_test, predictions)
print(precisao)

print("\n")
print("Matriz de confusão")
matriz = confusion_matrix(Y_test, predictions)
print(matriz)

print("\n")
print("Relatório de classificação")
relatorio = classification_report(Y_test, predictions)
print(relatorio)

print("\n")
scores_cross_val = cross_val_score(KNN, X, Y, cv=5)
print("Score de cross validação")
print(scores_cross_val)

print("Precisão: %0.2f (+/- %0.2f)" % (scores_cross_val.mean(), scores_cross_val.std() * 2))

print("\n")
#Graficando a matriz de confusão não normalizada
plot_confusion_matrix(Y_test, predictions, classes=classe, title='Matriz de confusão KNN')
plt.tight_layout()
plt.show()
savefig("Figs/Iris-MatrizConfusao_KNN.png",dpi=100)