import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor

arquivo = "CSV/Boston.csv"

#Significado das colunas
#CRIM - taxa de crime per capita por cidade
#ZN - proporção de zonas de terras residenciais para lotes maiores do que 25,000 pés quadrados
#INDUS - proporção de acres de negócios não varejistas por cidade
#CHAS - variável modelo de Charles River (=1 se tratar do limite do rio e 0 nos demais casos)
#NOX - concentração de óxido nitroso (partes por 10 milhões)
#RM - número médio de salas por habitação
#AGE - proporção de unidades próprias ocupados construídas antes de 1940
#DIS - distância ponderada aos cinco distritos empregatícios de Boston
#RAD - indice de acessibilidade para rodovias
#TAX - Valor total das taxas da propriedades por $10,000
#PTRATIO - Taxa de professores por cidade
#B - 1000(Bk - 0.63)^2 onde Bk é a proporção de negros por cidade
#LSTAT  - indice inferior de população
#MEDV - Valor mediano de casas próprias ocupadas em $1000's

colunas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dados = pd.read_csv(arquivo, names=colunas, delim_whitespace=True)

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
np.set_printoptions(precision=2)
print(dados.describe())

print("\n")
print("Exibindo os tipos de dados do arranjo")
print(dados.dtypes)

print("\n")
print("Gerando histograma dos dados")
dados.hist(bins=10,figsize=(9,7),grid=False)
plt.tight_layout()
plt.show()
savefig('Figs/House-Boston-Histo.png',dpi=100)

boston_corr = dados.corr(method='pearson')
print("\n")
print("Correlação dos dados")
print(boston_corr)

#armazenando os valores das casas na variável preços
precos = dados['MEDV']
#armazenando as características das casas na variável características
caracteristicas = dados.drop('MEDV', axis = 1)

print("\n")
print("Graficando o preço médio em função de cada parâmetro")

plt.figure(2,figsize=(9,7))
plt.subplot(3, 5, 1)
plt.scatter(caracteristicas['CRIM'],precos)
plt.xlabel('CRIM')
plt.ylabel('Preço')
plt.subplot(3, 5, 2)
plt.scatter(caracteristicas['ZN'],precos)
plt.xlabel('ZN')
plt.ylabel('Preço')
plt.subplot(3, 5, 3)
plt.scatter(caracteristicas['INDUS'],precos)
plt.xlabel('INDUS')
plt.ylabel('Preço')
plt.subplot(3, 5, 4)
plt.scatter(caracteristicas['CHAS'],precos)
plt.xlabel('CHAS')
plt.ylabel('Preço')
plt.subplot(3, 5, 5)
plt.scatter(caracteristicas['NOX'],precos)
plt.xlabel('NOX')
plt.ylabel('Preço')
plt.subplot(3, 5, 6)
plt.scatter(caracteristicas['RM'],precos)
plt.xlabel('RM')
plt.ylabel('Preço')
plt.subplot(3, 5, 7)
plt.scatter(caracteristicas['AGE'],precos)
plt.xlabel('AGE')
plt.ylabel('Preço')
plt.subplot(3, 5, 8)
plt.scatter(caracteristicas['DIS'],precos)
plt.xlabel('DIS')
plt.ylabel('Preço')
plt.subplot(3, 5, 9)
plt.scatter(caracteristicas['RAD'],precos)
plt.xlabel('RAD')
plt.ylabel('Preço')
plt.subplot(3, 5, 10)
plt.scatter(caracteristicas['TAX'],precos)
plt.xlabel('TAX')
plt.ylabel('Preço')
plt.subplot(3, 5, 11)
plt.scatter(caracteristicas['PTRATIO'],precos)
plt.xlabel('PTRATIO')
plt.ylabel('Preço')
plt.subplot(3, 5, 12)
plt.scatter(caracteristicas['B'],precos)
plt.xlabel('B')
plt.ylabel('Preço')
plt.subplot(3, 5, 13)
plt.scatter(caracteristicas['LSTAT'],precos)
plt.xlabel('LSTAT')
plt.ylabel('Preço')
plt.tight_layout()
plt.show()
savefig('Figs/House-Boston-Scatter.png',dpi=100)

dados_linear_fit = caracteristicas.drop(['CRIM','ZN','INDUS','NOX','AGE','DIS','RAD'],axis=1)

print("\n")
print("Variáveis CRIM, ZN, INDUS, NOX, DIS, AGE e RAD podem ser removidas do arranjo")

print("\n")
print("Exibindo 5 primeiras linhas do novo arranjo")
print(dados_linear_fit.head(5))

print("\n")
print("Graficando os dados com ajustes lineares")

x_chas = np.linspace(np.min(caracteristicas['CHAS']), np.max(caracteristicas['CHAS']), 100)
reta_chas = np.poly1d(np.polyfit(caracteristicas['CHAS'], precos, 1))

x_rm = np.linspace(np.min(caracteristicas['RM']), np.max(caracteristicas['RM']), 100)
reta_rm = np.poly1d(np.polyfit(caracteristicas['RM'], precos, 1))

x_tax = np.linspace(np.min(caracteristicas['TAX']), np.max(caracteristicas['TAX']), 100)
reta_tax = np.poly1d(np.polyfit(caracteristicas['TAX'], precos, 1))

x_ptratio = np.linspace(np.min(caracteristicas['PTRATIO']), np.max(caracteristicas['PTRATIO']), 100)
reta_ptratio = np.poly1d(np.polyfit(caracteristicas['PTRATIO'], precos, 1))

x_b = np.linspace(np.min(caracteristicas['B']), np.max(caracteristicas['B']), 100)
reta_b = np.poly1d(np.polyfit(caracteristicas['B'], precos, 1))

x_lstat = np.linspace(np.min(caracteristicas['LSTAT']), np.max(caracteristicas['LSTAT']), 100)
reta_lstat = np.poly1d(np.polyfit(caracteristicas['LSTAT'], precos, 1))

plt.figure(3,figsize=(9,7))
plt.subplot(3, 2, 1)
plt.scatter(caracteristicas['CHAS'],precos)
plt.plot(x_chas,reta_chas(x_chas),color='red')
plt.xlabel('CHAS')
plt.ylabel('Preço')
plt.subplot(3, 2, 2)
plt.scatter(caracteristicas['RM'],precos)
plt.plot(x_rm,reta_rm(x_rm),color='red')
plt.xlabel('RM')
plt.ylabel('Preço')
plt.subplot(3, 2, 3)
plt.scatter(caracteristicas['TAX'],precos)
plt.plot(x_tax,reta_tax(x_tax),color='red')
plt.xlabel('TAX')
plt.ylabel('Preço')
plt.subplot(3, 2, 4)
plt.scatter(caracteristicas['PTRATIO'],precos)
plt.plot(x_ptratio,reta_ptratio(x_ptratio),color='red')
plt.xlabel('PTRATIO')
plt.ylabel('Preço')
plt.subplot(3, 2, 5)
plt.scatter(caracteristicas['B'],precos)
plt.plot(x_b,reta_b(x_b),color='red')
plt.xlabel('B')
plt.ylabel('Preço')
plt.subplot(3, 2, 6)
plt.scatter(caracteristicas['LSTAT'],precos)
plt.plot(x_lstat,reta_lstat(x_lstat),color='red')
plt.xlabel('LSTAT')
plt.ylabel('Preço')
plt.tight_layout()
plt.show()
savefig('Figs/House-Boston-Scatter-Reta.png',dpi=100)

print("\n")
print("Gerando um histograma da distribuição de preços")

plt.figure(4,figsize=(9,7))
sns.distplot(precos)
plt.tight_layout()
plt.show()
savefig('Figs/House-Boston-HistogramaPreco.png',dpi=100)

print("\n")
print("Gerando box plots da quantidades essenciais para o cálculo do preço")
dados_linear_fit.plot(kind='box', subplots=True, layout=(2,4), figsize=(9,7), sharex=False, sharey=False)
plt.tight_layout()
plt.show()
savefig('Figs/House-Boston-BoxPlot.png',dpi=100)

boston_corr2 = dados.drop(['CRIM','ZN','INDUS','NOX','AGE','DIS','RAD'],axis=1).corr(method='pearson')

plt.figure(6,figsize=(8,8))
sns.heatmap(boston_corr2, vmax=.8, linewidths=0.01, square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlação entre as características')
plt.tight_layout()
plt.show()
savefig('Figs/House-Boston-Correlacao.png',dpi=100)

print("\n")
print("Determinando existencia de NaNs no arranjo geral")
n_nans = dados.isnull().sum()
print(n_nans)
print("\n")

plt.figure(7,figsize=(8,8))
res = stats.probplot(precos, plot=plt)
plt.xlabel('Quantiles')
plt.ylabel('Valores ordenados')
plt.title('Probabilidade')
plt.tight_layout()
plt.show()
savefig('Figs/House-Boston-Probabilidade.png',dpi=100)

X = dados_linear_fit.values
Y = precos.values

teste_tamanho = 0.25
seed = 7
num_folds = 10
RMS = 'neg_mean_absolute_error'

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=teste_tamanho, random_state=seed)

modelos = []
nomes = []
resultados = []
mses = []
maes = []
r2 = []
rmses = []

modelos.append(('RL', LinearRegression()))
modelos.append(('Lasso', Lasso()))
modelos.append(('KNN', KNeighborsRegressor()))
modelos.append(('CART', DecisionTreeRegressor()))
modelos.append(('SVR', SVR(gamma='auto')))
modelos.append(('Ada', AdaBoostRegressor()))
modelos.append(('GBM', GradientBoostingRegressor()))
modelos.append(('RF', RandomForestRegressor(n_estimators=10)))
modelos.append(('ET', ExtraTreesRegressor(n_estimators=10)))
modelos.append(('XGB', XGBRegressor()))


for nome, modelo in modelos:
	modelo.fit(X_treino, Y_treino)
	previsao = modelo.predict(X_teste)
	mse = mean_squared_error(Y_teste, previsao)
	mses.append(mse)
	rmse = np.sqrt(mean_squared_error(Y_teste, previsao))
	rmses.append(rmse)
	mae = mean_absolute_error(Y_teste, previsao)
	maes.append(mae)
	r2_ind = r2_score(Y_teste, previsao)
	r2.append(r2_ind)
	print("Modelo: %s MSE = %f" % (nome, mse))
	print("Modelo: %s RMSE = %f" % (nome, rmse))
	print("Modelo: %s MAE = %f" % (nome, mae))
	print("Modelo: %s R^2 = %f" % (nome, r2_ind))

print("\n")	
for nome, modelo in modelos:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_resultados = cross_val_score(modelo, X_treino, Y_treino, cv=kfold, scoring=RMS)
    resultados.append(cv_resultados)
    nomes.append(nome)
    msg = "Modelo: %s Média: %f (Sigma:%f)" % (nome, cv_resultados.mean(), cv_resultados.std())
    print(msg)
    
fig = plt.figure(8,figsize=(8,8))
fig.suptitle('Comparação dos modelos')
ax = fig.add_subplot(111)
plt.boxplot(resultados)
ax.set_xticklabels(nomes)
plt.tight_layout()
plt.show()
savefig('Figs/House-Boston-BoxPlotModelos.png',dpi=100)

print("\n")	
print("Dimensionando os dados")

pipelines = []
pipelines.append(('RL Dimensionado', Pipeline([('Scaler', StandardScaler()),('Regressão linear', LinearRegression())])))
pipelines.append(('Lasso Dimensionado', Pipeline([('Scaler', StandardScaler()),('Lasso', Lasso())])))
pipelines.append(('KNN Dimensionado', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('CART Dimensionado', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('SVR Dimensionado', Pipeline([('Scaler', StandardScaler()),('SVR', SVR(gamma='auto'))])))
pipelines.append(('Ada Dimensionado', Pipeline([('Scaler', StandardScaler()),('Ada', AdaBoostRegressor())])))
pipelines.append(('GBM Dimensionado', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
pipelines.append(('RF Dimensionado', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor(n_estimators=10))])))
pipelines.append(('ET Dimensionado', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor(n_estimators=10))])))
pipelines.append(('XGB Dimensionado', Pipeline([('Scaler', StandardScaler()),('XGB', XGBRegressor())])))

resultados2 = []
nomes2 = []

for nome, modelo in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_resultados = cross_val_score(modelo, X_treino, Y_treino, cv=kfold, scoring=RMS)
	resultados2.append(cv_resultados)
	nomes2.append(nome)
	msg = "%s: %f (%f)" % (nome, cv_resultados.mean(), cv_resultados.std())
	print(msg)
	
fig = plt.figure(9,figsize=(15,7))
fig.suptitle('Comparação dos modelos dimensionados')
ax = fig.add_subplot(111)
plt.boxplot(resultados2)
ax.set_xticklabels(nomes2)
plt.tight_layout()
plt.show()
savefig('Figs/House-Boston-BoxPlotModelosDimensionados.png',dpi=100)

print("\n")
print("Adotoando modelo KNN para determinar melhor número de vizinhos")

KNN_modelo = StandardScaler().fit(X_treino)
Xtreinomodelado = KNN_modelo.transform(X_treino)
k_valores = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
parametros_grid = dict(n_neighbors=k_valores)
modelo = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=modelo, param_grid=parametros_grid, scoring=RMS, cv=kfold)
grid_resultado = grid.fit(Xtreinomodelado, Y_treino)

print("Determinando melhor valor")
print("Melhor: %f usando %s" % (grid_resultado.best_score_, grid_resultado.best_params_))

medias = grid_resultado.cv_results_['mean_test_score']
desvios = grid_resultado.cv_results_['std_test_score']
parametros = grid_resultado.cv_results_['params']

print("\n")
print("Media"," Desvio"," Parâmetros")
for media, desvio, parametro in zip(medias, desvios, parametros):
    print("%f (%f) with: %r" % (media, desvio, parametro))
    
print("\n")
print("Adotoando modelo GBM para determinar melhor número de vizinhos por dar melhores resultados")
parametros_grid2 = dict(n_estimators=numpy.array([100,101,102,103,104,105,106,107,108,109,110]))
modelo2 = GradientBoostingRegressor(random_state=seed)
kfold2 = KFold(n_splits=num_folds, random_state=seed)
grid2 = GridSearchCV(estimator=modelo2, param_grid=parametros_grid2, scoring=RMS, cv=kfold2)
grid_resultado2 = grid2.fit(Xtreinomodelado, Y_treino)

print("Determinando melhor valor")
print("Melhor: %f usando %s" % (grid_resultado2.best_score_, grid_resultado2.best_params_))

medias2 = grid_resultado2.cv_results_['mean_test_score']
desvios2 = grid_resultado2.cv_results_['std_test_score']
parametros2 = grid_resultado2.cv_results_['params']

print("\n")
print("Media"," Desvio"," Parâmetros")
for media2, desvio2, parametro2 in zip(medias2, desvios2, parametros2):
    print("%f (%f) with: %r" % (media2, desvio2, parametro2))
    
print("\n")
print("Fazendo previsões utilizando modelo GBM")
escalonador = StandardScaler().fit(X_treino)
modelo_GBM = GradientBoostingRegressor(random_state=seed, n_estimators=102)
modelo_GBM.fit(Xtreinomodelado, Y_treino)
Xtestemodelado = escalonador.transform(X_teste)
previsaoGBM = modelo_GBM.predict(Xtestemodelado)
print("Erro quadrático médio")
print(mean_squared_error(Y_teste, previsaoGBM))

plt.figure(10,figsize=(8,8))
plt.scatter(Y_teste, previsaoGBM)
plt.xlabel("Preços")
plt.ylabel("Preços previstos")
plt.tight_layout()
plt.show()

valores = pd.DataFrame({"Valores estimados": Y_teste, "Valores previstos": previsaoGBM})
valores.to_csv("CSV/House-Boston-Previsao.csv", index=False)