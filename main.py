import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def ler_dados(filename, columns_to_read):
  # OBS: NÃO MUDAR ENCODING, SENAO PANDAS NAO VAI LER O CSV
  df = pd.read_csv(filename, usecols=columns_to_read, encoding="iso-8859-1", sep=";")

  print(df[:10])
  print("\n")
  return df


#input: dataframe
#output: datasets de treinamento e teste proecessados e normalizados
def preprocessamento(df, seed_programa):
  x = df.drop(['Fraude', 'Identificador da transação'], axis=1).values
  # sklearn aceita strings como variável de classificação
  y = df['Fraude'].values

  #TODO: remover virgulas dos campos de float e transformar em pontos
  for coluna in x[:]:
    if(coluna in ["Distância de Casa", "Distância da Última Transação", "Razão entre o valor da compra e o valor médio"]):
      

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed_programa)
  scaler = StandardScaler()

  # importante devido às escalas diferentes das features
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)


#depois usar biblioteca OS pra pegar o path do arquivo
filename = "C:\\Users\\Eu\\Downloads\\Cartao de credito.csv"
columns_to_read = ["Identificador da transação", "Distância de Casa", "Distância da Última Transação",
                   "Razão entre o valor da compra e o valor médio", "Local Repetido", "Usou Chip", "Usou Senha",
                   "Online", "Fraude"]
df = ler_dados(filename, columns_to_read)
seed_programa = 42

model = SVC(kernel='rbf', random_state=seed_programa)

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))