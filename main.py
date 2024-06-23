import pandas as pd
import numpy as np
import os
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay


def ler_dados(filename, columns_to_read):
  # OBS: NÃO MUDAR ENCODING, SENAO PANDAS NAO VAI LER O CSV
  df = pd.read_csv(filename, usecols=columns_to_read, encoding="iso-8859-1", sep=";")

  #print(df[:10])
  print("\n")
  return df


#input: dataframe
#output: datasets de treinamento e teste proecessados e normalizados
def preprocessamento(df, seed_programa):
  
  #TODO: remover virgulas dos campos de float e transformar em pontos
  for nome, values in df.items():
        if nome in ["Distância de Casa", "Distância da Última Transação", "Razão entre o valor da compra e o valor médio"]:
            df[nome] = df[nome].str.replace(',', '.').astype(float)
  

  x = df.drop(['Fraude', 'Identificador da transação'], axis=1).values
  # sklearn aceita strings como variável de classificação
  y = df['Fraude'].values

  
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed_programa)
  scaler = StandardScaler()

  # importante devido às escalas diferentes das features
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test





def plot_predictions(X, model, nomes, resolution=0.02, padding=1.0):
    """
    Plot model predictions for each feature of the dataset using scatter plots.

    Parameters:
    X (ndarray): The input dataset with shape (n_samples, n_features).
    model (object): The trained model with a predict method.
    resolution (float): The resolution of the grid.
    padding (float): Padding added to the min and max of each feature range.
    """
    n_features = X.shape[1]
    for i in range(n_features):
        # Ensure the feature values are numeric
        feature_values = X[:, i].astype(np.float64)
        print(type(feature_values[0]))
    
   

    for i in range(n_features):

        x_min, x_max = feature_values.min() - padding, feature_values.max() + padding
        
        xx = np.arange(x_min, x_max, resolution)
        X_grid = np.full((len(xx), n_features), X.mean(axis=0))
        X_grid[:, i] = xx

        predictions = model.predict(X_grid)

        plt.figure(figsize=(8, 8))
        
        plt.scatter(xx, predictions, label=f'Feature {i}', alpha=0.5, edgecolors='w', s=60)
        plt.xlabel(f'Feature {nomes[i]}')
        plt.ylabel('Prediction')
        plt.title(f'Predictions for Feature {nomes[i]}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()



#---------------------------------------- MAIN ---------------------------------------------------#
#depois usar biblioteca OS pra pegar o path do arquivo
diretorio = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(diretorio, "Cartao de credito.csv")

columns_to_read = ["Identificador da transação", "Distância de Casa", "Distância da Última Transação",
                   "Razão entre o valor da compra e o valor médio", "Local Repetido", "Usou Chip", "Usou Senha",
                   "Online", "Fraude"]
nomes = columns_to_read.copy()
nomes.remove("Identificador da transação")

df = ler_dados(filename, columns_to_read)
seed_programa = 42
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test  = preprocessamento(df, seed_programa)
print("\nPRE-PROCESSAMENTO OK!")
model = LinearSVC(verbose = 3, random_state=seed_programa, class_weight={"SIM": 5, "NÃO": 1}) 

# Train the model
print("\nIniciando treinamento\n")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
#TODO: plotar labels e predicao sobre cada feature, pra saber se o modelo relamente esta prevendo bem
#OBS: plotagem tem que ser baseada nao dataset sem normalização, pra testar de verdade.
plot_predictions(X_test, model, nomes)




# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
