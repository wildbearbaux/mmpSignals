import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict

data = pd.read_excel("DataSet_completo.xlsx")
data = data.drop(columns=['Longitud'])
normalized_data = data.copy()

cols = list(data.columns)

def calculate_z_score(df, col_name: str):
    return (df[col_name] - df[col_name].mean()) / df[col_name].std(ddof=0)

# Normalizar los datos #
for col in cols:
    normalized_data[col] = calculate_z_score(normalized_data, col)

x, y = normalized_data.drop(columns=['Clase']), data['Clase']

corr_matrix = normalized_data.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=True, yticklabels=True)
plt.title('Correlation matrix')
plt.show()

normalized_data.hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()

neural_net = MLPClassifier(hidden_layer_sizes=(22), activation='relu',
                           solver='adam', learning_rate_init=0.1,
                           max_iter=50_000,
                           random_state=42)

y_pred = cross_val_predict(neural_net, x, y, cv=5)
conf_matrix = confusion_matrix(y, y_pred)

sns.heatmap(conf_matrix, annot=True,  annot_kws={"size": 12})
plt.show()

accuracy = accuracy_score(y, y_pred)

print('Accuracy: ', accuracy * 100)



# sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, xticklabels=True, yticklabels=True)
# plt.show()

