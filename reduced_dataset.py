import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

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
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)

corr_matrix = normalized_data.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=True, yticklabels=True)
plt.title('Correlation matrix')
plt.show()

normalized_data.hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()

neural_net = MLPClassifier(hidden_layer_sizes=(22,), activation='relu',
                           solver='adam', learning_rate_init=0.001,
                           max_iter=50_000,
                           random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", neural_net),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
print(f"CV accuracy (train 80%): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True,  annot_kws={"size": 12})
plt.show()

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy test (20%): ', accuracy * 100)
print(classification_report(y_test, y_pred))



# sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, xticklabels=True, yticklabels=True)
# plt.show()
