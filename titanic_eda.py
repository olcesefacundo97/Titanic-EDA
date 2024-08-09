import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance

# Cargar el dataset
df = pd.read_csv('data/train.csv')

# Preprocesamiento de datos
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Crear nuevas variables
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
df['FarePerPerson'] = df['Fare'] / df['FamilySize']
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

# Convertir variables categóricas en dummy variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'AgeGroup'], drop_first=True)

# Seleccionar variables relevantes
features = ['Pclass', 'Sex_male', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson', 'Embarked_Q', 'Embarked_S', 'AgeGroup_Teen', 'AgeGroup_YoungAdult', 'AgeGroup_Adult', 'AgeGroup_Senior']
X = df[features]
y = df['Survived']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline con preprocesamiento y modelado
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Pclass', 'Age', 'Fare', 'FamilySize', 'FarePerPerson']),
        ('cat', 'passthrough', ['Sex_male', 'IsAlone', 'Embarked_Q', 'Embarked_S', 'AgeGroup_Teen', 'AgeGroup_YoungAdult', 'AgeGroup_Adult', 'AgeGroup_Senior'])
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LogisticRegression())])

# Entrenar y evaluar el pipeline
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# GridSearchCV para la Regresión Logística
param_grid = {'model__C': [0.01, 0.1, 1, 10, 100], 'model__max_iter': [100, 200, 300]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Mejores parámetros:", grid.best_params_)
print("Mejor precisión:", grid.best_score_)

# Evaluar el mejor modelo
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Comparación con otros modelos
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
print("K-Nearest Neighbors")
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# Análisis de sensibilidad
result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame(result.importances_mean, index=X.columns, columns=["Importance"])
importance_df.sort_values(by="Importance", ascending=False).plot(kind="bar")
plt.title('Importancia de las Características')
plt.show()

# Calcular y graficar la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()