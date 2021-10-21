# KNeighbors Classifier

El algoritmo K-vecinos cercanos es un algoritmo de clasificación no paramétrico. Se utiliza en conjuntos de datos pequeños y medianos ya que su costo computacional es alto con respecto a otros algoritmos.

```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from knn import KNeighbors

X = load_iris()["data"]
y = load_iris()["target"]

# Se crea un conjunto de entrenamiento y prueba
# Se utiliza un 30% para prueba.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Se crea un objeto del tipo KNeighbors
knn_clf = KNeighbors(n_neighbors=5)

# Se "entrena" el modelo
knn_clf.fit(x_train, y_train)

# Se predicen las clases
y_pred = knn_clf.predict(x_test)

# Evaluación del modelo
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy Score: {:.2f}".format(acc))
print("Confusion Matrix: ", cm)

```