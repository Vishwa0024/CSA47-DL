Q 6(e) . Finding accuracy value of iris data set using SVM  algorithm
  # Importing the dataset
dataset = pd.read_csv("/content/IRIS.csv ")
# Splitting the dataset into the Training set and Test set
# Feature Scaling
# Training the SVM model on the Training set
# Predicting the Test set results
# Evaluating the performance of the model using confusion matrix and accuracy.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv("/content/iris.csv")

# Splitting the dataset into the features (X) and the target variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM model on the Training set
classifier = SVC(kernel='linear')  # You can specify different kernels like 'linear', 'rbf', etc.
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluating the performance of the model using confusion matrix and accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
