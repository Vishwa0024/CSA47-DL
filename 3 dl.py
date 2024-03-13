from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X = data.data
y = data.target

clf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=5)
mean_cv_score = cv_scores.mean()

print("Mean Cross-Validation Score:", mean_cv_score)
