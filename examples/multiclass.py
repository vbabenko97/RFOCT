"""Fit RFOCT on a small synthetic multiclass classification problem."""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from rfoct import RFOCTClassifier

X, y = make_classification(
    n_samples=120,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=11,
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=11,
)

classifier = RFOCTClassifier(
    n_estimators=3,
    max_level=2,
    max_features=2,
    ga_population=2,
    ga_epochs=1,
    random_state=11,
)
classifier.fit(X_train, y_train)

print("classes:", classifier.classes_)
print("predictions:", classifier.predict(X_test[:5]))
print("vote shares:", classifier.predict_proba(X_test[:5]))
print("accuracy:", classifier.score(X_test, y_test))
