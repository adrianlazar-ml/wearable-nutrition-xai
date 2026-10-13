"""One-off check for T0.4: is balanced_accuracy == f1_macro a bug or structure?"""
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, balanced_accuracy_score,
                             precision_score, recall_score)

df = pd.read_csv(r"revision\results\test_predictions_persistence.csv")
y, p = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
cm = confusion_matrix(y, p, labels=[0, 1, 2])
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print("support per class (true):  ", cm.sum(axis=1))
print("predicted count per class: ", cm.sum(axis=0))
print("precision per class:", precision_score(y, p, labels=[0, 1, 2], average=None, zero_division=0))
print("recall per class:   ", recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0))
print("f1_macro:           ", f1_score(y, p, average="macro", zero_division=0))
print("balanced_accuracy:  ", balanced_accuracy_score(y, p))
print("y_pred[0] (= last train label):", p[0])
print("y_true[-1] (last test label):  ", y[-1])
