from sklearn.metrics import confusion_matrix
import numpy as np

actual = np.array(['Dog', 'Dog', 'Not Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Not Dog', 'Dog'])

predicted = np.array(['Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog'])
labels = ['Dog', 'Not Dog']
cm = confusion_matrix(actual, predicted, labels=labels)
print("Confusion Matrix:")
print(cm)
