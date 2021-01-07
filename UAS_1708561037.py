import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

df = pd.read_csv('Website Phishing.csv')

x = np.array(df.drop(['Result'], 1))
y = np.array(df['Result'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)

classifer = svm.SVC(kernel='rbf')

classifer.fit(x_train, y_train)

pred_test = classifer.predict(x_test)
skor_test = accuracy_score(y_test,pred_test)

print("Result SVM Classifier: ")
print("===========================================================")
print("")
print("test: ")
print('')
print("Confusion Matrix :")
print(confusion_matrix(y_test,pred_test))
print('')
print("Classification report: ")
print(classification_report(y_test,pred_test))
