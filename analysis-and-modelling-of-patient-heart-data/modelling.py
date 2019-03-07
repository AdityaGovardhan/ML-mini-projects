###############################################
## HEART DISEASE OCCURENCE PREDICTION MODEL  ##
## CREATED BY: ADITYA GOVARDHAN              ##
###############################################

import numpy as np
import pandas as pd
import numpy as np

#####################################
## DATA COLLECTION AND ADJUSTMENTS ##
#####################################
iris = pd.read_csv('heart1.csv')

X = iris.iloc[:, 0:-1].astype(float)    # first thirteen parameters
y = iris.iloc[:, -1].astype(float)      # target parameter a1p2

# split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# scale X
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print('============================================================')

######################
## PERCEPTRON MODEL ##
######################
from sklearn.linear_model import Perceptron
print("PERCEPTRON")

ppn = Perceptron(max_iter=40, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=False)
ppn.fit(X_train_std, y_train)

print('Number of samples in test: ', len(y_test))
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy for test data: %.2f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number of samples in combined: ', len(y_combined))

y_combined_pred = ppn.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('============================================================')

###############################
## LOGISTIC REGRESSION MODEL ##
###############################
from sklearn.linear_model import LogisticRegression
print("LOGISTIC REGRESSION")

lr = LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)

print('Number of samples in test: ', len(y_test))
y_pred = lr.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy for test data: %.2f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number of samples in combined: ', len(y_combined))

y_combined_pred = lr.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('============================================================')

###############
## SVM MODEL ##
###############
from sklearn.svm import SVC
print("SPACE VECTOR MACHINE (SVM)")

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

print('Number of samples in test: ', len(y_test))
y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy of test data: %.2f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number of samples in combined: ', len(y_combined))

y_combined_pred = svm.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('============================================================')

####################################
## DECISION TREE CLASSIFIER MODEL ##
####################################
from sklearn.tree import DecisionTreeClassifier
print("DECISION TREE LEARNING")

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train_std, y_train)

print('Number of samples in test: ', len(y_test))
y_pred = tree.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy of test data: %.2f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number of samples in combined: ', len(y_combined))

y_combined_pred = tree.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('============================================================')

###############################
## K NEAREST NEIGHBORS MODEL ##
###############################
from sklearn.neighbors import KNeighborsClassifier
print("K NEAREST NEIGHBORS")

knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

print('Number of samples in test: ', len(y_test))
y_pred = knn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy of test data: %.2f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number of samples in combined: ', len(y_combined))

y_combined_pred = knn.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('============================================================')