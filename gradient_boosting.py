import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

df = shuffle(pd.read_csv("features_train.csv"))
y_train = df.iloc[:,46]
X_train = np.asarray(df)[:, 1:46]

df_test = shuffle(pd.read_csv("features_test.csv"))
y_test = df_test.iloc[:,46]
X_test = np.asarray(df_test)[:, 1:46]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=4).fit(X_train, y_train)

predicted = clf.predict(X_test)
print(accuracy_score(y_test, predicted))

CM = confusion_matrix(y_test, predicted)
ax = plt.axes()


tags = ['Heart', 'Oval', 'Oblong', 'Round', 'Square']
sns.heatmap(CM, annot=True,
           xticklabels= tags,
           yticklabels= tags, ax= ax)
ax.set_title('Confusion matrix')
plt.show()

