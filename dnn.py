import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import seaborn as sns
from keras.layers import *
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import cv2
from feature_extractor1 import getFeatures
import feature_extractor
from sklearn.metrics import confusion_matrix


df = shuffle(pd.read_csv("features_train.csv"))
y_train = df.iloc[:,46]
y_train = to_categorical(y_train)
scaler = MinMaxScaler()
scaler.fit(df.iloc[:, 1:46])
X_train = np.array(scaler.transform(df.iloc[:, 1:46]))

df_test = shuffle(pd.read_csv("features_test.csv"))
y_test = df_test.iloc[:,46]
y_test = to_categorical(y_test)
X_test = np.array(scaler.transform(df_test.iloc[:, 1:46]))

clf = keras.Sequential([
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(5, activation='softmax'),
])
clf.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = clf.fit(X_train, y_train, batch_size=64, validation_split=0.1, epochs=250)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

clf.evaluate(X_test, y_test)

CM = confusion_matrix(df_test.iloc[:,46], np.argmax(clf.predict(X_test), axis=1))
ax = plt.axes()


tags = ['Heart', 'Oval', 'Oblong', 'Round', 'Square']
sns.heatmap(CM, annot=True,
           xticklabels= tags,
           yticklabels= tags, ax= ax)
ax.set_title('Confusion matrix')
plt.show()
