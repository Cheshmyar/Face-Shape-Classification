import os
import pandas as pd
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from PIL import ImageFile
import seaborn as sns


ImageFile.LOAD_TRUNCATED_IMAGES = True

class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (224, 224)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split= 0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        './Data/FaceShape Dataset/training_set',
        shuffle=True,
        batch_size=128,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        './Data/FaceShape Dataset/testing_set',
        shuffle=False,
        batch_size=128,
        class_mode='categorical')

base_model = keras.applications.InceptionV3(include_top=False,
                   input_shape=(512,512,3),
                   weights='imagenet')

x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
predictions = keras.layers.Dense(5, activation='softmax')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.000001, verbose=1)

model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 30

callbacks_list = [
    reduce_lr,
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,verbose=1)
]

history = model.fit(
        train_generator,
        epochs=epochs,
        callbacks = callbacks_list,
        validation_data=validation_generator,
        verbose = 1)

hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

model.evaluate(validation_generator)

y_test = np.argmax(np.array(pd.get_dummies(pd.Series(validation_generator.classes))), axis=1)
print(y_test)
prediction = np.argmax(np.array(model.predict(validation_generator)), axis=1)
print(prediction)


CM = confusion_matrix(y_test, prediction)
ax = plt.axes()
sns.heatmap(CM, annot=True,
           xticklabels= class_names,
           yticklabels= class_names, ax= ax)
ax.set_title('Confusion matrix')
plt.show()

print(accuracy_score(y_test, prediction))
print(precision_score(y_test, prediction, average='macro'))
print(recall_score(y_test, prediction, average='macro'))

# fig1 = plt.gcf()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.axis(ymin=0.4,ymax=1)
# plt.grid()
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.legend(['train', 'validation'])
# plt.show()
