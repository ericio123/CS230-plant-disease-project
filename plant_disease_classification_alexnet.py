#Classifies disease from images of plant leaves using the PlantVillage dataset and transfer learning from AlexNet

import os
from pathlib import Path
#Replace with path to AlexNet weights and model
os.listdir("C:/Users/erici/Desktop/pdisease")
# Importing required libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

#Creating the AlexNet architecture
classifier = Sequential()
classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())
classifier.add(Flatten())
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 1000, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 38, activation = 'softmax'))
classifier.summary()

transfer = False
#Replace with path to AlexNet weights
if transfer == True:
    classifier.load_weights('C:/Users/erici/Desktop/pdisease/plant-diseases-classification-using-alexnet/AlexNetWeights.hdf5')
#Freezing layers
if transfer == True:
    for i, layer in enumerate(classifier.layers[:6]):
        print(i, layer.name)
        layer.trainable = False
#Model summary
classifier.summary()

# Compiling model
classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Augmenting images with a data generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 128
#Replace with path to dataset
base_dir = "C:/Users/erici/Desktop/plant-diseases-alexnet/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
#Creating train, validation, and test sets
training_set = train_datagen.flow_from_directory(base_dir+'/train',
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')
valid_set = valid_datagen.flow_from_directory(base_dir+'/valid',
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')
test_set = valid_datagen.flow_from_directory(base_dir+'/test',
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')

class_dict = training_set.class_indices
train_num = training_set.samples
valid_num = valid_set.samples
test_num = test_set.samples

weightpath = "AlexNetWeights.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]
#Training
history = classifier.fit_generator(training_set,
                         steps_per_epoch=train_num//batch_size,
                         validation_data=valid_set,
                         epochs=25,
                         validation_steps=valid_num//batch_size,
                         callbacks=callbacks_list)

#Plotting training and validation accuracy and loss
sns.set()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, acc, color='red', label='Train Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, color='red', label='Train Loss')
plt.plot(epochs, val_loss, color='blue', label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Evaluating model
scores = classifier.evaluate_generator(test_set)
print(f"Test Accuracy: {scores[1]*100}")

cm_set = test_datagen.flow_from_directory(base_dir+'/test',
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')
cm_num = cm_set.samples
#Creating classes and prediction array
Y_pred = classifier.predict_generator(cm_set, cm_num // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
classes = []
prediction = []
count = 0
break_ = False
for (x,y) in cm_set:
    n, m = y.shape
    prediction.append(np.argmax(classifier.predict(x)))
    for i in range(n):
        classes.append(np.argmax(y[i, :]))
        count += 1
        if count == cm_num:
            break_ = True
            break
    if break_:
        break
print(classes, prediction)
print(len(classes), y_pred.shape)
cm = confusion_matrix(classes, prediction, labels = list(range(38)), normalize='true')

#Plotting confusion matrix
labels = list(class_dict)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion matrix ')
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(range(38))
plt.yticks(range(38), labels)
plt.rcParams["figure.figsize"] = [40,5]
plt.colorbar()
plt.show()
