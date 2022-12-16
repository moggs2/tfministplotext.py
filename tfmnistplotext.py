import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

#folderlist=[]
#
#photodir = Path('/root/flower_photos')
#
#for x in photodir.iterdir():
#    if x.is_dir():
#        folderlist.append(x.name)

#print(folderlist)

train_ds, info = tfds.load('mnist', split='train', batch_size=32, shuffle_files=True, as_supervised=True, with_info=True)
val_ds, info2 = tfds.load('mnist', split='test', batch_size=32, shuffle_files=True, as_supervised=True, with_info=True)

print(info)

print(info2)

#train_ds = tf.keras.utils.image_dataset_from_directory(
#  Path('/root/pizza_steak/train'),
#  #validation_split=0.2,
#  #subset="training", #deactivate splitting
#  seed=123,
#  #image_size=(128, 128),
#  label_mode="binary",
#  #label_mode="categorical",
#  batch_size=32
#  )
#
#print(train_ds)
#
#val_ds = tf.keras.utils.image_dataset_from_directory(
#  Path('/root/pizza_steak/test'),
#  #validation_split=0.2,
#  #subset="validation", deactivate splitting
#  seed=123,
#  #image_size=(128, 128),
#  label_mode="binary",
#  #label_mode="categorical",
#  batch_size=32
#  )
#
#print(val_ds)

#class_names = train_ds.class_names
class_names=info.features['label'].names
print(class_names)


import plotext as plt
#plt.image_plot("/root/pizza_steak/test/pizza/11297.jpg")
#plt.show()
#plt.clear_figure()

from PIL import Image
#train_ds = train_ds.shuffle(1000)
#train_ds = train_ds.batch(32)

import os
imageview=tfds.as_numpy(train_ds)
print(imageview)
i=0
for image in imageview:   #Iterating through all batches and shows example image
   print(str(i))
   print(image[1][0]) #Second number is image from batch, first number none zero is result name
   print(image[0][0].shape)
   img = tf.keras.preprocessing.image.array_to_img(image[0][0])
   img.save(str(image[1][0]) + ".jpg")
   plt.image_plot(str(image[1][0]) + ".jpg")
   plt.show()
   plt.clear_figure()
   #os.remove("jpgforplotext.jpg")
   i=i+1
   break


#print(int(image[1][0]))
#print(np.where(image[1][0]==1))

from tensorflow.keras import layers

data_augmentation_test = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

for i in range(3):
  augmented_image = data_augmentation_test(image[0][0])
  img = tf.keras.preprocessing.image.array_to_img(augmented_image)
  img.save(str(image[1][0]) + 'test.jpg')
  plt.image_plot(str(image[1][0]) + 'test.jpg')
  plt.show()
  plt.clear_figure()
  #os.remove("jpgforplotext.jpg")


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#model = tf.keras.Sequential([
#    tf.keras.layers.Flatten(input_shape=[28, 28]),
#    tf.keras.layers.AlphaDropout(rate=0.2),
#    tf.keras.layers.BatchNormalization(),
#    tf.keras.layers.Dense(300, activation="relu"),
#    tf.keras.layers.AlphaDropout(rate=0.2),
#    tf.keras.layers.BatchNormalization(),
#    tf.keras.layers.Dense(100, activation="relu"),
#    tf.keras.layers.AlphaDropout(rate=0.2),
#    tf.keras.layers.BatchNormalization(),
#    tf.keras.layers.Dense(10, activation="softmax")
#])


total_imagefolders = 10 #

model = tf.keras.Sequential([
  #tf.keras.layers.RandomWidth(factor=(0.1, 0.4), interpolation='bilinear', seed=None),
  #tf.keras.layers.RandomHeight(factor=(0.2, 0.4), interpolation='bilinear', seed=None),
  #tf.keras.layers.Resizing(180, 180),
  tf.keras.layers.Rescaling(1./255),

  #tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  #tf.keras.layers.RandomRotation(0.2),
  #tf.keras.layers.RandomZoom(.1, .2),
  #Shearing an image is not included here

  tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu'),
  #tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
  tf.keras.layers.Flatten(),
  #tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(total_imagefolders, activation='softmax')
  #tf.keras.layers.Dense(1, activation='sigmoid') # binary activation output
])

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
  #loss="binary_crossentropy",   
  metrics=['accuracy'])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=20)
best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("mnist_model.h5", save_best_only=True)
learningratecallbackchange=tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0015 * 0.9 ** epoch)

fittingdiagram=model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100,
  callbacks=[best_checkpoint_callback, early_stopping_callback, learningratecallbackchange])

model.summary()

import plotext as plt

plt.clp()
plt.plot(fittingdiagram.history['loss'], xside= "lower", yside = "left", label="loss")
plt.plot(fittingdiagram.history['accuracy'], xside= "lower", yside = "left", label="accuracy")
plt.plot(fittingdiagram.history['val_loss'], xside= "lower", yside = "left", label="val_loss")
plt.plot(fittingdiagram.history['val_accuracy'], xside= "lower", yside = "left", label="val_accuracy")
plt.plot(fittingdiagram.history['lr'], xside= "lower", yside = "left", label="learning_rate")
plt.title("Loss and accuracy")
plt.show()

lrloss=pd.DataFrame(fittingdiagram.history['lr'], fittingdiagram.history['loss'])
print(lrloss)
plt.clp()
plt.plot(lrloss[0], xside= "lower", yside = "left", label="learning rate * 100")
plt.plot(lrloss.index, xside= "lower", yside = "right", label="loss")
plt.show()

pred_image_name=class_names[int(image[1][3])]
print(pred_image_name)
prediction=model.predict(image[0][3][np.newaxis, :, :])
print(round(prediction[0][0]))
print(class_names[round(prediction[0][0])])
