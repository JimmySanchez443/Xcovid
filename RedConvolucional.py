# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:50:42 2021

@author: jimsa
"""


import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
K.clear_session()

#extraer caracteristcas de las imagenes 
#directorio de nuestas imagenes  
data_entrenamiento = './dataNueva/data/train'
data_validacion = './dataNueva/data/test'

"""
Paramets
"""

epocas = 20
# numero de veces que vamops a iterar
altura, longitud = 100, 100
# cambiamos todas las imagenes a 100 100 pixeles
batch_size = 32 
#numero de imagenes que mandamos a procesar 
pasos = 1000
#numero de veces que vamos a procesar la imagen por cada epoca
pasos_validacion = 200

filtrosConvl = 64
# despues de hacer el filtro con una convolucionde 32 y despues de64
filtrosConv2 = 32
tamano_filtro1 = (2, 2)

tamano_filtro2 = (3, 3)

tamano_pool = (2,2)
clases = 3
#ajustes que va hacer la red convolucional 
lr = 0.0005

#pre procesamiento de imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.3, #va a inclinar nuestras imagenes 
    zoom_range  =0.3, # va hacer zoom a las imagenes
    horizontal_flip=True) # toma una imagen y la invierte 

validacion_datagen = ImageDataGenerator(
    rescale=1. / 255) # solo enviamos la imagen rescalada


entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento, 
    target_size = (altura, longitud), 
    batch_size = batch_size, 
    class_mode ='categorical') # es una clasificacion categorical 

validacion_generador = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
     class_mode = 'categorical')

cnn = Sequential()
cnn.add(Convolution2D(filtrosConvl , tamano_filtro1, padding ="same", input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))


cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

#es laimagen que es profunda pero muy pequeña la hacemos plana  que tiene toda la informacion
cnn.add(Flatten())
# enviamos a una capa de neuronas 256
cnn.add(Dense(512, activation='relu')) 
# apgamos el 50% de la neuronas de manera aleatoria va  aprender de diferentes maneras
cnn.add(Dropout(0.2))
# es la ultima capa  como una funcion de activacion softmax no diceel porcentahe de clasificacion
cnn.add(Dense(clases, activation='softmax'))
 
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])



cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch = pasos,
    epochs = epocas,
    validation_data = validacion_generador,
    validation_steps = pasos_validacion)





target_dir = './modeloNuevo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modeloNuevo/modelo.h5')
cnn.save_weights('./modeloNuevo/pesos.h5')

from sklearn.metrics import classification_report, confusion_matrix

validacion_generador2 = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
     class_mode = 'categorical',
     shuffle=False
     )

ytest= validacion_generador2.classes

Y_pred = cnn.predict_generator(validacion_generador2, steps= len(validacion_generador2))
y_pred = np.argmax(Y_pred, axis=1)
class_labels = list(validacion_generador2.class_indices.keys())   


print(validacion_generador)
print(y_pred.ndim)
print(classification_report(ytest, y_pred, target_names=class_labels))
print('Confusion Matrix')
print(list(validacion_generador2.class_indices.keys()))
print(confusion_matrix(ytest, y_pred))


