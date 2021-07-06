# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:25:49 2021

@author: jimsa
"""

import librosa
import os
import os, sys
import librosa
import json
import imutils
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import librosa.display
import scipy.io.wavfile as waves
import base64
from os import remove

import glob
import numpy as np
from tensorflow.python.keras.preprocessing.image import  load_img, img_to_array
from keras.models import load_model

longitud, altura = 100,100
modelo = 'D:/Semestre 2020-2021/Aplicaciones de Mineria de Datos/Proyecto/modelo1/modelo.h5'
pesos = 'D:/Semestre 2020-2021/Aplicaciones de Mineria de Datos/Proyecto/modelo1/pesos.h5'

modelo1 = 'D:/Semestre 2020-2021/Aplicaciones de Mineria de Datos/Proyecto/modelo8/modelo.h5'
pesos1 = 'D:/Semestre 2020-2021/Aplicaciones de Mineria de Datos/Proyecto/modelo8/pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos)

cnn1 = load_model(modelo1)
cnn1.load_weights(pesos1)



app = Flask(__name__)


@app.route("/xcovid", methods=['POST'])
def service():
   
    respuesta = ""
    values = request.get_json()
    datos = values['img']
    print("DATOS ENVIADOS POR JSON:")
    #print(datos)
    deserializar(datos)
    prediccion = predict('img.jpg')

    respuesta = {'pred':prediccion}

    return json.dumps(respuesta)

@app.route("/xcovid1", methods=['POST'])
def service1():
   
    respuesta = ""
    values = request.get_json()
    datos = values['img']
    print("DATOS ENVIADOS POR JSON:")
    #print(datos)
    deserializar(datos)
    prediccion = predecir1('img.jpg')

    respuesta = {'pred':prediccion}

    return json.dumps(respuesta)

#Convertimos la imagen a .jpg
def deserializar(imgstr):
    imgBytes = base64.b64decode(imgstr)
    filename = 'img.jpg'
    with open(filename, 'wb') as f:
        f.write(imgBytes)
        
 
        
def predict(file):
    x = load_img(file,target_size=(longitud,altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)
    if respuesta == 0:
        print('covid')
        respuesta1='covid'
    elif respuesta == 1:
        print('normal')
        respuesta1='normal'
    elif respuesta == 2:
        print('neumonia')
        respuesta1='neumonia'
        
        
    pre = str(respuesta1)    
        
    return pre
  
def predecir1(file):
    x = load_img(file,target_size=(longitud,altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arreglo = cnn1.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)
    if respuesta == 0:
        print('covid')
        resp = 'covid'
    elif respuesta == 1:
        print('normal')
        resp = 'normal'
    pro = str(resp)
    
    return pro
  
    
if __name__ == '__main__':
    app.run()

