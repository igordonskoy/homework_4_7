import numpy as np

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    #image = np.array(list(map(int, image[2:].split(','))))
    #image = np.array(list(map(int, image['data'].split(','))))
    #image = np.array(image)
    image = np.array(list(map(int, image.split(','))))
    #print(image)
    #print(image.shape)
    #print(type(image))
    pred = model.predict(image[1:])
    return {'prediction': pred}
    #print(pred)
    #return int(pred)

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
