import uvicorn
import pickle
from fastapi import FastAPI
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

app = FastAPI()
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
data = np.load('emb_desc_names.npz',allow_pickle=True)
nm_lod = data['names']
desc_lod =  data['desc']



@app.get("/")
async def home():
    return "<h2>Search your hero</h2>"


@app.post("/predict")
async def predict_api(ip:str):
    ip_em = model([ip]).numpy()
    all_dist = (desc_lod @ ip_em.T )
    top5 = all_dist.flatten().argsort()[-5:]
    top5_res = str(nm_lod[top5])
    return {'prediction': top5_res}


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=4000, debug=True