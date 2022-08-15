import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)

data = np.load('emb_desc_names.npz',allow_pickle=True)

nm_lod = data['names']
desc_lod =  data['desc']

ip = 'strong big man'

ip_em = model([ip]).numpy()


all_dist = (desc_lod @ ip_em.T )

top5 = all_dist.flatten().argsort()[-5:]

print(nm_lod[top5])

