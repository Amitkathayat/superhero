import uvicorn
from fastapi import FastAPI

import pandas as pd
import re
from nltk.stem import PorterStemmer
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

Dframe1=pd.read_csv('nm_desc.csv')


def cleaning1(inpdata):
    cleanedArticle1=re.sub(r'[?|$|(),"".@#=><|!]Â&*/',r' ',inpdata)
    cleanedArticle2=re.sub(r'[^a-z A-Z]',r' ',cleanedArticle1)
    cleanedArticle3=cleanedArticle2.lower()
    cleanedArticle4=re.sub(r'\b\w{1,3}\b', ' ',cleanedArticle3)
    cleanedArticle5=re.sub(r'https?://\S+|www\.\S+',r' ',cleanedArticle4)
    cleanedArticle6=re.sub(r' +', ' ',cleanedArticle5)
    return(cleanedArticle6)

Dframe1['description']=Dframe1['description'].apply(cleaning1)

en_stop = get_stop_words('en')
ps = PorterStemmer()
LM=WordNetLemmatizer()
def Token_new1(inpdata):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(inpdata)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    porter_stem=[ps.stem(word) for word in stopped_tokens]
    lemm=[LM.lemmatize(word) for word in stopped_tokens]    
    return(porter_stem+lemm)

def argsort(seq):
    return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]

app = FastAPI()

@app.get("/")
async def home():
    return "<h2>Search your hero</h2>"


@app.post("/predict")
async def predict_api(ip:str):
    ip_tok = set(Token_new1(ip))
    i = 0
    counts = []
    for txt in Dframe1['description']:
        j=0
        for pat in ip_tok:
            match = re.findall(pat, txt) 
            j+=len(match)
        counts.append(j)
        i +=1

    args =argsort(counts)[-5:]
    top5=''
    for k,i in enumerate( args[::-1]):
        top5+=(str(k+1) + '. ' + Dframe1['name'][i] + " ")
    return {'Top 5 results': top5}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000, debug=True)
