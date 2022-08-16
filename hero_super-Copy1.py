
import enum
import pandas as pd
import re
import string, sys, time
from nltk.stem import PorterStemmer
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 

import nltk
nltk.download('punkt')

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


ip = 'hammer thunder lightening strong god of thunder'

def Token_new1(inpdata):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(inpdata)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    porter_stem=[ps.stem(word) for word in stopped_tokens]
    lemm=[LM.lemmatize(word) for word in stopped_tokens]    
    return(porter_stem+lemm)



print(top5)
