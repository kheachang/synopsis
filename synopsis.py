import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from scipy import spatial
import networkx as nx

nltk.download("punkt")

with open("text.txt", "r", encoding="utf-8") as file:
    text = file.read()

sentences = sent_tokenize(text)

sentences_clean = [re.sub(r"[^\w\s]", "", sentence.lower()) for sentence in sentences]

stop_words = stopwords.words("english")
sentence_tokens = [
    [words for words in sentence.split(" ") if words not in stop_words]
    for sentence in sentences_clean
]

w2v=Word2Vec(sentence_tokens,size=1,min_count=1,iter=1000)

sentence_embeddings=[[w2v[word][0] for word in words] for words in sentence_tokens]

max_len=max([len(tokens) for tokens in sentence_tokens])

sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]

