import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

with open('text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

sentences = sent_tokenize(text)


