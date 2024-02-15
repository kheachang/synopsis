import nltk
from nltk.tokenize import sent_tokenize
import re
from nltk.corpus import stopwords

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
