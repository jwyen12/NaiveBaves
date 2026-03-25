from nltk.stem import WordNetLemmatizer, PorterStemmer
import sklearn
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

class DataLoader:


    @staticmethod
    def preprocess(text):
        cleanText = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokenizedText = word_tokenize(cleanText)
        cleanTokenizedText = []

        for word in tokenizedText:
            cleanWord = WordNetLemmatizer.lemmatize(word)
            cleanWord = WordNetLemmatizer.stem(cleanWord)
            cleanTokenizedText.append(cleanWord)
            
        return cleanTokenizedText


    def load_data():
        print('hey')


    def split_data():
        print('hey')