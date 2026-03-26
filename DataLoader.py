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


    @staticmethod
    def convert_ham_or_spam(canidate):
        if canidate == 'ham':
            return 0
        if canidate == 'spam':
            return 1
        else:
            print("Incorrect identifier")
            exit(1)


    @staticmethod
    def load_data(filepath):
        x = []
        y = []
        i = 0
        with open(filepath) as file:
            for line in file:
                cleanTokenizedText = DataLoader.preprocess(line)
                label = DataLoader.convert_ham_or_spam(cleanTokenizedText[0])

                y[i] = label
                cleanTokenizedText.pop(0)
                x[i] = cleanTokenizedText

        
    def split_data():
        print('hey')