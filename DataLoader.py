from nltk.stem import PorterStemmer
import random
import re

class DataLoader:


    @staticmethod
    #function that cleans and tokenizes text
    def preprocess(text):
        #initial cleaning that removes special characters and tokenizes 
        cleanText = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokenizedText = cleanText.split()
        cleanTokenizedText = []
        stemmer = PorterStemmer()

        #goes through the inital tokenized list and applys stemming 
        for word in tokenizedText:
            cleanWord = stemmer.stem(word)
            cleanTokenizedText.append(cleanWord)
            
        return cleanTokenizedText


    @staticmethod
    #helper function that takes ham or spam and converts it into its correct tag
    def convert_ham_or_spam(candidate):
        if candidate == 'ham':
            return 0
        if candidate == 'spam':
            return 1
        else:
            print("Incorrect identifier")
            exit(1)


    @staticmethod
    #takes the text document and processes it into an array of tags and clean tokenized lines
    def load_data(filepath):
        x = []
        y = []
        with open(filepath) as file:
            #goes line by line and runs preprocess() then seperates out the tag and text
            #then appends the tag and text to there respective lists
            for line in file:
                cleanTokenizedText = DataLoader.preprocess(line)
                label = DataLoader.convert_ham_or_spam(cleanTokenizedText[0])

                y.append(label)
                cleanTokenizedText.pop(0)
                x.append(cleanTokenizedText)

        return x,y


    @staticmethod
    #splits the data into training and test based on the given parameter
    def split_data(x, y, test_ratio=0.2):
        # Combine x and y so they stay aligned
        data = list(zip(x, y))
        random.shuffle(data)
        
        # Split index
        split_index = int(len(data) * (1 - test_ratio))
        
        # Split data
        train_data = data[:split_index]
        test_data = data[split_index:]
        
        # Unzip back into x and y
        x_train, y_train = zip(*train_data)
        x_test, y_test = zip(*test_data)
        
        return list(x_train), list(x_test), list(y_train), list(y_test)
    