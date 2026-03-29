import math
from collections import defaultdict

class NaiveBayes:
    #initial state with everything set to 0 or empty 
    def __init__(self):
        self.log_prior_ham = 0.0
        self.log_prior_spam = 0.0

        self.ham_word_counts = defaultdict(int)
        self.spam_word_counts = defaultdict(int)

        self.ham_word_probability = {}
        self.spam_word_probability = {}

        self.vocabulary = set()

    
    #training function that takes the training data and does the Naive Bayes calculations
    def train(self, x_train, y_train):
        num_of_ham_lines = y_train.count(0)
        num_of_spam_lines = y_train.count(1)

        self.log_prior_ham = math.log(num_of_ham_lines/len(y_train))
        self.log_prior_spam = math.log(num_of_spam_lines/len(y_train))

        #goes through and adds each word to the vocabulary then increments the ham or spam word count based on the tag
        for line, tag in zip(x_train, y_train):
            for word in line:
                self.vocabulary.add(word)

                if tag == 0:
                    self.ham_word_counts[word] +=1
                else:
                    self.spam_word_counts[word] +=1
        

        vocab_size = len(self.vocabulary)
        #The math to find P(word | ham/spam)
        #Includes +1 for smoothing to prevent log(0) errors
        for word in self.vocabulary:
            self.ham_word_probability[word] = math.log((self.ham_word_counts[word] + 1)/(sum(self.ham_word_counts.values()) + vocab_size))
            self.spam_word_probability[word] = math.log((self.spam_word_counts[word] + 1)/(sum(self.spam_word_counts.values()) + vocab_size))

    #prediction function that uses probabilites gathered in train() to make predictions 
    def predict(self, x_test) -> list[int]:
        results = []
        
        for text in x_test:
            ham = self.log_prior_ham
            spam = self.log_prior_spam

            #inner for loop that goes word by word and increases the ham or spam probability score based on that words individual probability of being in ham or spam
            for word in text:
                if(word in self.ham_word_probability) and (word in self.spam_word_probability):
                    ham += self.ham_word_probability[word]
                    spam += self.spam_word_probability[word]

            
            if ham > spam:
                results.append(0)
            else:
                results.append(1)

        return results
    