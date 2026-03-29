import math
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.log_prior_ham = 0.0
        self.log_prior_spam = 0.0

        self.ham_word_counts = defaultdict(int)
        self.spam_word_counts = defaultdict(int)

        self.ham_word_probability = {}
        self.spam_word_probability = {}

        self.vocabulary = set()

    
    def train(self, x_train, y_train):
        num_of_ham_lines = y_train.count(0)
        num_of_spam_lines = y_train.count(1)

        self.log_prior_ham = math.log(num_of_ham_lines/len(y_train))
        self.log_prior_spam = math.log(num_of_spam_lines/len(y_train))

        for line, tag in zip(x_train, y_train):
            for word in line:
                self.vocabulary.add(word)

                if tag == 0:
                    self.ham_word_counts[word] +=1
                else:
                    self.spam_word_counts[word] +=1
        

        vocab_size = len(self.vocabulary)

        for word in self.vocabulary:
            self.ham_word_probability[word] = math.log((self.ham_word_counts[word] + 1)/(sum(self.ham_word_counts.values()) + vocab_size))
            self.spam_word_probability[word] = math.log((self.spam_word_counts[word] + 1)/(sum(self.spam_word_counts.values()) + vocab_size))


    def predict(self, x_test) -> list[int]:
        results = []

        for text in x_test:
            ham = self.log_prior_ham
            spam = self.log_prior_spam

            for word in text:
                if(word in self.ham_word_probability) and (word in self.spam_word_probability):
                    ham += self.ham_word_probability[word]
                    spam += self.spam_word_probability[word]

                #ham += self.ham_word_probability.get(word, self.default_ham_log_prob)
                #spam += self.spam_word_probability.get(word, self.default_spam_log_prob)
            
            if ham > spam:
                results.append(0)
            else:
                results.append(1)

        return results