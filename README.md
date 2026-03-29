# Naive Bayes Spam Classifier

This is a simple spam classifier built from scratch in Python using the Naive Bayes algorithm. It classifies SMS messages as either spam or ham based on word probabilities.

## Overview

The model is trained on the SMS Spam Collection dataset. It processes each message, learns how often words appear in spam vs ham, and then uses those probabilities to classify new messages.

No machine learning libraries are used for the model itself.

## How it works

- Messages are cleaned (lowercased, non-letters removed)
- Text is split into words and stemmed
- Word counts are tracked separately for spam and ham
- Probabilities are calculated using Laplace smoothing
- Log probabilities are used to avoid underflow
- Predictions are made by comparing total scores for each class

## Files

- `DataLoader.py`  
  Loads the dataset, preprocesses text, and splits data into training and testing sets

- `NaiveBayes.py`  
  Contains the implementation of the Naive Bayes classifier

- `EvaluationMetrics.py`  
  Calculates accuracy, precision, recall, and F1 score

- `main.py`  
  Runs training, testing, and outputs results

- `results.log`  
  Example output from running the model

## Results

On the SMS dataset:

- Test Accuracy: 98.92%  
- Precision: 97.69%  
- Recall: 0.93  
- F1 Score: 0.95  

## How to run

1. Install dependencies:
 ```pip install -r requirements.txt```

2. Make sure the dataset file (`SMSSpamCollection.txt`) is in the project directory

3. Run: 
```python main.py```


This will train the model and print results to the console, and also write them to `results.log`.

## Notes

- The classifier is implemented from scratch for learning purposes  
- Only basic preprocessing is used (no advanced NLP features)  
- The focus is on understanding how Naive Bayes works internally  