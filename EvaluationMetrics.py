class EvaluationMetrics:

    @staticmethod
    #Computes all result metrics
    #Args: results list returned from predict() and the y_test which is the key
    def compute_metrics(results, y_test):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        #iterates the prediction and answer together and compares
        for result, answer in zip(results, y_test):
            if result == 0:
                if result != answer:
                    FN += 1
                else:
                    TN += 1
            else: 
                if result != answer:
                    FP += 1
                else:
                    TP += 1
        
        accuracy = (TP+TN)/(TP+TN+FN+FP)

        precision = TP / (TP+FP) if(TP+FP) > 0 else 0

        recall = TP / (TP+FN) if(TP+FN) > 0 else 0

        F1 = 2 * ((precision * recall) / (precision + recall))

        #returns a dictonary with all values
        return {
            "TP":        TP,
            "TN":        TN,
            "FP":        FP,
            "FN":        FN,
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "F1":        F1,
        }
    