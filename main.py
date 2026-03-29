from EvaluationMetrics import EvaluationMetrics
from DataLoader import DataLoader
from NaiveBayes import NaiveBayes


def main():
    x, y = DataLoader.load_data("SMSSpamCollection.txt")
    x_train, x_test, y_train, y_test = DataLoader.split_data(x,y)

    #create a NaiveBayes object and then train it on the training data
    model = NaiveBayes()
    model.train(x_train, y_train)

    #run the predictions on the test and training data
    testResults = model.predict(x_test)
    trainingResults = model.predict(x_train)

    testMetrics = EvaluationMetrics.compute_metrics(testResults, y_test)
    trainingMetrics = EvaluationMetrics.compute_metrics(trainingResults, y_train)
    
    #output format for results.log
    output = [
        "Naive Bayes Spam — Results Log",

        "\n--- Dataset ---",
        f"  Total samples    : {len(y)}",
        f"  Training samples : {len(y_train)}",
        f"  Testing samples  : {len(y_test)}",

        "\n--- Training Metrics ---",
        f"  Accuracy  : {trainingMetrics["accuracy"]*100:.2f}%",
        f"  Precision : {trainingMetrics["precision"]*100:.2f}%",
        f"  Recall    : {trainingMetrics["recall"]:.2f}",
        f"  F1 Score  : {trainingMetrics["F1"]:.2f}",
        f"  TP : {trainingMetrics["TP"]}",
        f"  TN : {trainingMetrics["TN"]}",
        f"  FP : {trainingMetrics["FP"]}",
        f"  FN : {trainingMetrics["FN"]}",

        "\n--- Testing Metrics ---",
        f"  Accuracy  : {testMetrics["accuracy"]*100:.2f}%",
        f"  Precision : {testMetrics["precision"]*100:.2f}%",
        f"  Recall    : {testMetrics["recall"]:.2f}",
        f"  F1 Score  : {testMetrics["F1"]:.2f}",
        f"  TP : {testMetrics["TP"]}",
        f"  TN : {testMetrics["TN"]}",
        f"  FP : {testMetrics["FP"]}",
        f"  FN : {testMetrics["FN"]}",
    ]

    output = "\n".join(output)
    print(output)

    #write the output to the results file
    with open("results.log", "w") as file:
        file.write(output)


if __name__ == "__main__":
    main()
