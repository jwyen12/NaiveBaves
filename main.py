from EvaluationMetrics import EvaluationMetrics
from DataLoader import DataLoader
from NaiveBayes import NaiveBayes


def main():
    x, y = DataLoader.load_data("resources/SMSSpamCollection.txt")
    x_train, x_test, y_train, y_test = DataLoader.split_data(x,y)

    model = NaiveBayes()
    model.train(x_train, y_train)

    testResults = model.predict(x_test)
    trainingResults = model.predict(x_train)

    testMetrics = EvaluationMetrics.compute_metrics(testResults, y_test)
    trainingMetrics = EvaluationMetrics.compute_metrics(trainingResults, y_train)

    output = [
        "Naive Bayes Spam — Results Log",

        "\n--- Dataset ---",
        f"  Total samples    : {len(y)}",
        f"  Training samples : {len(y_train)}",
        f"  Testing samples  : {len(y_test)}",

        "\n--- Training Metrics ---",
        f"  Accuracy  : {trainingMetrics.accuracy:.2f}",
        f"  Precision : {trainingMetrics.precision:.2f}",
        f"  Recall    : {trainingMetrics.recall:.2f}",
        f"  F1 Score  : {trainingMetrics.f1:.2f}",
        f"  TP : {trainingMetrics.TP}",
        f"  TN : {trainingMetrics.TN}",
        f"  FP : {trainingMetrics.FP}",
        f"  FN : {trainingMetrics.FN}",

        "\n--- Testing Metrics ---",
        f"  Accuracy  : {testMetrics.accuracy:.2f}",
        f"  Precision : {testMetrics.precision:.2f}",
        f"  Recall    : {testMetrics.recall:.2f}",
        f"  F1 Score  : {testMetrics.f1:.2f}",
        f"  TP : {testMetrics.TP}",
        f"  TN : {testMetrics.TN}",
        f"  FP : {testMetrics.FP}",
        f"  FN : {testMetrics.FN}",
    ]
    output = "\n".join(output)

    with open("results.log", "w") as file:
        file.write(output)

    if __name__ == "__main__":
        main()
