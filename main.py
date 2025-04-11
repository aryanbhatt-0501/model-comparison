from src.preprocessing import load_data, preprocess_data, get_train_test_data;
from src.models import train_random_forest, evaluate_model;
import csv
import os
from src.preprocessing import load_data, preprocess_data, get_train_test_data
from src.models import (
    train_random_forest,
    train_logistic_regression,
    train_svm,
    train_knn,
    train_naive_bayes,
    evaluate_model,
)
import os
import csv

def save_results_to_csv(model_name, metrics, file_path="results/results.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
        writer.writerow([
            model_name,
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['Precision']:.4f}",
            f"{metrics['Recall']:.4f}",
            f"{metrics['F1 Score']:.4f}",
        ])

def main():
    # Load and preprocess data
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = get_train_test_data(X, y)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")

    # List of models to train
    models = [
        ("Random Forest", train_random_forest),
        ("Logistic Regression", train_logistic_regression),
        ("Support Vector Machine", train_svm),
        ("K-Nearest Neighbors", train_knn),
        ("Naive Bayes", train_naive_bayes),
    ]

    # Train, evaluate, and save results for each model
    for name, trainer in models:
        print(f"\n🔧 Training {name}...")
        model = trainer(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        print(f"\n📊 {name} Evaluation Metrics:")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1 Score: {metrics['F1 Score']:.4f}")

        save_results_to_csv(name, metrics)

if __name__ == "__main__":
    main()



# def save_results_to_csv(model_name, metrics, file_path="results/results.csv"):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     file_exists = os.path.isfile(file_path)
#     with open(file_path, mode="a", newline="") as csvfile:
#         writer = csv.writer(csvfile)
        
#         if not file_exists:
#             writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1 Score"])

#         writer.writerow([
#             model_name,
#             f"{metrics['Accuracy']:.4f}",
#             f"{metrics['Precision']:.4f}",
#             f"{metrics['Recall']:.4f}",
#             f"{metrics['F1 Score']:.4f}",
#         ])

# def main():
#     df = load_data("data/Telco_customer_churn.csv");
#     X, y = preprocess_data(df);
#     # print("y value counts:\n", y.value_counts(dropna=False))

#     X_train, X_test, y_train, y_test = get_train_test_data(X, y)
    
#     print(f"Training samples: {X_train.shape[0]}");
#     print(f"Testing samples: {X_test.shape[0]}");

#     model = train_random_forest(X_train, y_train);
#     results = evaluate_model(model, X_test, y_test);

#     print("\n📊 Model Evaluation Metrics:")
#     for metric, value in results.items():
#         if metric != "Report":
#             print(f"{metric}: {value:.4f}");
    
#     metrics = evaluate_model(model, X_test, y_test)
#     save_results_to_csv("Random Forest", metrics)



# if __name__ == "__main__":
#     main()


# df = load_data()
# X, y = preprocess_data(df)
# X_train, X_test, y_train, y_test = get_train_test_data(X, y)
