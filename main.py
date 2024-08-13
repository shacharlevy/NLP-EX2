import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import SelectKBest


def classify(train_file, test_file):
    # todo: implement this function
    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')

    # todo: you can try working with various classifiers from sklearn:
    #  https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    #  https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #  please use the LogisticRegression classifier in the version you submit

    # todo: fill in the dictionary below with actual scores obtained on the test data
    test_results = {'class_1_F1': 0.0,
                    'class_2_F1': 0.0,
                    'class_3_F1': 0.0,
                    'class_4_F1': 0.0,
                    'class_5_F1': 0.0,
                    'accuracy': 0.0}

    train_reviews = load_data(train_file)
    test_reviews = load_data(test_file)

    train_texts = [review["reviewText"] + "  " + review.get("summary", "") for review in train_reviews]
    train_labels = [int(review.get("overall", 0)) for review in train_reviews]

    test_texts = [review["reviewText"] + "  " + review.get("summary", "") for review in test_reviews]
    test_labels = [int(review.get("overall", 0)) for review in test_reviews]

    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(train_features, train_labels)
    predicted_labels = model.predict(test_features)
    calculate_f1_accuracy(test_labels, predicted_labels, test_results)
    confusion_matrix = print_confusion_matrix(test_labels, predicted_labels)
    print_classes_with_highest_confusion(confusion_matrix)
    feature_names = vectorizer.get_feature_names_out()
    print_15_features_most_effective(train_features, train_labels, feature_names)

    return test_results


def load_data(file_path):
    with open(file_path, 'r') as file:
        json_data = file.read()
        data = []
        for line in json_data.split('\n'):
            if line.strip():
                current_json = json.loads(line)
                if "overall" in current_json and "reviewText" in current_json:
                    data.append(json.loads(line))
    return data


def calculate_f1_accuracy(test_labels, predicted_labels, test_results):
    f1_scores = metrics.f1_score(test_labels, predicted_labels, average=None)
    overall_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
    test_results['class_1_F1'] = f1_scores[0]
    test_results['class_2_F1'] = f1_scores[1]
    test_results['class_3_F1'] = f1_scores[2]
    test_results['class_4_F1'] = f1_scores[3]
    test_results['class_5_F1'] = f1_scores[4]
    test_results['accuracy'] = overall_accuracy


def print_classes_with_highest_confusion(confusion_matrix):
    misclass_sums = [sum(row) - confusion_matrix[i, i] for i, row in enumerate(confusion_matrix)]
    max_misclass_sum = max(misclass_sums)
    most_confused_classes = [i + 1 for i, value in enumerate(misclass_sums) if value == max_misclass_sum]

    print(f"The classes with the highest confusion are classes {most_confused_classes}.")


def print_confusion_matrix(test_labels, predicted_labels):
    confusion_matrix = metrics.confusion_matrix(test_labels, predicted_labels)
    print(confusion_matrix)
    return confusion_matrix


def print_15_features_most_effective(train_features, train_labels, features_name):
    k_best = SelectKBest(k=15)
    k_best.fit_transform(train_features, train_labels)
    selected_feature_indices = k_best.get_support(indices=True)
    selected_feature_names = [features_name[i] for i in selected_feature_indices]
    print("Top 15 features:")
    for feature_name in selected_feature_names:
        print(feature_name)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v)

