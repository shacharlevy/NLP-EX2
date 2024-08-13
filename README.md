# NLP-EX2

Project Description
This project involves two primary tasks: analyzing the Viterbi algorithm and performing text classification on Amazon product reviews.

Task 1: Viterbi Algorithm Analysis
You will need to address the following points:

Optimal Solution Guarantee: Explain whether the Viterbi algorithm guarantees finding the most probable sequence of states given a sequence of observations, and how it achieves this.
Matrix Computation: Evaluate the impact of modifying the matrix computation during the forward pass of the algorithm. Specifically, contrast the effects of using:
(a) C(i, j) = max_k [C(k, j-1) * A(k, i) * B(i, ind(wj))]
(b) C(i, j) = max_k [A(k, i) * B(i, ind(wj))]
Provide a small example with two tags and two words to illustrate a case where the alternative (b) is insufficient compared to the original computation (a).


Task 2: Text Classification
This task involves predicting the ratings of Amazon product reviews, spanning from 1 to 5.

Data
Training Data: 2000 reviews per rating class.
Test Data: 400 reviews per rating class.
Requirements
Implement the function classify(train_data, test_data) to predict review ratings.
Metrics to Report: Provide the F1 score per class and overall accuracy.
Tooling: Use scikit-learn's classifiers and text vectorization tools (CountVectorizer, TfidfVectorizer).
Instructions
Feature Extraction: Reduce vocabulary to the top-K most frequent words or n-grams (K < 1000) to manage processing time.
Confusion Matrix: Analyze and interpret the confusion matrix, noting which classes are most confused.
Feature Selection: Utilize SelectKBest to extract and report the 15 most discriminative features.
Cross-Domain Classification: Train on one domain and test on another. Compare results and interpret any discrepancies.
Runtime: Ensure the code completes within 5 minutes.
Files Provided
config.json: Contains file locations for training and testing data.
main.py: Contains code for parsing config.json and calling your classify function.
Deliverables
A document with concise answers to Task 1.
Results and analysis for Task 2, including accuracy, confusion matrix, selected features, and cross-domain performance.
