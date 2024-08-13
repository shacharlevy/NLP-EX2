# NLP-EX2

This repository contains an assignment with two main tasks: analyzing the Viterbi decoding algorithm and performing text classification on Amazon product reviews. This README provides a detailed description of each task, implementation requirements, and evaluation criteria.

## Table of Contents

1. [Task 1: Viterbi Algorithm Analysis](#task-1-viterbi-algorithm-analysis)
2. [Task 2: Text Classification](#task-2-text-classification)
3. [Implementation Details](#implementation-details)
4. [Files Provided](#files-provided)
5. [Deliverables](#deliverables)
6. [Running the Code](#running-the-code)

## Task 1: Viterbi Algorithm Analysis

### Overview

In this task, you will analyze the Viterbi algorithm used for sequence decoding and address the following:

#### Optimal Solution Guarantee

**Objective:** Explain whether the Viterbi algorithm guarantees finding the most probable sequence of states given a sequence of observations. Discuss how the algorithm’s structures and computations ensure this guarantee.

#### Matrix Computation

**Objective:** Compare the effects of two matrix computation approaches during the forward pass:
- (a) `C(i, j) = max_k [C(k, j-1) * A(k, i) * B(i, ind(wj))]`
- (b) `C(i, j) = max_k [A(k, i) * B(i, ind(wj))]`

**Example:** Provide a small example with two tags and two words to illustrate a case where computation (b) is insufficient compared to (a).

## Task 2: Text Classification

### Overview

This task involves building a classifier to predict Amazon product review ratings, which range from 1 to 5 stars.

#### Data

- **Training Data:** 2000 reviews per rating class.
- **Test Data:** 400 reviews per rating class.

#### Requirements

**Function Implementation:**

- Implement the `classify(train_data, test_data)` function to predict review ratings based on provided data.

**Metrics:**

- Report the F1 score for each class and overall accuracy.

**Tooling:**

- Utilize scikit-learn’s classifiers and text vectorization tools (`CountVectorizer`, `TfidfVectorizer`).

#### Instructions

**Feature Extraction:**

- Limit the vocabulary to the top-K most frequent words or n-grams (K < 1000) to optimize processing time.

**Confusion Matrix:**

- Analyze and interpret the confusion matrix to identify and explain which classes are most confused.

**Feature Selection:**

- Use `SelectKBest` to extract and report the 15 most effective features for classification.

**Cross-Domain Classification:**

- Train the classifier on one domain and test it on another. Compare and interpret the results.

**Runtime:**

- Ensure that the code execution time does not exceed 5 minutes.

## Implementation Details

### Code Structure

- **`config.json`** – Configuration file for specifying the locations of training and testing data. You may modify this file but should not alter the parsing code in `main.py`.
- **`main.py`** – Main script for executing the classification task.

### Libraries

- **scikit-learn** – For classifiers, vectorization, and feature selection.
- **NumPy/Pandas** – For data manipulation and analysis.

## Files Provided

- **`config.json`** – Contains file locations for data.
- **`main.py`** – Script for running the classification task.

## Deliverables

- **Task 1 Report:** Document with concise answers to Viterbi algorithm questions.
- **Task 2 Results:**
  - F1 scores for each class.
  - Overall accuracy.
  - Confusion matrix and interpretation.
  - Report of the top 15 features.
  - Analysis of cross-domain classification results.

## Running the Code

### Setup Environment

- Ensure Python and necessary libraries are installed.

### Run the Script

- Modify `config.json` if necessary.
- Execute `main.py` to run the classification task and generate results.

### Evaluation

- Check the output to ensure results meet required metrics and runtime constraints.

For questions or issues, please open an issue in this repository.
