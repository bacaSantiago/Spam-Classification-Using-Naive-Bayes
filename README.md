# Spam Classification Using Naive Bayes: An Applied Machine Learning Approach

---

## Overview

This project presents a complete system for spam classification using a Naive Bayes classifier, applied to a dataset of SMS messages. The classifier, built using Python, is trained to distinguish between spam and legitimate (ham) messages based on textual content. The **SMS Spam Collection** dataset from the UCI Machine Learning Repository is used to evaluate the model's performance in terms of **accuracy**, **precision**, **recall**, and the **Receiver Operating Characteristic (ROC) curve**. The implementation demonstrates the effectiveness of Naive Bayes in real-world spam detection applications.

The project leverages several **mathematical** and **computational** concepts, such as, Bayes' Theorem, Conditional Independence, Probabilistic Classifiers, Text Vectorization (Bag of Words model), Model Evaluation Metrics (Accuracy, Precision, Recall, AUC), and, Confusion Matrix Analysis. Moreover, the project relies on robust **machine learning libraries** and **text processing tools** in Python, making it scalable for various natural language processing (NLP) tasks, especially in environments where fast and lightweight models are necessary.

---

## Introduction

Spam messages in digital communications pose a significant challenge due to the sheer volume and variety of unwanted content. Traditional rule-based systems struggle to keep up with evolving spam techniques, requiring more advanced methods to automate the identification process. This project implements a **Multinomial Naive Bayes** classifier, tailored specifically for text classification problems, such as **SMS spam detection**.

The project focuses on several core goals:

1. **Dataset Exploration**: A thorough examination of the dataset to understand the distribution of spam and ham messages.
2. **Text Preprocessing**: Converting raw text into numerical representations using **Bag of Words** and **CountVectorizer**, enabling the classifier to process and learn from the data.
3. **Training a Classifier**: The Naive Bayes classifier is trained on the dataset, capitalizing on its ability to work well with high-dimensional data, such as word frequencies.
4. **Performance Evaluation**: We measure the effectiveness of the classifier using standard metrics, including **accuracy**, **precision**, **recall**, and **AUC**, to assess its ability to identify spam with minimal false positives.

This project provides a comprehensive analysis of the Naive Bayes model and explores its advantages and limitations in spam classification tasks.

---

## Theoretical Framework

### 1. **Naive Bayes Classifier:**

The **Naive Bayes** classifier is based on **Bayes’ Theorem**, which calculates the posterior probability of a class given some observed features. The "naive" assumption implies that all features (in this case, words) are conditionally independent of each other given the class label (spam or ham). Despite this strong assumption, Naive Bayes has proven effective for **text classification** tasks like spam detection, where word frequencies are used to classify messages.

- **Bayes’ Theorem** is expressed as:
  $$ P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)} $$

  Where:
  - $P(C|X)$ is the posterior probability of class $C$ given features $X$.
  - $P(X|C)$ is the likelihood of the features $X$ given the class $C$.
  - $P(C)$ is the prior probability of class $C$.
  - $P(X)$ is the prior probability of the features.

### 2. **Vectorization (Bag of Words Model):**

Before applying the Naive Bayes algorithm, raw text data is converted into a format that the model can understand. **Text vectorization** transforms each SMS message into a numerical vector based on word frequencies. The **CountVectorizer** function from Scikit-learn is used to perform this transformation. Each word in the dataset is treated as a feature, and the frequency of each word is used to classify the message as either spam or ham.

### 3. **Multinomial Naive Bayes:**

This specific variant of the Naive Bayes classifier is well-suited for discrete data, particularly word counts in text. It calculates the probability of a message being spam or ham based on the frequencies of the words it contains.

---

## Methodology

1. **Dataset Preparation**:
   - The dataset is split into **training** and **testing** sets using an 80-20 ratio to ensure a balanced evaluation of the model.
   - The text is converted into a **Bag of Words** format, where each message is represented by a vector of word frequencies.

2. **Training**:
   - The **Multinomial Naive Bayes** classifier is trained using the word frequency vectors from the training set.
   - The model learns the probability distributions of words in spam and ham messages, allowing it to make predictions on unseen data.

3. **Evaluation**:
   - After training, the classifier is tested on the remaining 20% of the dataset, and its performance is evaluated using:
     - **Accuracy**: The proportion of correctly classified messages.
     - **Precision**: The proportion of true spam messages among those classified as spam.
     - **Recall**: The proportion of actual spam messages that were correctly identified.
     - **AUC**: The area under the ROC curve, which shows the model's performance in distinguishing between spam and ham across different thresholds.

---

## Dependencies

This project uses the following libraries and tools:

- **Python 3.x**: The core programming language used for implementing the classifier and handling data.
- **NumPy**: Essential for numerical operations, especially in vectorizing text data.
- **Pandas**: Used for loading and manipulating the dataset.
- **Scikit-learn**: Provides tools for vectorization (CountVectorizer), implementing the **Multinomial Naive Bayes** classifier, and evaluating the model through metrics like **accuracy**, **precision**, **recall**, and **ROC curves**.
- **Matplotlib**: For visualizing the **confusion matrix** and **ROC curve**.
- **Seaborn**: Enhances the visual presentation of the confusion matrix.
- **SMS Spam Collection Dataset**: The dataset, available from the UCI Machine Learning Repository, contains 5572 messages labeled as either **spam** or **ham**.

---

## Resources

- **SMS Spam Collection Dataset**:  [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) from the **UCI Machine Learning Repository**.
- **Documentation for Scikit-learn**: [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- **Matplotlib and Seaborn for Visualization**: Essential for plotting the confusion matrix and ROC curve.

---

## Summary

This project successfully demonstrates the use of a **Naive Bayes classifier** for spam detection, emphasizing the power of **Bayes' Theorem** and **probabilistic reasoning** in text classification tasks. The **Bag of Words** model provides an efficient means to represent text data, allowing the classifier to make quick and accurate predictions. Further improvements can be made by experimenting with advanced preprocessing techniques, such as **TF-IDF vectorization**, or by integrating more complex models. This project serves as a foundational example of **machine learning** applied to a real-world problem, with potential applications in a wide range of **natural language processing** tasks.
