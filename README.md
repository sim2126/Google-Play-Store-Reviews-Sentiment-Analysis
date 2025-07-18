# Google Play Store Reviews Sentiment Analysis

This project builds a machine learning pipeline to perform sentiment analysis on Google Play Store reviews. The project demonstrates data retrieval, preprocessing, model implementation (comparing three different approaches), and comprehensive evaluation.

## Core Functionality

1.  **Data Retrieval**: Fetches up to 50,000 of the most recent reviews for a specified Android app from the Google Play Store.
2.  **Model Comparison**: Implements and compares three distinct sentiment analysis models:
    * A rule-based baseline (VADER).
    * A traditional machine learning classifier (Logistic Regression with TF-IDF).
    * A pre-trained transformer model (DistilBERT).
3.  **Evaluation**: Conducts a proper train/test evaluation using a suite of metrics: accuracy, precision, recall, F1-score, and confusion matrices.
4.  **Analysis**: Provides a clear analysis of model performance to determine the most suitable model for the task.

## Technical Requirements & Implementation

### 1. Data Pipeline

* **Data Collection**: The `google-play-scraper` library is used to fetch reviews for the app 'com.instagram.android'.
* **Preprocessing**:
    * Text is cleaned by removing special characters and converting it to lowercase.
    * Empty reviews are handled by dropping them.
    * Review scores (1-5 stars) are mapped to sentiment labels:
        * **Positive**: 4-5 stars
        * **Neutral**: 3 stars
        * **Negative**: 1-2 stars
* **Train/Test Split**: The dataset is split into a 70% training set and a 30% testing set.

### 2. Model Implementation & Evaluation

* **Baseline Model (VADER)**: `vaderSentiment` is used as a zero-shot, rule-based classifier. It calculates a compound sentiment score which is then mapped to positive, neutral, or negative labels.
* **ML Model (Logistic Regression)**: A `LogisticRegression` classifier from `scikit-learn` is trained on TF-IDF features extracted from the review text. This model learns from the specific vocabulary of the app reviews.
* **Pre-trained Model (DistilBERT)**: A pre-trained `DistilBERT` model, fine-tuned for sentiment analysis, is used via the `transformers` library pipeline. This model leverages deep learning to understand language context and semantics.
* **Evaluation Metrics**: All models are evaluated on the test set using a weighted average for precision, recall, and F1-score to account for class imbalance. Accuracy and detailed classification reports are also generated.

## How to Run the Code

This project is designed to be run in a Google Colab environment.

1.  **Open in Colab**: Open a new Colab notebook.
2.  **Copy the Code**: Copy the Python code from the provided script into a single cell in the notebook.
3.  **Run the Cell**: Execute the cell. The script will automatically install dependencies, fetch and process data, train and evaluate all three models, and print a detailed performance comparison.

## Results Summary

The overall performance of the three models on the test set is summarized below.

| Model                          | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :----------------------------- | :------- | :------------------- | :---------------- | :------------------ |
| **VADER (Rule-Based)** | 0.696    | 0.829                | 0.696             | 0.749               |
| **Logistic Regression (TF-IDF)** | 0.866    | 0.835                | 0.866             | 0.846               |
| **DistilBERT (Transformer)** | 0.795    | 0.840                | 0.795             | 0.809               |

### Key Findings

* **Overall Performance**: The **Logistic Regression with TF-IDF** model achieved the best overall performance, with the highest accuracy (86.6%) and weighted F1-score (84.6%). This indicates it is the most effective model for the dataset as a whole.
* **The "Neutral" Challenge**: A key insight from the detailed classification reports is that all three models struggled significantly to correctly identify the `neutral` class. The F1-scores for this class were extremely low across the board (VADER: 0.08, Logistic Regression: 0.01, DistilBERT: 0.03). This is likely due to two factors:
    1.  **Class Imbalance**: The `neutral` class is much smaller than the `positive` and `negative` classes.
    2.  **Ambiguous Language**: Reviews with a 3-star rating often contain a mix of positive and negative language, making them inherently difficult to classify.
* **Model-Specific Behavior**:
    * The **VADER** model served as a reasonable, no-training-required baseline but was outperformed by the trained models.
    * **DistilBERT's** performance was hampered by being pre-trained on a binary (positive/negative) task, which explains its difficulty with the third `neutral` label.

### Conclusion

For a practical and effective sentiment analysis solution for this dataset, the **Logistic Regression model is the recommended choice**. Despite its poor performance on the neutral class, its excellent performance on the majority positive and negative classes gives it the highest overall accuracy and F1-score. It provides a robust and computationally efficient solution.

If improving the detection of neutral reviews were a primary goal, future work would need to focus on techniques to handle class imbalance, such as oversampling the neutral class (e.g., with SMOTE) or fine-tuning a transformer model like DistilBERT specifically on the three-class dataset.
