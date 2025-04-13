Offensive Language Classification

## Project Overview

This project focuses on detecting and classifying various types of offensive language in user feedback. It is a multi-label classification task, where a single comment can be labeled with one or more categories such as toxic, abusive, vulgar, menace, offense, and bigotry.

We implemented and evaluated the following models:

    - Baseline Models: Logistic Regression and Random Forest

    - Advanced Model: LSTM

    - Transformer Model: BERT

## Dataset Description

File: train.csv
This dataset contains labeled user feedback with annotations for six types of offensive language.

Columns:

    id: Unique identifier for each comment

    feedback_text: The feedback text to be classified

    toxic: 1 if the feedback is toxic

    abusive: 1 if it contains abusive language

    vulgar: 1 if it contains vulgar/obscene content

    menace: 1 if it includes threats

    offense: 1 if it includes insulting language

    bigotry: 1 if it includes identity-based hate speech

Each feedback entry can belong to one or more of these categories.

## Model Implementation Details
1. Baseline Models - Logistic Regression & Random Forest

    - Text preprocessing: Lowercasing, punctuation and stop word removal

    - Feature extraction using TF-IDF

    - Multi-label handled with One-vs-Rest strategy

    - Fast and interpretable; Logistic Regression gives solid baseline, Random Forest adds non-linearity

2. Advanced Model - LSTM

    - Tokenized and padded sequences

    - Embedding + LSTM layers (Keras)

    - Binary cross-entropy loss with sigmoid activation

3. Transformer Model - BERT

    - Fine-tuned bert-base-uncased using Hugging Face Transformers

    - Preprocessing includes tokenization using BERT tokenizer

    - Multi-label sigmoid output and optimized using AdamW with learning rate scheduling

## Steps to Run the Code

    Clone the Repository

git clone https://github.com/yourusername/offensive-language-classification.git

cd offensive-language-classification

    Run the Notebooks

    Baseline (Logistic Regression & Random Forest) and LSTM:
    Code is in task/model1_implementation.ipynb or .py

    Transformer Model (BERT):
    Code is in task/model2_implementation.ipynb or .py

Use Jupyter Notebook or Colab to execute.
## Model Evaluation Results
| Metric           | Logistic Regression | Random Forest     | LSTM   | BERT       |
|------------------|---------------------|-------------------|--------|------------|
| Macro F1-Score   | Moderate            | Moderate          | Higher | Highest    |
| AUC-ROC          | Fair                | Fair              | Good   | Excellent  |
| Recall (avg)     | Lower               | Lower             | Better | Best       |
| Precision (avg)  | Reasonable          | Slightly better   | High   | Very High  |



    Evaluation includes Confusion Matrix, ROC Curve, Precision, Recall, F1-Score.

    BERT outperforms all other models, especially in detecting rare labels like menace and bigotry.

For more detailed visualizations and metric analysis (including ROC curves, confusion matrices, and class-wise performance), please refer to the  
📄 **[Performance_Analysis_Report.pdf](./Performance_Analysis_Report.pdf)** available in the repository root.

## Additional Observations

- Class imbalance affected the minority label detection.

- TF-IDF + Logistic Regression offers a fast, explainable baseline.

- Random Forest can capture non-linear relationships but may overfit.

- LSTM captures the sequence of text for better generalization.

- BERT is the most effective due to its contextual understanding but requires more computation.

- Future work could include data augmentation for further performance gain.



Author

Marzia Sultana

Data Science Enthusiast

email: marziasultanar01@gmail.com
