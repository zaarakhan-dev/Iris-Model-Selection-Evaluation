# Iris Species Classification: Model Selection and Evaluation

## Project Overview

This project focuses on comparing multiple machine learning classification algorithms using the Iris Species Dataset. The objective is to evaluate different models, compare their performance using standard evaluation metrics, and select the most suitable model for classifying iris flower species.

The project demonstrates the complete machine learning workflow, including data loading, train-test splitting, model training, evaluation, cross-validation, and model comparison.

---

## Objective

The main goal of this project is to determine which machine learning algorithm performs best for classifying iris flower species.

Models Compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

---

## Dataset Information

The Iris dataset is a well-known classification dataset available in Scikit-learn.

### Features

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Target Classes

* Iris Setosa
* Iris Versicolor
* Iris Virginica

### Dataset Statistics

* Total Samples: 150
* Features: 4
* Classes: 3

The dataset contains no missing values and is suitable for supervised classification tasks.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Machine Learning Workflow

### 1. Data Loading

The Iris dataset was loaded using Scikit-learn.

### 2. Data Splitting

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

A random state of 42 was used to ensure reproducibility.

### 3. Model Training

The following models were trained:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### 4. Model Evaluation

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

### 5. Cross Validation

5-Fold Cross Validation was performed to evaluate model generalization and reduce the risk of overfitting.

---

## Results

### Logistic Regression

* Accuracy: 1.00
* Precision: 1.00
* Recall: 1.00
* F1 Score: 1.00
* Cross Validation Accuracy: 97.33%

### Decision Tree

* Accuracy: 1.00
* Precision: 1.00
* Recall: 1.00
* F1 Score: 1.00
* Cross Validation Accuracy: 95.33%

### Random Forest

* Accuracy: 1.00
* Precision: 1.00
* Recall: 1.00
* F1 Score: 1.00
* Cross Validation Accuracy: 96.67%

---

## Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | Cross Validation Accuracy |
| ------------------- | -------- | --------- | ------ | -------- | ------------------------- |
| Logistic Regression | 1.00     | 1.00      | 1.00   | 1.00     | 97.33%                    |
| Decision Tree       | 1.00     | 1.00      | 1.00   | 1.00     | 95.33%                    |
| Random Forest       | 1.00     | 1.00      | 1.00   | 1.00     | 96.67%                    |

---

## Best Model

Based on cross-validation performance, **Logistic Regression** was selected as the best model.

### Reasons

* Highest Cross Validation Accuracy (97.33%)
* Perfect Accuracy, Precision, Recall, and F1 Score
* Simple and efficient algorithm
* Strong generalization performance

---

## Project Structure

```text
Iris-Model-Selection-Evaluation/

│
├── README.md
├── model_selection.ipynb
├── model_selection.py
├── Model_Selection_Report.docx
└── images/
    ├── confusion_matrix.png
    └── model_comparison.png
```

---

## How to Run

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run Python Script

```bash
python model_selection.py
```

### Run Notebook

Open:

```text
model_selection.ipynb
```

and execute all cells.

---

## Key Learning Outcomes

* Understanding classification problems
* Training multiple machine learning models
* Evaluating models using Accuracy, Precision, Recall, and F1 Score
* Applying Cross Validation
* Comparing algorithms objectively
* Selecting the best-performing model

---

## Conclusion

This project successfully compared three machine learning classification algorithms on the Iris Species Dataset. While all models achieved perfect performance on the test set, cross-validation revealed that Logistic Regression provided the most reliable results with a cross-validation accuracy of 97.33%.

The project highlights the importance of evaluating multiple models and using cross-validation when selecting the best machine learning algorithm.
