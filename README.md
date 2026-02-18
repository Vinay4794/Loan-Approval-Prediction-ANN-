# Loan-Approval-Prediction-ANN-
This project builds a Loan Approval Prediction System using an Artificial Neural Network (ANN) based on applicant financial and credit history.


# 📌 Overview

Loan approval is a crucial decision for financial institutions.
This project builds a machine learning pipeline that:

Performs data preprocessing & feature engineering

Uses correlation analysis for feature selection

Trains an ANN model for binary classification

Evaluates model performance using standard metrics


# 🎯 Objective

To develop a predictive model that accurately classifies whether a loan should be:

✅ Approved
❌ Rejected

based on key applicant attributes.


# 📂 Dataset

The dataset contains borrower information such as:

Income

Employment experience

Loan amount

Interest rate

Credit score

Credit history length

Previous loan defaults

Target Variable:

loan_status


# 🧠 Model Architecture

The Artificial Neural Network consists of:

Input Layer   → Selected financial & credit features
Hidden Layer  → 32 neurons (ReLU)
Hidden Layer  → 16 neurons (ReLU)
Output Layer  → 1 neuron (Sigmoid)


# ⚙️ Configuration

Optimizer → Adam

Loss Function → Binary Crossentropy

Evaluation Metric → Accuracy


# 🔄 Project Workflow

Data Collection
      ↓
Data Cleaning
      ↓
Label Encoding
      ↓
Feature Scaling (StandardScaler)
      ↓
Correlation Heatmap
      ↓
Feature Selection
      ↓
Train–Test Split
      ↓
ANN Model Training
      ↓
Model Evaluation


# 📊 Exploratory Data Analysis

EDA was performed to understand:

Class distribution

Feature relationships

Impact of financial attributes on loan approval

Key Visualizations

Correlation heatmap

Income vs loan status

Loan amount distribution

Credit score analysis

Training vs validation accuracy

Confusion matrix


# 📈 Model Performance

The model was evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

The ANN achieved strong performance in predicting loan approval status with balanced generalization on unseen data.


# 🛠️ Tech Stack

Programming Language: Python

Libraries:

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

TensorFlow / Keras


# 🚀 Installation
1️⃣ Clone the repository
git clone https://github.com/Vinay4794/Loan-Approval-Prediction-ANN-
cd loan-ann-prediction


# 📷 Results

🔹 Correlation Heatmap

(Shows the relationship between features and loan status)

🔹 Training vs Validation Accuracy

(Indicates model learning and generalization)

🔹 Confusion Matrix

(Displays classification performance)

Add screenshots in the images/ folder and link them here.


# 🔬 Key Learnings

Importance of feature scaling for ANN

Effect of correlated features on model performance

Handling categorical variables using label encoding

Preventing overfitting using validation monitoring


# 🌟 Future Improvements

Hyperparameter tuning

Dropout & Batch Normalization

K-Fold cross validation

Model deployment using Streamlit or Flask

Handling class imbalance


#🎓 Academic Relevance

This project showcases:

Deep learning for structured data

Financial risk prediction

End-to-end ML pipeline

Suitable for:

Final year major project

Deep learning portfolio

Research implementation


# 🤝 Contributing

Contributions are welcome!

If you’d like to improve this project:

Fork the repository

Create a new branch

Submit a pull request
