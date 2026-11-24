

# Tata Steel Machine Failure Prediction

## Project Overview
This project uses Machine Learning to predict potential machine failures based on operating conditions. The goal is to support predictive maintenance, reduce unexpected downtime, and improve operational efficiency.

## Problem Statement
Unexpected machine failures can lead to production delays, maintenance costs, and reduced productivity. This project builds a classification model to identify potential machine failures using machine operating data.

## Dataset
The dataset contains machine operating information, including:
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type
- Machine Failure

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Project Workflow
1. Data Loading and Cleaning
2. Exploratory Data Analysis (EDA)
3. Data Visualization
4. Hypothesis Testing
5. Feature Engineering
6. Handling Class Imbalance
7. Model Building
8. Model Evaluation
9. Model Selection
10. Streamlit Deployment

## Machine Learning Models
Three classification algorithms were implemented:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

Random Forest was selected as the final model based on its validation precision and F1-score.

## Model Evaluation
The selected Random Forest model achieved:
- Accuracy: 99%
- Precision (Failure Class): 61%
- Recall (Failure Class): 30%
- F1-score (Failure Class): 40%

Note: The dataset is highly imbalanced. The model may miss some actual machine failures, so further recall improvement is recommended before real-world use.

## Feature Engineering
Two new features were created:
- Temperature Difference
- Mechanical Power

## Deployment
The trained Random Forest model was deployed using Streamlit Community Cloud.

## Future Improvements
- Improve failure detection recall.
- Perform additional hyperparameter tuning.
- Explore threshold optimization.
- Monitor model performance with new machine data.

## Author
Data Analytics and Machine Learning Project




## Live Application

Try the deployed Machine Failure Prediction application:

[Open Tata Steel Machine Failure Prediction App](https://tata-steel-machine-failure-prediction-abwecebl5t7yhmbliqfugg.streamlit.app)

