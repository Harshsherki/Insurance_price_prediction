# Insurance Charges Prediction

This repository contains a project for predicting insurance charges based on various factors using multiple regression techniques.

## Dataset

The dataset used in this project is from Kaggle and can be found [here](https://www.kaggle.com/mirichoi0218/insurance?select=insurance.csv).

## Data Preprocessing

The preprocessing steps include:

1. Importing necessary libraries.
2. Loading the dataset.
3. Data exploration and analysis.
4. Handling missing values (if any).
5. Encoding categorical data.
6. Feature scaling.

## Model Building

Three different regression models are built to predict insurance charges:

1. Multiple Linear Regression
2. Random Forest Regression
3. XGBoost Regression

### Steps:

1. Split the data into training and testing sets.
2. Apply feature scaling.
3. Train and evaluate the models.
4. Compare the performance of the models using the R-squared score.

## Results

| Model                      | R-squared Score |
|----------------------------|-----------------|
| Multiple Linear Regression | 0.7999          |
| Random Forest Regression   | 0.8816          |
| XGBoost Regression         | 0.8954          |

