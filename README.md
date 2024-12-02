# COVID-19 Case Prediction Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Building](#model-building)
5. [Evaluation and Results](#evaluation-and-results)

## Project Overview
This project leverages Random Forest and ARIMA models to predict COVID-19 cases, providing valuable insights for public health monitoring and forecasting. The process includes exploratory data analysis (EDA), data wrangling, model training, evaluation, and visualization.

## Dataset Description

The dataset contains information about COVID-19 cases, including confirmed cases, new cases, recoveries, and deaths from various countries and regions. Key features include:

- **NewCases**: Number of new confirmed COVID-19 cases.
- **ConfirmedCases**: Total cumulative confirmed cases.
- **Deaths**: Number of fatalities recorded.
- **Recovered**: Number of recoveries reported.

### Key Notes:
- The dataset was sourced from the [COVID-19 Dataset on Kaggle](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)).
- Extensive preprocessing was performed, including handling missing values, removing irrelevant columns (e.g., WHO Region, Continent, and Country/Region), and feature scaling.
- Only columns relevant to the analysis, prediction, and time series modeling were retained.

This dataset provides insights into the global trends and patterns of COVID-19 cases, aiding in building robust prediction models and time series analysis.

## Data Preprocessing

Data preprocessing was a critical step to ensure the dataset was clean, structured, and suitable for analysis and model training. Below are the steps performed:

### 1. Loading the Data
- The dataset was loaded into a Pandas DataFrame for further manipulation.
- Initial exploration was conducted to understand its structure, size, and key features.

### 2. Handling Missing Values
- Identified missing values in critical columns such as 'NewCases', 'ConfirmedCases', 'Deaths', and 'Recovered'.
- Missing values were addressed as follows:
  - **NewCases**: Filled using mean or forward-fill techniques depending on the distribution.
  - **ConfirmedCases, Deaths, Recovered**: Forward-fill method was used for continuity.

### 3. Dropping Irrelevant Columns
- Removed columns such as 'WHO Region', 'Continent', and 'Country/Region' after exploratory analysis, as they were not directly relevant to the predictive modeling.

### 4. Feature Engineering
- Created new features, such as '%Inc Recovered' , '%Inc Cases', and '%Inc Deaths', for deeper insights and better modeling performance.
- Used NewCases to ascertain time series analysis as the date column is not available.

### 5. Data Transformation
- Applied scaling techniques where necessary (e.g., MinMaxScaler) to standardize numerical features and improve model convergence.

### 6. Dataset Splitting
- Divided the data into training and test sets in an 80:20 ratio for supervised learning models.
- Prepared a time-indexed series for ARIMA modeling.

### 7. Data Validation
- Conducted final checks to ensure no inconsistencies, outliers, or errors remained in the dataset before moving forward.

This preprocessing ensured the data was clean, relevant, and ready for effective modeling and visualization.

## Model Building

The project employed various machine learning models to predict daily new COVID-19 cases. The steps for model building are outlined below:

### 1. Defining the Problem
- The task was framed as a supervised learning regression problem to predict 'NewCases' using historical data.

### 2. Models Evaluated
- Multiple models were evaluated to identify the best-performing one:
  - **Linear Regression**: Baseline model to assess relationships in the data.
  - **Random Forest Regressor**: For robust, non-linear predictions with feature importance.
  - **ARIMA**: Time-series model specifically for sequential forecasting.

### 3. Model Training
- Split the data into training and test sets (80:20 ratio).
- Each model was trained on the processed training set and hyperparameter tuning was conducted where applicable:
  - **Random Forest**: Tuned 'n_estimators', 'max_depth', and 'min_samples_split.
  - **ARIMA**: Parameters '(5, 1, 0)' were selected based on autocorrelation and partial autocorrelation analysis.

### 4. Model Evaluation
- Metrics used to evaluate model performance included:
  - **Mean Absolute Error (MAE)**: Measures average prediction error.
  - **Mean Squared Error (MSE)**: Highlights larger prediction errors.
  - **Root Mean Squared Error (RMSE)**: Square root of MSE for interpretability.
  - **R² Score**: Assesses model fit quality.

- Random Forest showed superior performance with lower MAE and RMSE compared to Linear Regression and ARIMA.

### 5. Feature Importance
- The Random Forest model provided insights into feature importance, highlighting which predictors had the greatest impact on daily new cases.
![Screenshot (105)](https://github.com/user-attachments/assets/63fbd8fb-e0ae-40ae-88a2-b187b2629e01)

### 6. Time-Series Forecasting with ARIMA
- ARIMA was used to model sequential trends in the data, focusing on 'NewCases'.
- Model parameters '(5, 1, 0)' were optimized using AIC and BIC scores.
- The ARIMA model faced challenges due to the limited number of observations, impacting prediction accuracy.

### 7. Final Model Selection
- The **Random Forest Regressor** was selected as the final model due to its robustness and superior predictive performance.
- Forecasting with ARIMA was included for comparison but required a larger dataset for reliable predictions.

This comprehensive model-building approach ensured a balance of interpretability, accuracy, and robustness.

## Evaluation and Results

The project evaluated the performance of multiple models to predict daily new COVID-19 cases. The results from the evaluation process are summarized below:

### 1. Evaluation Metrics
The following metrics were used to assess the accuracy and reliability of the models:
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted values.
- **Mean Squared Error (MSE)**: Highlights larger prediction errors due to squaring differences.
- **Root Mean Squared Error (RMSE)**: Provides a scale-aligned measure of prediction error.
- **R² Score**: Determines how well the model explains variability in the data.

### 2. Model Comparisons
The models evaluated include:
- **Linear Regression**: Used as a baseline for comparison.
- **Random Forest Regressor**: A robust model that handles non-linear relationships effectively.
- **ARIMA**: Utilized for time-series forecasting, focusing on sequential patterns.

### 3. Results
| Model                   | MAE      | MSE             | RMSE     | R²   |
|-------------------------|----------|-----------------|----------|-------|
| **Linear Regression**   | 5762.32  | 33,204,331.78   | 5762.32  | ---  |
| **Random Forest**       | 4150.50  | 21,604,720.44   | 4647.71  | 0.72  |


- **Random Forest Regressor** outperformed other models with the lowest MAE and RMSE, indicating superior predictive performance.
- **Linear Regression** showed significant errors due to the complexity of the dataset.
- **ARIMA** was tested but required a larger dataset for reliable predictions.

### 4. Feature Importance
Random Forest provided insights into key predictors influencing 'NewCases'. Top features included:
- 'NewDeaths': Indicated a strong correlation with new case counts.
- 'NewRecovered': Captured trends in case recovery and its influence on predictions.

- **Feature Importance Plot**: Highlighted the most significant features impacting predictions.
- [featureimportance](https://github.com/user-attachments/assets/93ec75d1-08bd-41ec-bf57-4d310a91e28b)


### 6. Insights
- The dataset's limited size and variability posed challenges for ARIMA modeling.
- Random Forest was robust to these constraints and delivered reliable predictions.
- Feature engineering and more data could further enhance model performance.

### 7. Recommendations
- For future work, collect more time-series data to improve ARIMA's forecasting capability.
- Apply ensemble methods or hybrid models to explore further performance gains.
- Utilize the insights from feature importance to guide resource allocation and response planning.

The **Random Forest Regressor** was selected as the final model due to its superior predictive accuracy and interpretability.


