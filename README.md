# Capstone_Team10
Predicting the Recession of the U.S

# Data Preprocessing with EDA
*  Class Imbalance Problem and Comparing OECD and NBER cycles
*  Data Transformation
*  Selecting features with minimum 50 years of availability 
*  Imputing missing values using linear interpolation
*  Blocked Time Series Split for train-validation dataset + 5years of test dataset
*  Stationarity Test(ADF test) for final full X data  


# LSTM/GRU modeling

1. Modeling
* Used Functions 
- create_dataset(): transform X dataset into dataset splitted in timestep-unit 
- BlockingTimeSeriesSplit(): make train-validation split object
** reference : https://gmnam.tistory.com/230#:~:text=class%20BlockingTimeSeriesSplit%28%29%3A%20def%20__init__%28self%2C%20n_splits%29%3A%20self.n_splits%20%3D%20n_splits,indices%20%5Bstart%3A%20mid%5D%2C%20indices%20%5Bmid%20%2B%20margin%3A%20stop%5D 
- make_split(): train-validation split using BlockedTimeSeriesSplit() object
- make_lstm_model(): make lstm model structure
- make_gru_model(): make gru model structure
- grid_search(): execute grid search for lstm/gru model and save the models and performance results in history_dict.pkl in the designated folder path

2. Model Selection
* df_results.pkl & df_selection.pkl
- df_results: concatenate all candidate LSTM/GRU models result by model specification and each validation
- df_selection: concatenate all candidate LSTM/GRU models result by model specification


# RF/GB modeling
* Used Functions 
-- BlockingTimeSeriesSplit(): make train-validation split object
& plot_cv_indices(): plot block split results
** reference : https://gmnam.tistory.com/230#:~:text=class%20BlockingTimeSeriesSplit%28%29%3A%20def%20__init__%28self%2C%20n_splits%29%3A%20self.n_splits%20%3D%20n_splits,indices%20%5Bstart%3A%20mid%5D%2C%20indices%20%5Bmid%20%2B%20margin%3A%20stop%5D 
-- make_split(): train-validation split using BlockedTimeSeriesSplit() object
-- classification_report_csv(): convert classification report into dataframe
-- rf_grid_search(): execute grid search for rf model and save the models and performance results in clf_{target type}_rf.pkl in the model folder
-- gb_grid_search(): execute grid search for gb model and save the models and performance results in clf_{target type}_gb.pkl in the model folder

##1. RF model 
* RF model trained with all features(45 features)
* RF model trained with top 10 features extraced from the previous model

##2. GB model 
* GB model trained with all features(45 features)
* GB model trained with top 10 features extraced from the previous model

## Random Forest(RF) Modeling
### define function for random forest grid search
### execute grid search for rf model with full features

# SHAP Analysis for top LSTM/GRU models
## import top 5 models from df_selection.pkl file


## Train and evaluate top 5 models and calculate SHAP values

* train each model with (train + validation) dataset and evaluate for final test dataset (last 5 years)
-- final train dataset = 74.12~17.12 (The first 23 month from 73.1 were removed due to application of 24month-timesteps)
-- final test dataset = 18.1~22.12

* for each model
-- train the model
-- evaluation (classification report & confusion matrix)
-- calculate SHAP values for train and test dataset

## save the final results of top 5 models in analysis_dict.pkl file

## plot the prediction vs. true recession during final test period (18.1~22.12)

## plot the prediction vs. true recession during final train period (74.12~17.12)

## plot feature importance based on SHAP values

## plot SHAP summary plot for {idx+1} ranked model

## plot SHAP dependence plot for specified feature list for {idx+1} ranked model

## plot SHAP force plot for the COVID-19 pandemic case study
