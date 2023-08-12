# U.S Business Cycle Prediction using deep learning model
In this git repository, we will test LSTM(Long Short-Term Memory) and GRU(Gated Recurrent Unit) model for classifying the current month into recession or expansion and compare their performances with baseline models such as RF(Random Forest) and GB(Gradient Boost) models. We chose these RNN models because they are appropriate for automatic feature selection from a large number of features and learning temporal dependencies of the data for predicting recession. 
Though macroeconomic variables such as term spread and employment are known to be powerful signals for identifying recession, we will confirm the previously known predictors and explore other strong predictors for business cycles through comparison among the models. Especially for LSTM and GRU models, SHAP analysis will be used for extracting important features. 

## **Table of Contents**
- [Data Preparation](#data-preparation)
- [Data Preprocessing with EDA](#data-preprocessing-with-eda)
- Modeling
  - [LSTM/GRU modeling](#lstmgru-modeling)
  - [RF/GB modeling](#rfgb-modeling)
- Evaluation
  - [SHAP Analysis for top LSTM/GRU models](#shap-analysis-for-top-lstmgru-models)


## Data Preparation
[View code](0.Data%20Preparation.ipynb)

[Data](/data/rawdata_USA.csv) can be accessed through data folder
- Data Sources: OECD, FRED, Yahoo Finance, NBER
  - [OECD](https://stats.oecd.org/Index.aspx?QueryId=6617)
  - [FRED](https://fred.stlouisfed.org/)
  - [Yahoo Finance](https://finance.yahoo.com/quote/%5EVIX?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuYmluZy5jb20v&guce_referrer_sig=AQAAAA4PqCecADB7jRPFc1zok8-qFh4DqQSOk34atVLfrbddWTd55E2A15f7EStgzKyffi1VS95neqtNMTSYDHO_F4f2tfYUe2MxEe3Twn17hbNcjsMwMdzV_PVcRGgqZGiFhYM1dr8N3fA_ARafsb1wHvxHVRo-UPk3OLARkSDjcIKE)
  - [NBER](https://www.nber.org/research/business-cycle-dating)
- All data sources are open to the public and lincences for data use and distribution are respected.

1. Data Downloading through OECD API
2. Manual Feature selection using relevant features
- [View the list of manually selected features](/data/metadata_filter.xlsx)


## Data Preprocessing with EDA
[View code](1.Data%20Preprocessing.ipynb)

1. EDA for class imbalance
- check class imbalance problem of recession prediction in that recession is far less frequent than that of expansion and compare between OECD-based cycles and NBER-based cycles

2. Data Transformation
- transform features for better interpretability
  - [View the transformation method using 'Tcode'](/data/metadata_final.csv)
  - transformation method: difference, log-difference, no transformation

3. Selecting features with minimum 50 years of availability 

4. Imputing missing values using linear interpolation

5. Blocked Time Series Split for train-validation dataset + 5years of test dataset
- split train-validation dataset with 45 years
  - Blocked split method for avoiding 'look ahead bias'
- assign test dataset to last 5 years of the data

6. Stationarity Test(ADF test) for final full X data  
- Stationarity of the features are required for Blocked Time Series Split


## LSTM/GRU modeling
[View code](2-1.LSTM_GRU%20modeling.ipynb)

1. Define function
- create_dataset(): transform X dataset into dataset splitted in timestep-unit 
- BlockingTimeSeriesSplit(): make train-validation split object
  - reference : https://gmnam.tistory.com/230#:~:text=class%20BlockingTimeSeriesSplit%28%29%3A%20def%20__init__%28self%2C%20n_splits%29%3A%20self.n_splits%20%3D%20n_splits,indices%20%5Bstart%3A%20mid%5D%2C%20indices%20%5Bmid%20%2B%20margin%3A%20stop%5D 
- make_split(): train-validation split using BlockedTimeSeriesSplit() object
- make_lstm_model(): make lstm model structure
  - X.shape=[#obs-(t-1), t, #feature], y.shape=[#obs-(t-1), #class] 
  - output.shape=[#obs-(t-1), #class]: ex.column 0 will show the predicted probability for class 0
- make_gru_model(): make gru model structure
  - X.shape=[#obs-(t-1), t, #feature], y.shape=[#obs-(t-1), #class] 
  - output.shape=[#obs-(t-1), #class]: ex.column 0 will show the predicted probability for class 0
- grid_search(): execute grid search for lstm/gru model and save the models and performance results in history_dict.pkl in the designated folder path

2. Model Selection
- select the best models in terms of MAR(Macro Averaging Recall) scores averaged across 3CV 


## RF/GB modeling
[View code](2-2.%20RF_GB%20modeling.ipynb)

1. Define function
- BlockingTimeSeriesSplit(): make train-validation split object& plot_cv_indices(): plot block split results
  - reference : https://gmnam.tistory.com/230#:~:text=class%20BlockingTimeSeriesSplit%28%29%3A%20def%20__init__%28self%2C%20n_splits%29%3A%20self.n_splits%20%3D%20n_splits,indices%20%5Bstart%3A%20mid%5D%2C%20indices%20%5Bmid%20%2B%20margin%3A%20stop%5D 
- make_split(): train-validation split using BlockedTimeSeriesSplit() object
- classification_report_csv(): convert classification report into dataframe
- rf_grid_search(): execute grid search for rf model and save the models and performance results in clf_{target type}_rf.pkl in the model folder
- gb_grid_search(): execute grid search for gb model and save the models and performance results in clf_{target type}_gb.pkl in the model folder

2. Grid Search for 4 cases of the baseline models
- RF model trained with all features(45 features)
- RF model trained with top 10 features extraced from the previous model
- GB model trained with all features(45 features)
- GB model trained with top 10 features extraced from the previous model

3. evaluate final test performance of the 4 best models each for 4 cases.

4. calculate feature importance for the 4 best models each for 4 cases.


## SHAP Analysis for top LSTM/GRU models
[View code](3.Shap%20Analysis%20for%20top%20LSTM_GRU%20models.ipynb)

In terms of 3CV MAR scores, the best model was LSTM model. Using top 5 models among LSTM/GRU models, execute SHAP analysis to extract important predictors for business cycles. 

1. Train and evaluate top 5 models and calculate SHAP values
- train each model with (train + validation) dataset and evaluate for final test dataset (last 5 years)
  - final train dataset = 74.12~17.12 (The first 23 month from 73.1 were removed due to application of 24month-timesteps)
  - final test dataset = 18.1~22.12
- calculate SHAP values for train and test dataset
  - feature importance plot
  - SHAP summary plot
  - SHAP dependence plot
  - SHAP force plot for the COVID-19 pandemic recession(positive samples within the final test dataset) 