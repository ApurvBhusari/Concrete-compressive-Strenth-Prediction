from data_injection.data_injection import DataLoader
from data_preprocessing.DataPreprocessing import DataPreprocessing
from algo.LinearModels import LinearRegressionWithFeatureSelection
from algo.TreeModels import TreeModelsReg
from algo.BoostingModels import BoostingModelReg


import pandas as pd
import warnings
import joblib

import warnings

from evaluation_matrix.EvaluationMatrix import EvaluationMetrix

warnings.filterwarnings('ignore')

from logger.LoggerClass import LoggerFileClass

logger = LoggerFileClass("main")

data = r'datasets/concrete_data.csv'

logger.add_info_log("Starting development Process... main.py ")

"""Loading datasets """
d = DataLoader(data)
df = d.fetch_data()

"""Data Preprocessing """
dp = DataPreprocessing(df)
# As per the EDA, 'age' column is having more number of outliers. Let's remove them.
df = dp.remove_outliers('age')

"""Splitting the data into train and test sets respectively"""
df_train, df_test = dp.data_split(test_size=0.30)  # splitting the dataframe into train and test sets respectively.

"""Feature scaling for linear regression"""
df_train_sc, df_test_sc = dp.feature_scaling(df_train, df_test)

"""Splitting datasets into independent and dependent datasets respectively"""
X_train, y_train, X_test, y_test = dp.splitting_as_x_y(df_train_sc, df_test_sc,
                                                       'concrete_compressive_strength')
# Splitting data into independent and dependent variables respectively

"""Experimenting with Linear regression model, feature selection using Backward elimination approach"""

lr = LinearRegressionWithFeatureSelection(X_train, y_train, X_test, y_test)
""" 1) Linear regression with Backward elimination approach """
lr_be, X_train_be, y_pred_train_be, X_test_be, y_pred_test_be, relevant_features_by_BE = lr.backward_elimaination_approach()

logger.add_info_log(
    "Defining a function to record the relevant features of each model in the 'relevant_features_by_models.csv' file"
    "in the 'results' directory")

imp_f = pd.DataFrame(columns=['Algorithm', 'Imp_Features'])


def rec_imp_features(dataframe, algorithm_name, imp_features):
    """Description: This function stores the important features of an algorithm we used, in a Pandas dataframe.

     parameters

     ----------------------------------------
     dataframe: A Pandas dataframe in which the results are recorded
     algorithm_name: Algorithm we used
     imp_features: Important features chose by the Algorithm

     returns
     ----------------------------------------
     Pandas dataframe containing the important features information w.r.t. the algorithm, saved to local disk as a
     .csv file.
     """
    try:
        dataframe.loc[len(dataframe)] = [algorithm_name, imp_features]

        logger.add_info_log(f"Important features of the {algorithm_name} updated in the 'relevant_features_by_models.csv' ")

        return dataframe.to_csv('results/relevant_features_by_models.csv', index=False)

    except Exception as e:
        logger.add_exception_log("Exception occurred in the function 'rec_imp_features'. "
                      "Exception message: " + str(e))


rec_imp_features(imp_f, "Linear Regression_BE", relevant_features_by_BE)
# calling the function to store the info of relevant features

metrics = EvaluationMetrix()  # instantiating the Metrics object

"""Storing the results of model performance"""

algo_results = pd.DataFrame(columns=['Algorithm', 'Train_R2 score', 'Train_Adj_R2 score', 'Train_RMSE score',
                                     'Test_R2 score', 'Test_Adj_R2 score',
                                     'Test_RMSE score'])  # Data frame to record performance of every model

logger.add_info_log(
    'Created a blank data frame "algo_results" to store the results of each and every model')  # logging operation

logger.add_exception_log('Defining a function "evaluate" to evaluate the performance metrics of a model')  # logging operation


def evaluate(dataframe, algorithm_name,
             x_train, y_train, y_pred_train,
             x_test, y_test, y_pred_test):  # Function to update the dataframe 'algo_results'
    """ Description: This function stores the model's performance metrics and returns a pandas dataframe.

    parameters
    ------------------------------
    dataframe: A pandas dataframe to store the results of experiments on various algorithms
    algorithm_name: Name of an algorithm w.r.t. its results
    x_train: Training data with independent features
    y_train: Training data with dependent feature i.e., actual values
    y_pred_train: Predictions on the training data
    x_test: Testing data with independent features
    y_test: Testing data with dependent feature i.e, actual values
    y_pred_test: Predictions on the testing data

    returns
    -----------------------------
    Pandas dataframe containing the performance metrics of algorithms and saved to local disk as .csv file
    """
    try:
        dataframe.loc[len(dataframe)] = [algorithm_name, metrics.r2_score(y_train, y_pred_train),
                                         metrics.adj_r2_score(x_train, y_train, y_pred_train),
                                         metrics.rmse_score(y_train, y_pred_train),
                                         metrics.r2_score(y_test, y_pred_test),
                                         metrics.adj_r2_score(x_test, y_test, y_pred_test),
                                         metrics.rmse_score(y_test, y_pred_test)]
        logger.add_info_log(
            f'Results Dataframe saved to disk as "Performance of algorithms.csv" file in the "results" directory. ')
        logger.add_info_log("Updated the results in 'Performance of algorithms.csv'")
        return dataframe.to_csv('results/Performance of algorithms.csv', index=False)

    except Exception as e:
        logger.add_exception_log("Exception occurred in the function 'evaluate'. "
                      "Exception message: " + str(e))


evaluate(algo_results, 'Linear Regression_BE',
         X_train_be, y_train, y_pred_train_be,
         X_test_be, y_test, y_pred_test_be)  # calling the function to store results into local disk

""" 2) Linear regression with RFE approach """

lr_rfe, X_train_rfe, y_pred_train_rfe, X_test_rfe, y_pred_test_rfe, relevant_features_by_RFE = lr.rfe_approach()
# Linear regression model, its predictions using the relevant features by the RFE method.

rec_imp_features(imp_f, "Linear Regression_RFE",
                 relevant_features_by_RFE)  # storing the info of the relevant features

"""Storing the results of model performance """

evaluate(algo_results, 'Linear Regression_RFE',
         X_train_rfe, y_train, y_pred_train_rfe,
         X_test_rfe, y_test, y_pred_test_rfe)  # calling the function "evaluate" to store the results into .csv file


""" 2) Linear regression with RFE approach """

lr_rfe, X_train_rfe, y_pred_train_rfe, X_test_rfe, y_pred_test_rfe, relevant_features_by_RFE = lr.rfe_approach()
# Linear regression model, its predictions using the relevant features by the RFE method.

rec_imp_features(imp_f, "Linear Regression_RFE",
                 relevant_features_by_RFE)  # storing the info of the relevant features

"""Storing the results of model performance """

evaluate(algo_results, 'Linear Regression_RFE',
         X_train_rfe, y_train, y_pred_train_rfe,
         X_test_rfe, y_test, y_pred_test_rfe)  # calling the function "evaluate" to store the results into .csv file

""" 3) Linear regression with LassoCV approach """
ls = LinearRegressionWithFeatureSelection(X_train, y_train, X_test, y_test)

  # An object which is responsible for building Linear regression models with Lasso regularization.

lr_lasso, X_train_lasso, y_pred_train_lasso, X_test_lasso, y_pred_test_lasso = ls.lassocv()  # Linear regression model and predictions of both the train and test data respectively

relevant_features_by_Lasso = ['cement', 'age', 'blast_furnace_slag', 'fly_ash', 'superplasticizer',
                              'fine_aggregate', 'water']  # relevant features
rec_imp_features(imp_f, "Linear Regression_Lasso",
                 relevant_features_by_Lasso)  # storing the info of the relevant features

"""Storing the results of model performance """

evaluate(algo_results, 'Linear Regression_Lasso',
         X_train_lasso, y_train, y_pred_train_lasso,
         X_test_lasso, y_test, y_pred_test_lasso)  # calling the function "evaluate" to store the results into local disk

"""As per the results, in the "Experiments with algorithms.csv", all the 3 approaches of linear regression
delivering the similar results. Linear Regression_RFE approach is having very low rmse score on the test set, 
so we can consider this as a best model among the three. 
And after conducting the residual analysis using this model in the jupyter notebook "EDA + Model building.ipynb" it is evident 
that, it satisfies all the Linear regression assumptions. """

"""But all the above 3 variants of Linear regression shows slight under-fit. Let's also experiment with different tree models."""

"""Splitting datasets into independent and dependent datasets respectively"""

X_train, y_train, X_test, y_test = dp.splitting_as_x_y(df_train, df_test,
                                                       'concrete_compressive_strength')
# Tree models do not require feature scaling, hence we are using the original data

t = TreeModelsReg(X_train, y_train, X_test, y_test,
                  )  # An object which is responsible for building tree based models

""" 1) Decision Tree Regressor """

# logging operation
logger.add_info_log(
    "Building a Decision tree regressor model on the training data and the importance of each feature will be displayed "
    "in the console.")

dt_model = t.decision_tree_regressor()  # Decision tree regressor model

top_features_dt = ['cement', 'age', 'water', 'blast_furnace_slag']  # top 4 features as per the feature importance

rec_imp_features(imp_f, "Decision tree regressor", top_features_dt)  # storing the top features data in a file
X_train_dt = X_train[top_features_dt]  # considering only the relevant features
X_test_dt = X_test[top_features_dt]

t = TreeModelsReg(X_train_dt, y_train, X_test_dt, y_test,
                  )  # An object which is responsible for building tree based models

# logging operation
logger.add_info_log("Building a Decision tree regressor model on the training data with the relevant features only")

dt_model_2 = t.decision_tree_regressor()  # building a decision tree regressor model

y_pred_train_dt = t.model_predict(dt_model_2, X_train_dt)  # predictions on the training data
y_pred_test_dt = t.model_predict(dt_model_2, X_test_dt)  # predictions on the testing data

logger.add_info_log(
    'Using Decision tree regressor model, successfully made predictions on both the training and testing data respectively')

"""Storing the results of model performance """

evaluate(algo_results, 'Decision tree regressor',
         X_train_dt, y_train, y_pred_train_dt,
         X_test_dt, y_test, y_pred_test_dt)  # calling the function "evaluate" to store the results into local disk.

""" Result:- Model over-fit.
Let's try with post pruning technique to reduce over-fitting in the decision tree"""

# logging operation
logger.add_info_log(
    "Building a Decision tree regressor model with post pruning technique. Considering only the relevant features.")

rec_imp_features(imp_f, "Decision tree regressor_post pruning",
                 top_features_dt)  # storing the top features data in a file

dt_model_pp = t.decision_tree_regressor_post_pruning()  # building a decision tree regressor by post pruning technique

y_pred_train_dt_pp = t.model_predict(dt_model_pp, X_train_dt)  # predictions on the training data
y_pred_test_dt_pp = t.model_predict(dt_model_pp, X_test_dt)  # predictions on the testing data

logger.add_info_log(
    'Using Decision tree regressor_post pruning model, successfully made predictions on both the training and testing data respectively')

"""Storing the results of model performance """

evaluate(algo_results, 'Decision tree regressor_post pruning',
         X_train_dt, y_train, y_pred_train_dt_pp,
         X_test_dt, y_test, y_pred_test_dt_pp)  # calling the function "evaluate" to store the results into local disk

"""Result: Over-fitting issue resolved and got a better model compared to the previous one"""

""" Let's experiment with some ensemble techniques as well to get even better R2 scores"""

""" 2) Random forest regressor """
t = TreeModelsReg(X_train, y_train, X_test, y_test,
                  )  # An object which is responsible for building tree based models

# logging operation
logger.add_info_log(
    "Building a Random forest regressor model on the training data and the importance of each feature will be displayed "
    "in the console.")

rf_model = t.random_forest_regressor()  # building a random forest regressor model

top_features_rf = ['age', 'cement', 'water', 'blast_furnace_slag',
                   'superplasticizer']  # top 5 features as per the feature importance

rec_imp_features(imp_f, "Random Forest regressor", top_features_rf)  # storing the relevant features in a file.

X_train_rf = X_train[top_features_rf]  # considering only the relevant features
X_test_rf = X_test[top_features_rf]

t = TreeModelsReg(X_train_rf, y_train, X_test_rf, y_test,
                  )  # An object which is responsible for building tree based models
# logging operation
logger.add_info_log(
    "Building a Random forest regressor model on the training data with the relevant features only.")

rf_model_2 = t.random_forest_regressor()  # building a random forest regressor model

y_pred_train_rf = t.model_predict(rf_model_2, X_train_rf)  # predictions on the training data
y_pred_test_rf = t.model_predict(rf_model_2, X_test_rf)  # predictions on the testing data

logger.add_info_log(
    'Using Random Forest Regressor model, successfully made predictions on both the training and testing data respectively')

"""Storing the results of model performance """

evaluate(algo_results, 'Random Forest regressor',
         X_train_rf, y_train, y_pred_train_rf,
         X_test_rf, y_test, y_pred_test_rf)  # calling the function "evaluate" to store the results into local disk

""" Result:- Random forest regressor model looks far more better than the Decision tree regressor.
# Let's try with Boosting algorithms as well and see whether we get any even better scores"""

""" 3) AdaBoost Regressor """

b = BoostingModelReg(X_train, y_train, X_test, y_test,
                  )  # An object which is responsible for building tree based models

# logging operation
logger.add_info_log(
    "Building an Adaboost regressor model on the training data and the importance of each feature will be displayed "
    "in the console.")
adb_model = b.adaboost_regressor()  # building an Adaboost regressor model

top_features_adb = ['cement', 'age', 'water', 'blast_furnace_slag', 'superplasticizer', 'fine_aggregate ',
                    'coarse_aggregate']  # Top features by the Adaboost regressor

rec_imp_features(imp_f, "Adaboost regressor", top_features_adb)  # storing the relevant features in the file.

X_train_adb = X_train[top_features_adb]  # considering only the relevant features
X_test_adb = X_test[top_features_adb]

b = BoostingModelReg(X_train_adb, y_train, X_test_adb, y_test,
                  )  # An object which is responsible for building tree based models)

logger.add_info_log(
    "Building an Adaboost regressor model on the training data with the relevant features only.")
adb_model_2 = b.adaboost_regressor()  # building an Adaboost regressor model

y_pred_train_adb = b.model_predict(adb_model_2, X_train_adb)  # predictions on the training data
y_pred_test_adb = b.model_predict(adb_model_2, X_test_adb)  # predictions on the testing data

logger.add_info_log(
    'Using Adaboost Regressor model, successfully made predictions on both the training and testing data respectively')

"""Storing the results of model performance """

evaluate(algo_results, 'Adaboost regressor',
         X_train_adb, y_train, y_pred_train_adb,
         X_test_adb, y_test, y_pred_test_adb)  # calling the function "evaluate" to store the results into local disk.

""" Result:- Model accuracy is less that the random forest regressor. Let's experiment with Gradient boosting regressor."""

""" 4) Gradient Boosting Regressor """

b = BoostingModelReg(X_train, y_train, X_test, y_test,
                  )  # An object which is responsible for building tree based models

# logging operation
logger.add_info_log(
    "Building a Gradient Boosting regressor model on the training data and the importance of each feature will be displayed "
    "in the console.")
gbr_model = b.gradientboosting_regressor()  # building a gradient boosting regressor

top_features_gbr = ['age', 'cement', 'water', 'blast_furnace_slag']  # top 4 features by the Gradient boosting regressor

rec_imp_features(imp_f, "Gradient Boost regressor", top_features_gbr)  # storing the relevant features in the file.

X_train_gbr = X_train[top_features_gbr]  # considering only the relevant features
X_test_gbr = X_test[top_features_gbr]

b = BoostingModelReg(X_train_gbr, y_train, X_test_gbr, y_test,
                  )  # An object which is responsible for building tree based models)

# logging operation
logger.add_info_log("Building a Gradient Boosting regressor model on the training data with the relevant features only")

gbr_model_2 = b.gradientboosting_regressor()  # building a Gradient boosting regressor model

y_pred_train_gbr = b.model_predict(gbr_model_2, X_train_gbr)  # predictions on the training data
y_pred_test_gbr = b.model_predict(gbr_model_2, X_test_gbr)  # predictions on the testing data

logger.add_info_log(
    'Using Gradient Boosting Regressor model, successfully made predictions on both the training and testing data respectively')

"""Storing the results of model performance """

evaluate(algo_results, 'Gradient Boost regressor',
         X_train_gbr, y_train, y_pred_train_gbr,
         X_test_gbr, y_test, y_pred_test_gbr)  # calling the function "evaluate" to store the results into local disk

""" Result:- Compared to Adaboost regressor, Gradient Boost Regressor has performed well. Let's check with XGBoost regressor as well."""

""" 5) XGBoost Regressor """

xg = BoostingModelReg(X_train, y_train, X_test, y_test,
                  )  # An object which is responsible for building tree based models

# logging operation
logger.add_info_log(
    "Building an XGBoost regressor model on the training data and the importance of each feature will be displayed "
    "in the console.")

xgbr_model = xg.xgb_regressor()  # building an XGBoost regressor model

top_features_xgbr = ['age', 'cement', 'water', 'fly_ash', 'superplasticizer',
                     'blast_furnace_slag']  # top features by the XGBoost regressor

rec_imp_features(imp_f, "XGBoost regressor", top_features_xgbr)  # storing the relevant features into file.

X_train_xgbr = X_train[top_features_xgbr]  # considering only the relevant features
X_test_xgbr = X_test[top_features_xgbr]

xg = BoostingModelReg(X_train_xgbr, y_train, X_test_xgbr, y_test,
                  )  # An object which is responsible for building tree based models)

# logging operation
logger.add_info_log("Building an XGBoost regressor model on the training data with the relevant features only. ")

xgbr_model_2 = xg.xgb_regressor()  # building a XGBoost regressor model

y_pred_train_xgbr = xg.model_predict(xgbr_model_2, X_train_xgbr)  # predictions on the training data
y_pred_test_xgbr = xg.model_predict(xgbr_model_2, X_test_xgbr)  # predictions on the testing data

logger.add_info_log(
    'Using XGBoost regressor model, successfully made predictions on both the training and testing data respectively')

"""Storing the results of model performance """

evaluate(algo_results, 'XGBoost regressor',
         X_train_xgbr, y_train, y_pred_train_xgbr,
         X_test_xgbr, y_test, y_pred_test_xgbr)  # calling the function "evaluate" to store the results into local disk.

""" Result:- As per the results recorded in the "Experiments with algorithms.csv" , 
XGBoost Regressor is the best one in terms of adjusted R2 score on testing data, followed by Random Forest regressor """

logger.add_info_log("Best Model: XGBoost regressor ")  # logging operation

# saving the best models into disk

logger.add_info_log('Saving the XGBoost regressor model into the "models" directory')  # logging operation

joblib.dump(xgbr_model_2, 'models\XGBoost_Regressor_model.pkl')

logger.add_info_log('Saving the Random Forest regressor model into the "models" directory')  # logging operation

joblib.dump(rf_model_2, 'models\RandomForest_Regressor_model.pkl')

logger.add_info_log(" Solution development part completed successfully. Thank you _/\_ , Stay safe and healthy :-) ")



if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
