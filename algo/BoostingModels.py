from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from logger.LoggerClass import LoggerFileClass

logger = LoggerFileClass("boosting_model")

class BoostingModelReg:
    """This class is used to build regression models using different ensemble techniques.
            Author: Bhushan Patil
            References I referred:
            Reference 1 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html?highlight=adaboost%20regressor#sklearn.ensemble.AdaBoostRegressor
            reference 2 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html?highlight=gradient%20boost%20regressor#sklearn.ensemble.GradientBoostingRegressor
            reference 3 - https://xgboost.readthedocs.io/en/latest/get_started.html
            reference 4 - https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

            parameters:
            --------------------------------
            x_train: Training data frame containing the independent features.
            y_train: Training dataframe containing the dependent or target feature.
            x_test: Testing dataframe containing the independent features.
            y_test: Testing dataframe containing the dependent or target feature.
            """

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def adaboost_regressor(self):
        """Description: This method builds a model using AdaBoostRegressor algorithm, a type of ensemble technique imported
        from the sci-kit learn library. It uses cross validation technique and chooses the best estimator with the
        best hyper parameters.
        Raises an exception if it fails

        returns
        ----------------------------------
        The Adaboost regressor model and prints the importance of each feature
        """
        logger.add_info_log(
            "Enter class BoostingModelReg : adaboost_regressor function")

        try:
            adb = AdaBoostRegressor()  # instantiating the AdaBoostRegressor object

            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                      'loss': ['linear', 'square', 'exponential']
                      }  # parameter grid

            rcv = RandomizedSearchCV(estimator=adb, param_distributions=params, n_iter=10, scoring='r2',
                                     n_jobs=-1, cv=10, verbose=5, random_state=42, return_train_score=True)

            # instantiating RandomizedSearchCV
            print('Cross validation process for the Adaboost regressor')

            rcv.fit(self.x_train, self.y_train)  # fitting on the train data
            print()

            print('The best estimator for the Adaboost regressor is',
                  rcv.best_estimator_)  # displaying the best estimator

            adb = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final Adaboost regressor.

            adb.fit(self.x_train, self.y_train)  # fitting on the train data

            # Feature importance by the Adaboost regressor
            adb_feature_imp = pd.DataFrame(adb.feature_importances_, index=self.x_train.columns,
                                           columns=['Feature_importance'])
            adb_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the Adaboost regressor: ', adb_feature_imp)
            print()

            logger.add_info_log("class BoostingModelReg : adaboost_regressor. Model "
                                "Build successfully")
            return adb

        except Exception as e:

            logger.add_exception_log(f'class BoostingModelReg : adaboost_regressor. Model '
                                f'Build failed. Exception {str(e)}')


    def gradientboosting_regressor(self):
        """Description: This method builds a model using GradientBoostingRegressor algorithm, a type of ensemble technique imported
        from the sci-kit learn library. It uses cross validation technique and chooses the best estimator with the
        best hyper parameters.
        Raises an exception if it fails

        returns
        -------------------------------------
        The Gradientboosting regressor model and prints the importance of each feature
        """
        logger.add_info_log(
            "Enter class BoostingModelReg : gradientboosting_regressor function")

        try:

            gbr = GradientBoostingRegressor()  # instantiating the GradientBoostingRegressor object.

            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'learning_rate': [0.1, 0.2, 0.5, 0.8, 1],
                      'loss': ['lr', 'lad', 'huber'],
                      'subsample': [0.001, 0.009, 0.01, 0.09, 0.1, 0.4, 0.9, 1],
                      'criterion': ['friedman_mse', 'mse'],
                      'min_samples_split': [2, 4, 8, 10],
                      'min_samples_leaf': [1, 10, 20, 50]
                      }  # Parameter grid

            rcv = RandomizedSearchCV(estimator=gbr, param_distributions=params, n_iter=10, scoring='r2', n_jobs=-1,
                                     cv=10, verbose=5, random_state=42,
                                     return_train_score=True)  # instantiating RandomizedSearchCV

            print('Cross validation process for the Gradient Boosting Regressor')

            rcv.fit(self.x_train, self.y_train)  # Fitting on the train data
            print()

            print('The best estimator for the GradientBoosting regressor is',
                  rcv.best_estimator_)  # displaying the best estimator

            gbr = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final Gradient Boosting regressor.

            gbr.fit(self.x_train, self.y_train)  # fitting on the train data

            # Feature importance by the Gradient Boosting regressor
            gbr_feature_imp = pd.DataFrame(gbr.feature_importances_, index=self.x_train.columns,
                                           columns=['Feature_importance'])
            gbr_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the Gradient boosting regressor: ', gbr_feature_imp)
            print()

            logger.add_info_log("class BoostingModelReg : gradientboosting_regressor. Model "
                                "Build successfully")

            return gbr

        except Exception as e:

            logger.add_exception_log(f'class BoostingModelReg : gradientboosting_regressor. Model '
                                f'Build failed. Exception {str(e)}')

    def xgb_regressor(self):
        """Description: This method builds a model using XGBRegressor algorithm, a type of ensemble technique imported from the
        xgboost library.It uses cross validation technique and chooses the best estimator with the
        best hyper parameters.
        Raises an exception if it fails

        returns
        -----------------------------
        The XGBoost regressor model and prints the importance of each feature
        """
        logger.add_info_log(
            "Enter class BoostingModelReg : xgb_regressor function")
        try:
            xgbr = XGBRegressor()  # instantiating the XGBRegressor object

            params = {
                'learning_rate': [0.1, 0.2, 0.5, 0.8, 1],
                'max_depth': [2, 3, 4, 5, 6, 7, 8, 10],
                'subsample': [0.001, 0.009, 0.01, 0.09, 0.1, 0.4, 0.9, 1],
                'min_child_weight': [1, 2, 4, 5, 8],
                'gamma': [0.0, 0.1, 0.2, 0.3],
                'colsample_bytree': [0.3, 0.5, 0.7, 1.0, 1.4],
                'reg_alpha': [0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1, 4, 8, 10, 50, 100],
                'reg_lambda': [1, 4, 5, 10, 20, 50, 100, 200, 500, 800, 1000]
            }  # Parameter grid

            rcv = RandomizedSearchCV(estimator=xgbr, param_distributions=params, n_iter=10,
                                     scoring='r2', cv=10, verbose=2,
                                     random_state=42, n_jobs=-1,
                                     return_train_score=True)  # instantiating RandomizedSearchCV
            print('Cross validation process for the XGBoost regressor')

            rcv.fit(self.x_train, self.y_train)  # Fitting on the train data
            print()

            print('The best estimator for the XGBoost regressor is',
                  rcv.best_estimator_)  # displaying the best estimator

            xgbr = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final XGBoosting regressor.

            xgbr.fit(self.x_train, self.y_train)  # fitting on the train data

            # Feature importance by the XGBoosting regressor
            xgbr_feature_imp = pd.DataFrame(xgbr.feature_importances_, index=self.x_train.columns,
                                            columns=['Feature_importance'])
            xgbr_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the XGBoost regressor: ', xgbr_feature_imp)
            print()

            logger.add_info_log("class BoostingModelReg : xgb_regressor. Model "
                                "Build successfully")

            return xgbr

        except Exception as e:
            logger.add_exception_log(f'class BoostingModelReg : xgb_regressor. Model '
                                     f'Build failed. Exception {str(e)}')

    def model_predict(self, model, X):
        """Description: This method makes predictions using the given model
        raises an exception if it fails

        parameters
        ----------------------------------
        model:- model to be used for making predictions
        X = A pandas dataframe with independent features

        returns
        ----------------------------------
        The predictions of the target variable.
        """
        try:
            logger.add_info_log(
                "Enter class BoostingModelReg : model_predict function")
            pred = model.predict(X)

            return pred

        except Exception as e:
            logger.add_exception_log(f'class BoostingModelReg : model_predict. Model '
                                     f'Build failed. Exception {str(e)}')
