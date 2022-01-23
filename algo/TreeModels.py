from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd
import warnings


warnings.filterwarnings('ignore')

from logger.LoggerClass import LoggerFileClass

logger = LoggerFileClass("tree_model")


class TreeModelsReg:
    """This class is used to build regression models using different tree techniques.
        Author: Bhushan Patil
        References I referred:
        reference 1 - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        reference 2 - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decision%20tree%20regressor#sklearn.tree.DecisionTreeRegressor
        reference 3 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest%20regressor#sklearn.ensemble.RandomForestRegressor


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

    def decision_tree_regressor(self):
        """Description: This method builds a model using DecisionTreeRegressor algorithm imported from the sci-kit learn,
        by implementing cross validation technique to choose the best estimator with the best hyper parameters.
        Raises an exception if it fails

        returns
        --------------------------------
        The Decision tree regressor model and prints the importance of each feature
        """
        logger.add_info_log(
            "Enter class TreeModelsReg : decision_tree_regressor function")

        try:
            dt = DecisionTreeRegressor()  # instantiating DecisionTreeRegressor object

            params = {'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                      'max_depth': [2, 5, 10, 20],
                      'min_samples_split': [2, 4, 8, 12],
                      'min_samples_leaf': [2, 4, 6, 8, 10]}  # parameter grid

            rcv = RandomizedSearchCV(estimator=dt, param_distributions=params, n_iter=10,
                                     scoring='r2', cv=10, verbose=2,
                                     random_state=42, n_jobs=-1,
                                     return_train_score=True)  # Randomized search cross validation object imported from sci-kit learn

            print('Cross validation process for Decision tree regressor')
            rcv.fit(self.x_train, self.y_train)  # fitting on the train data
            print()

            print('The best estimator for Decision tree regressor is ',
                  rcv.best_estimator_)  # display the best estimator

            dt = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final decision tree regressor.

            dt.fit(self.x_train, self.y_train)  # fitting on the train data.

            # Feature importance by the Decision tree regressor
            dt_feature_imp = pd.DataFrame(dt.feature_importances_, index=self.x_train.columns,
                                          columns=['Feature_importance'])
            dt_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the Decision tree regressor: ', dt_feature_imp)
            print()

            logger.add_info_log("class TreeModelsReg : decision_tree_regressor. Model "
                                "Build successfully")

            return dt

        except Exception as e:

            logger.add_exception_log(f'class TreeModelsReg : decision_tree_regressor. Model '
                                f'Build failed. Exception {str(e)}')

    def decision_tree_regressor_post_pruning(self):
        """Description: This method implements the post pruning technique to tackle over-fitting in the decision tree regressor.
        While doing so, we found out the optimum cost complexity pruning or ccp_alpha parameter as 0.8 in the
        'EDA + Model building.ipynb' jupyter notebook using visualization.
         Raises an exception if it fails

         returns
         -------------------------------
         The Decision tree regressor model post pruning
         """
        logger.add_info_log(
            "Enter class TreeModelsReg : decision_tree_regressor_post_pruning function")
        try:
            dt = DecisionTreeRegressor(random_state=42, ccp_alpha=0.8)  # instantiating the DecisionTreeRegressor object

            dt.fit(self.x_train, self.y_train)  # fitting the model

            logger.add_info_log("class TreeModelsReg : decision_tree_regressor_post_pruning. Model "
                                "Build successfully")

            return dt
        except Exception as e:

            logger.add_exception_log(f'class TreeModelsReg : decision_tree_regressor_post_pruning. Model '
                                f'Build failed. Exception {str(e)}')

    def random_forest_regressor(self):
        """Description: This method builds a model using RandomForestRegressor algorithm, a type of ensemble technique
        imported from sci-kit learn library. It uses cross validation technique and chooses the best estimator with the
        best hyper parameters.
        Raises an exception if it fails
        returns
        --------------------------------
        The Random forest regressor model and prints the importance of each feature
        """
        logger.add_info_log(
            "Enter class TreeModelsReg : random_forest_regressor function")
        try:
            rf = RandomForestRegressor()  # instantiating the RandomForestRegressor object

            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'criterion': ['mse', 'mae'],
                      'max_depth': [2, 5, 10, 20],
                      'min_samples_split': [2, 4, 8, 12],
                      'min_samples_leaf': [2, 4, 6, 8, 10],
                      'oob_score': [True]}  # parameter grid

            rcv = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=10, scoring='r2', cv=10,
                                     verbose=5,
                                     random_state=42, n_jobs=-1,
                                     return_train_score=True)  # instantiating RandomizedSearchCV

            print('Cross validation process for Random forest regressor')

            rcv.fit(self.x_train, self.y_train)  # Fitting on the train data
            print()

            print('The best estimator for the Random forest regressor is',
                  rcv.best_estimator_)  # displaying the best estimator

            rf = rcv.best_estimator_  # Building the best estimator recommended by the randomized search CV
            # as the final random forest regressor.

            rf.fit(self.x_train, self.y_train)  # fitting on the train data

            # Feature importance by the Random Forest regressor
            rf_feature_imp = pd.DataFrame(rf.feature_importances_, index=self.x_train.columns,
                                          columns=['Feature_importance'])
            rf_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)
            print()
            print('Feature importance by the Random Forest regressor: ', rf_feature_imp)
            print()

            logger.add_info_log("class TreeModelsReg : random_forest_regressor. Model "
                                "Build successfully")
            return rf

        except Exception as e:

            logger.add_exception_log(f'class TreeModelsReg : decision_tree_regressor_post_pruning. Model '
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

            pred = model.predict(X)

            return pred

        except Exception as e:

            logger.add_exception_log('Exception occurred in "model_predict" method of the TreeModelsReg class. Exception '
                          'message:' + str(e))  # logging operation




