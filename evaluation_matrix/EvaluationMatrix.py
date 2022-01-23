from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

import warnings

warnings.filterwarnings('ignore')

from logger.LoggerClass import LoggerFileClass

logger = LoggerFileClass("evaluation_matrix")


class EvaluationMetrix:
    """This class is used to evaluate the models by returning their performance metrics.
       Author: Bhushan Patil
       """

    def __init__(self):
        pass

    def r2_score(self, y_true, y_pred):
        """Description: This method calculates the r2_score of the model, which tells us how much variance our model
        can explain on the given data. This method uses r2_score method imported from the sci-kit learn.
        Raises an exception if it fails

        parameters
        --------------------------------
        y_true: Dataframe containing the actual values of the dependent or the target feature
        y_pred: Dataframe containing the predicted values of the dependent or the target feature

        returns
        --------------------------------
        r2 score of the model
        """
        logger.add_info_log('Enter class EvaluationMetrix : r2_score function')  # logging operation

        try:
            score = r2_score(y_true, y_pred)
            logger.add_info_log(f' class EvaluationMetrix : r2_score function Exit.. r2_score {str(score)}')
            return score

        except Exception as e:
            logger.add_exception_log(
                f' class EvaluationMetrix : r2_score function .. Exception raised {str(e)}')

    def rmse_score(self, y_true, y_pred):
        """Description: Calculates the root mean square error.
        Raises an exception if it fails

        parameters
        --------------------------------
        y_true: Dataframe containing the actual values of the dependent or the target feature
        y_pred: Dataframe containing the predicted values of the dependent or the target feature

        returns
        --------------------------------
        root mean square error of the model
        """
        logger.add_info_log('Enter class EvaluationMetrix : rmse_score function')  # logging operation

        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logger.add_info_log(f' class EvaluationMetrix : rmse_score function Exit.. adj_r2 {str(rmse)}')

            return rmse

        except Exception as e:
            logger.add_exception_log(
                f' class EvaluationMetrix : rmse_score function .. Exception raised {str(e)}')

    def adj_r2_score(x, y_true, y_pred):
        """Description: Calculates the adjusted r2_score of the model.
        Raises an exception if it fails.

        parameters
        ---------------------------------
        x: Dataframe containing the independent features
        y_true: Dataframe containing the actual values of the dependent or the target feature
        y_pred: Dataframe containing the predicted values of the dependent or the target feature

        returns
        ---------------------------------
        adjusted r2 score of the model

        """
        logger.add_info_log('Enter class EvaluationMetrix : adj_r2_score function')  # logging operation

        try:
            r2 = r2_score(y_true, y_pred)
            n = x.shape[0]
            p = x.shape[1]
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            logger.add_info_log(f' class EvaluationMetrix : adj_r2_score function Exit.. adj_r2 {str(adj_r2)}')

            return adj_r2

        except Exception as e:
            logger.add_exception_log(
                f' class EvaluationMetrix : adj_r2_score function .. Exception raised {str(e)}')
