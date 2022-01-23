import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')

from logger.LoggerClass import LoggerFileClass

logger = LoggerFileClass("boosting_model")


class DataPreprocessing:
    """This class is used to preprocess the data for modelling
        Author: Bhushan Patil

        parameters
        _________________________________________
        dataframe: A pandas dataframe that has to be preprocessed
        """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_outliers(self, column_name):
        """ Description: This method removes outliers from the specified column using Inter quartile range method.
        Here, first we consider the values which are at the upper and lower limits and store it in one dataframe say data_inc.
        Then, we exclude the values which are at the upper and lower limits and store it in one dataframe say data_esc.
        Then, we concatenate both the data frames into a single dataframe.
        Raises an exception if it fails.

        parameters
        ----------------------------
        column_name: Column for which the outliers has to be removed.

        returns
        -----------------------------
        returns a dataframe having outliers removed in the given column.
        """
        logger.add_info_log(
            "Enter class DataPreprocessing : remove_outliers function")

        try:
            q1 = self.dataframe[column_name].quantile(0.25)  # Quartile at 25th percentile
            q3 = self.dataframe[column_name].quantile(0.75)  # Quartile at 75th percentile
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            data_inc = self.dataframe.loc[(self.dataframe[column_name] >= lower_limit) &
                                          (self.dataframe[column_name] <= upper_limit)]

            logger.add_info_log(
                f'Outlier treatment using IQR method: Successfully removed outliers in the {column_name} column. '
                f'So, now the shape is {self.dataframe.shape}')
            logger.add_info_log('Exited the remove_outliers method of the DataPreprocessor class ')

            return data_inc

        except Exception as e:
            logger.add_exception_log(f'class DataPreprocessing : remove_outliers. Exception raised {str(e)}')

    def data_split(self, test_size):
        """ Description: This method splits the dataframe into train and test data respectively
        using the sklearn's "train_test_split" method.
        Raises an exception if it fails.

        parameters
        ------------------------------
        test_size: Percentage of the Dataframe to be taken as a test set

        returns
        ------------------------------
        training and testing dataframes respectively.
        """
        logger.add_info_log(
            "Enter class DataPreprocessing : data_split function")

        try:
            df_train, df_test = train_test_split(self.dataframe, test_size=test_size, shuffle=True, random_state=42)

            logger.add_info_log(
                f'class DataPreprocessing : data_split function ... Train test split successful. The shape of train '
                f'data set is {df_train.shape} and the shape of '
                f'test data set is {df_test.shape}')

            return df_train, df_test

        except Exception as e:
            logger.add_exception_log(f'class DataPreprocessing : data_split function. Exception raised {str(e)}')

    def feature_scaling(self, df_train, df_test):
        """ Description: This method scales the features of both the train and test datasets
        respectively, using the sklearn's "StandardScaler" method.
        Raises an exception if it fails.

        parameters
        --------------------------------
        df_train: A pandas dataframe representing the training data set
        df_test: A pandas dataframe representing the testing data set

        returns
        --------------------------------
        training and testing dataframes in a scaled format.
        """
        # logging operation
        logger.add_info_log("Enter class DataPreprocessing : feature_scaling function")

        try:
            columns = df_train.columns
            scaler = StandardScaler()
            df_train = scaler.fit_transform(df_train)
            df_test = scaler.transform(df_test)

            logger.add_info_log('class DataPreprocessing : feature_scaling function ...Feature scaling of both train '
                                'and test datasets successful. ')

            df_train = pd.DataFrame(df_train, columns=columns)  # converting the numpy arrays into pandas Dataframe
            df_test = pd.DataFrame(df_test, columns=columns)  # converting the numpy arrays into pandas Dataframe
            logger.add_info_log('class DataPreprocessing : feature_scaling function ...Exit ')
            return df_train, df_test

        except Exception as e:
            logger.add_exception_log(f'class DataPreprocessing : feature_scaling function. Exception raised {str(e)}')

    def splitting_as_x_y(self, df_train, df_test, column_name):
        """Description: This method splits the data into dependent and independent variables respectively
        i.e., X and y.
        Raises an exception if it fails.

        parameters
        -------------------------------
        df_train: A pandas dataframe representing the training data set
        df_test: A pandas dataframe representing the testing data set
        column_name: Target column or feature, which has to be predicted using other features

        returns
        -------------------------------
        independent and dependent features of the both training and testing datasets respectively.
        i.e., df_train into X_train, y_train and df_test into X_test, y_test respectively.
        """
        # logging operation
        logger.add_info_log('Enter class DataPreprocessing : splitting_as_x_y function')

        try:
            x_train = df_train.drop(column_name, axis=1)
            y_train = df_train[column_name]
            x_test = df_test.drop(column_name, axis=1)
            y_test = df_test[column_name]
            logger.add_info_log('class DataPreprocessing : splitting_as_x_y function ...Exit')
            return x_train, y_train, x_test, y_test

        except Exception as e:
            logger.add_exception_log(f'class DataPreprocessing : splitting_as_x_y function. Exception raised {str(e)}')

