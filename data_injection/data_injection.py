import pandas as pd
from logger.LoggerClass import LoggerFileClass

logger = LoggerFileClass("data_injection")

class DataLoader :
    """ This class is used to fetch data for the training.
        Author: Bhushan Patil
        """
    def __init__(self,dataset):
        self.dataset = dataset

    def fetch_data(self):
        """ Description: This method reads data from source and returns a pandas dataframe
               Raises an exception if it fails
                parameters
               -------------------------------------
               dataset: dataset in the .csv format

               returns
               -------------------------------------
               Given dataset in the form of Pandas Dataframe
               """
        logger.add_info_log("In DataLoader, fetch_data function start")

        try:
            df = pd.read_csv(self.dataset)
            logger.add_info_log(f"In DataLoader, fetch_data function, successfully read data.. shape {df.shape}")
            return df

        except Exception as e:
            logger.add_exception_log(f"In DataLoader, fetch_data function, exception raised data load failed... {str(e)}")

