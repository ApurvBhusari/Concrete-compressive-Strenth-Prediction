import logging


class LoggerFileClass:
    LOGGER_TYPE_FILE = "file"
    LOGGER_TYPE_MONGO_DB = "mongo_db"
    LOG_FILE_NM = "LoggerFile.log"
    """
         Description: This class is used to do logging of events, events like user action, exception, error.
    """

    def __init__(self, logger_nm):
        """
        Description: This function is used to create logger object. Initialization of logger done here.
        :param logger_nm:  name of logger

        """
        # Get Logger
        self.logger = logging.getLogger(logger_nm)
        # Creating Log formatter
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

        file_handler = logging.FileHandler(self.LOG_FILE_NM)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def add_debug_log(self, msg):
        """
        Description : Use to add debug log
        :param msg: Log message
        :return:
        """
        self.logger.debug(msg)

    def add_info_log(self, msg):
        """
        Description : Use to add info log
        :param msg: Log message
        :return:
        """
        self.logger.info(msg)

    def add_warning_log(self, msg):
        """
        Description : Use to add warning log
        :param msg: Log message
        :return:
        """
        self.logger.warning(msg)

    def add_exception_log(self, msg):
        """
        Description : Use to add exception log
        :param msg: Log message
        :return:
        """
        self.logger.exception(msg)

    def add_error_log(self, msg):
        """
        Description : Use to add error log
        :param msg: Log message
        :return:
        """
        self.logger.error(msg)
