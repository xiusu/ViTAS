import logging

logger_id = 0

def create_logger(name, log_file, level=logging.INFO):
    """create logger for training"""
    logger = logging.getLogger(name)
    global logger_id
    if logger_id == 1:
        return logger
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s]'
                                  '[line:%(lineno)4d][%(levelname)8s]%(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger_id = 1
    return logger
