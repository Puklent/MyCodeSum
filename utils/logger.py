import os
import time
import logging
import logging.handlers

logger = logging.getLogger("logger")

file_name = "/data/alumpuk/github/TransCodeSum/tmp/" + time.asctime() + "-Info.log"
handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(filename=file_name)

logger.setLevel(logging.INFO)
handler1.setLevel(logging.INFO)
handler2.setLevel(logging.INFO)

# formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
formatter = logging.Formatter("[%(asctime)s] --%(levelname)s--: %(message)s")

handler1.setFormatter(formatter)
handler2.setFormatter(formatter)

logger.addHandler(handler1)
logger.addHandler(handler2)     

logger.info('[ ----------------------------New Start-------------------------- ]')
logger.info('[ Logger OK...                                                    ]')