# encoding: utf-8
"""
@author: sunkai
@contact: moguizhiz@126.com
@time: 2020/10/20 17:58
@file: logger.py
@desc: 
"""

import os
import logging
import time


def config_logger(log_prefix):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        level=logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.getcwd() + '/logs/' + log_prefix + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + rq + '.log'
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    return logger
