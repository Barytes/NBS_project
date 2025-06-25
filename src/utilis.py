# src/utils.py
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import logging
import json

import src.models as models

# ================= IO utils =================================================

def setup_logger(log_file):
    """
    设置 logger，将日志输出到文件和控制台
    """
    logger = logging.getLogger("sweeper_logger")
    logger.setLevel(logging.INFO)

    # 如果已有 handler 则清空，避免重复写入
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器，写入指定文件
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器（可选）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def log_experiment_result(exp_result, logger):
    """
    记录单个实验结果到日志，exp_result 包含：
      - "sweep_param": 当前 sweep 参数值（如 f_max、num_users 等）
      - "node_info": { "X":..., "Y":..., "U_current":..., ... }
      - "user_info": [ { user1 的详细信息 }, { user2 的详细信息 }, ... ]
    """
    logger.info("----- Single Experiment Result -----")
    logger.info("Sweep Parameter: %s", exp_result.get("sweep_param"))
    logger.info("SP Info:\n%s", json.dumps(exp_result.get("sp_info"), ensure_ascii=False, indent=2))
    logger.info("User Game Info:\n%s", json.dumps(exp_result.get("user_game_info"), ensure_ascii=False, indent=2))
    logger.info("User Info:")
    for user in exp_result.get("user_info", []):
        logger.info(json.dumps(user, ensure_ascii=False, indent=2))
    logger.info("----- End of Single Experiment -----\n")

def log_sweep_experiments_results(sweep_results, log_file):
    """
    记录多组实验结果到 log_file，
    sweep_results 是一个列表，每个元素都是一次实验的结果字典
    """
    logger = setup_logger(log_file)
    logger.info("===== Sweep Experiments Results =====")
    for idx, exp_result in enumerate(sweep_results):
        logger.info("===== Experiment %d =====", idx + 1)
        log_experiment_result(exp_result, logger)
    logger.info("===== End of Sweep Experiments Results =====")

def save_results(data, filename):
    # 获取目标目录
    directory = os.path.dirname(filename)
    
    # 如果目录不存在，则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 {directory} 已创建")
      
    """
    保存实验结果，data 可以是 dict 或 pandas DataFrame
    """
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data
    # df.to_csv(filename, index=False)
    df.to_csv(filename, index=False, mode="w", header=True)
    print(f"结果已保存到 {filename}")

def log_message(msg):
    print(f"[LOG] {msg}")