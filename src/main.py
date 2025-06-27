# experiments/run_experiment.py
import numpy as np
import os
import pandas as pd
import json
import copy
from src.config import Config
from src.model import MD

def create_MDs(config,seed=None,real=False):
    rng = np.random.RandomState(seed)
    MDs = []
    for i in range(config.num_mds):
        md_param = config.generate_md_param(rng)
        MDs.append(MD(md_param))
    return MDs
