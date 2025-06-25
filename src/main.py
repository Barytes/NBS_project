# experiments/run_experiment.py
import numpy as np
import os
import pandas as pd
import json
import copy
from src.config import Config
from src.models import Task, Channel, User, Provider
from src.utils import log_message, save_results

def create_users(config,seed=None,real=False):
    rng = np.random.RandomState(seed)
    users = []
    for i in range(config.num_users):
        task_param = config.generate_random_task_param(rng)
        channel_param = config.generate_random_channel_param(rng,real)
        task = Task(d=task_param["d"], b=task_param["b"], alpha=task_param["alpha"])
        channel = Channel(trans_power=channel_param["trans_power"], channel_gain=channel_param["channel_gain"], background_noise=channel_param["background_noise"])
        local_cpu = config.generate_random_local_compute_power(rng)
        users.append(User(user_id=i, task=task, local_cpu=local_cpu, channel=channel))
    return users

def create_provider(config):
  provider = Provider(f_max=config.param_groups[0]["f_max"], B_max=config.param_groups[0]["B_max"], c_E=config.param_groups[0]["c_E"], c_N=config.param_groups[0]["c_N"])
  return provider


class ParamSweeper:
    def __init__(self, sweep_config_path):
        with open(sweep_config_path, 'r') as f:
            sweep_settings = json.load(f)
        
        self.vary_param = sweep_settings["vary_param"]
        self.values = sweep_settings["values"]
        self.fixed_config_file = sweep_settings["fixed_config_file"]

        with open(self.fixed_config_file, 'r') as f:
            self.fixed_config_data = json.load(f)

    def sweep_configs(self):
        configs = []
        for val in self.values:
            config_data = copy.deepcopy(self.fixed_config_data)
            if self.vary_param in ["f_max", "B_max", "c_E", "c_N"]:
                config_data["param_groups"][0][self.vary_param] = val
            elif self.vary_param == "num_users":
                config_data["num_users"] = val
            else:
                raise ValueError(f"暂不支持的 sweep 参数: {self.vary_param}")
            
            # 构造 Config 实例
            cfg = Config()
            cfg.__dict__.update(config_data)
            configs.append(cfg)
        return configs

def run_param_sweep(sweeper, solver_fn, result_dir="./results/"):
    os.makedirs(result_dir, exist_ok=True)
    results = []

    for i, cfg in enumerate(sweeper.sweep_configs()):
        users = create_users(cfg)
        provider = create_provider(cfg)
        
        try:
            X_star, U_best, _ = solver_fn(users, provider)
            results.append({
                "param_value": sweeper.values[i],
                "U_best": U_best,
                "offload_num": len(X_star),
                "f_max": provider.f_max,
                "B_max": provider.B_max,
                "c_E": provider.c_E,
                "c_N": provider.c_N,
                "num_users": len(users)
            })
        except Exception as e:
            print(f"Run {i} failed:", e)
            continue

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_dir, f"sweep_{sweeper.vary_param}.csv"), index=False)
    return df