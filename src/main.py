# experiments/run_experiment.py
import numpy as np
import os
import pandas as pd
import json
import copy
from src.config import Config
from src.model import MD,ESP

def create_MDs(config,seed=None):
    rng = np.random.RandomState(seed)
    MDs = []
    for i in range(config.params["num_mds"]):
        md_param = config.generate_md_param(rng)
        MDs.append(MD(md_param))
    return MDs

def create_ESP(config,seed=None):
    ESP_param = {
        "lambda0": config.params["lambda0"],
        "D0": config.params["D0"],
        "theta": config.params["theta"],
        "o": config.params["o"],
        "omega_0": config.params["omega_0"],
        "s": config.params["s"],
        "l": config.params["l"]
    }
    return ESP(ESP_param)

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

    def run_param_sweep(self, solver_fn, result_dir="./results/"):
        os.makedirs(result_dir, exist_ok=True)
        results = []

        for i, cfg in enumerate(self.sweep_configs()):
            users = create_users(cfg)
            provider = create_provider(cfg)
            
            try:
                X_star, U_best, _ = solver_fn(users, provider)
                results.append({
                    "param_value": self.values[i],
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
        df.to_csv(os.path.join(result_dir, f"sweep_{self.vary_param}.csv"), index=False)
        return df
