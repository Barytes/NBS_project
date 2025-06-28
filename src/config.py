# src/config.py
import json
import os
import numpy as np

class Config:
    def __init__(self, config_file):
        self.load_from_file(config_file) 

    def generate_md_param(self,rng):
        """
        生成一个随机的MD参数
        """
        md = {
            "s": self.params["s"],
            "l": self.params["l"],
            "cn": rng.uniform(*self.params["cn_range"]),  # 能耗系数
            "Fn": rng.uniform(*self.params["Fn_range"]),  # 算力总量 (GHz)
            "kn": rng.uniform(*self.params["kn_range"]),  # 算力敏感度
            "omega_n": self.params["omega_n"],  # 议价能力
            "Rn": self.generate_transmission_rate(rng),  # 传输速率 (MBps)
        } 
        return md

    def generate_transmission_rate(self,rng):
        """
        生成一个随机的传输速率
        """
        B = 1e7 # 10 MHz
        h = rng.exponential(scale=1.0) # small-scale fading channel power gain
        g0_dB=-40 # path-loss constant
        d0=1 # reference distance (m)
        varpi=4 # path-loss exponent
        distance=rng.uniform(100, 150) # user to base station distance (m)
        l0_linear = 10 ** (g0_dB / 10)
        channel_gain = h * l0_linear * (d0 / distance) ** varpi
        power = 10
        N0_dBm_per_Hz = -174
        N0_W_per_Hz   = 10 ** ((N0_dBm_per_Hz - 30) / 10)
        rate = B * np.log2(1 + power * channel_gain / (N0_W_per_Hz * B))
        rate = rate / 8e6  # 转换为 MBps
        return rate

    def load_from_file(self, config_file):
        with open(config_file, 'r') as f:
          data = json.load(f)
        self.params = data