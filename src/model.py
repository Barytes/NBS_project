# src/model.py


class MD:
    def __init__(self, param):
        self.param = param
        self.cn = param["cn"]  # 能耗系数
        self.Fn = param["Fn"]  # 算力总量 (GHz)
        self.kn = param["kn"]  # 算力敏感度
        self.omega_n = param["omega_n"]  # 议价能力
        self.Rn = param["Rn"]  # 传输速率 (MBps)