# src/Model.py
from Config import Config


class Model:
    """
    系统模型封装。所有数学公式、参数、约束都在此定义，
    并提供必要的接口供求解器调用。
    """

    def __init__(self, config: Config):
        mp: Dict[str, Any] = config.model_params
        # 示例：读取模型参数
        self.lambda0: float = mp.get("lambda0", 1.0)
        self.F_n: Dict[str, Any] = mp.get("F_n", {})
        self.D0: float = mp.get("D0", 1.0)
        # TODO: 根据你的 NBS 问题结构，继续加载其他参数

    def compute_social_welfare(self, decision: Dict[str, Any]) -> float:
        """
        根据给定的决策变量计算社会福利值。
        decision: 如 {"lambda": [...], "p": [...], "D_max": ...}
        """
        # TODO: 用你的效用函数/损失函数实际实现
        raise NotImplementedError("请在 Model 中实现 compute_social_welfare")

    # 你还可以在这里添加：约束检查、各玩家收益计算等方法
