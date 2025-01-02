import numpy as np
from sklearn import metrics
import sys
from pathlib import Path


# 引入项目根目录
home_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(home_dir))

import utils.config_util as config_util

# 计算AUC指标
def cal_auc(labels, predict):
    if len(np.unique(labels)) < 2:
        return config_util.get_one_class_auc()
    return metrics.roc_auc_score(labels, predict)
