
"""
@Description: 
"""

import yaml
from easydict import EasyDict

def setup_config(path):
    """获取配置信息
    """
    with open(path, encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg