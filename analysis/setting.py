import os
import random
import numpy as np
import torch
import transformers
import pandas as pd


def now():
    current_directory = os.getcwd()
    print("현재 작업 디렉토리:", current_directory)
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    
    # GPU seed 고정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # PyTorch 재현성 설정 (CUDNN)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False