import os

def solve_environment(): 
    if 'COLAB_GPU' in os.environ.keys(): 
        return 'COLAB'
    else: 
        return 'KAGGLE'