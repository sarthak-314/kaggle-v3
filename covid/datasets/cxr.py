import pandas as pd
from sklearn.model_selection import train_test_split

def build_train_valid(dataset_dir, gcs_path): 
    train = pd.read_csv(dataset_dir/'train.txt', sep=' ', header=None)
    train.columns = ['patient_id', 'filename', 'label', 'source']
    train.label.value_counts() # 6x more negative
    train['img_path'] = train.filename.apply(lambda f: gcs_path+'/train/'+f)
    train = train[train.filename.str.contains('png')]
    train.source.value_counts()
    train = train[train.source != 'actmed'] # BUG: BMP Images
    train, valid = train_test_split(train, test_size=0.2)
    return train, valid

