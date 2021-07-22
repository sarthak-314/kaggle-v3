import pandas as pd

DATASET_TARGETS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
u_one_features = ['Atelectasis', 'Edema']
u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']

def build_train_valid(fold, dataset_dir): 
    train = pd.read_csv(dataset_dir/'CheXpert-v1.0-small'/'train.csv')
    valid = pd.read_csv(dataset_dir/'CheXpert-v1.0-small'/'valid.csv')
    
    for split in 'train', 'valid': 
        df = {'train': train, 'valid': valid}[split]
        df['feature_string'] = df.apply(feature_string,axis = 1).fillna('')
        df['feature_string'] = df['feature_string'].apply(lambda x:x.split(";"))
        df['label'] = df.feature_string.apply(get_one_hot_label)
        df['img_path'] = df.Path.apply(lambda x: dataset_dir / x)
    return train, valid

def feature_string(row):
    feature_list = []
    for feature in u_one_features:
        if row[feature] in [-1,1]:
            feature_list.append(feature)
            
    for feature in u_zero_features:
        if row[feature] == 1:
            feature_list.append(feature)
            
    return ';'.join(feature_list)

def get_one_hot_label(feature_string):
    res = []
    for label in DATASET_TARGETS: 
        res.append(1 if label in feature_string else 0)
    return res