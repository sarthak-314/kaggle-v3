# import pandas as pd
# for fold in [0, 1, 2, 3]: 
#     df_path = f'./rsna-miccai-brain-tumor-radiogenomic-classification/fold_{fold}.pkl'
#     df = pd.read_pickle(df_path)
#     df = df.loc[:,~df.columns.duplicated()]    
#     df.to_pickle(df_path)