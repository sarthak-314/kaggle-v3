import tensorflow as tf
from tumor.tensorflow.datasets.utils import *

def process_df(df, png_gcs_path):
    dataset_name =  'rsna-miccai-png'
    gcs_path = get_gcs_path(dataset_name)
    gcs_filepaths = tf.io.gfile.glob(f'{png_gcs_path}/*/*/*/*png')
    gcs_filepaths_set = set(gcs_filepaths)

    def _get_gcs_path(dicom_src): 
        return dicom_src.replace('./', png_gcs_path+'/').replace('dcm', 'png') 
    
    def _is_empty(gcs_img_path):
        is_in_dataset =  str(gcs_img_path) in gcs_filepaths_set
        return not is_in_dataset

    df['img_path'] = df.dicom_src.apply(_get_gcs_path)
    df['is_empty'] = df.img_path.apply(_is_empty)
    df = df[df.is_empty==False]
    return df