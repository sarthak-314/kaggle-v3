# Login to AWS & WandB
import subprocess
def login(aws_access_key, aws_secret_key, wandb_api): 
    subprocess.run(['aws', 'configure', 'set', 'aws_access_key_id', aws_access_key])
    subprocess.run(['aws', 'configure', 'set', 'aws_secret_access_key', aws_secret_key])
    subprocess.run(['aws', 'configure', 'set', 'default.region', 'us-east-1'])
    
S3_PATH = 's3://siim-covid19-detection/'
def download_from_s3(dataset_name, output_path):
    output_path = str(output_path)
    bucket_path = S3_PATH + dataset_name
    subprocess.run(['aws', 's3', 'cp', bucket_path, output_path, '--recursive', '--quiet'], stdout=subprocess.PIPE)

def upload_to_s3(folder_path, dataset_name):
    folder_path = str(folder_path)
    bucket = S3_PATH + dataset_name
    subprocess.run(['aws', 's3', 'cp', folder_path, bucket, '--recursive', '--quiet'], stdout=subprocess.PIPE)