import sys 
sys.path.append('/kaggle/working/temp')
sys.path.append('/content/temp')

from kaggle_utils.startup import * 

COMP_NAME = 'nfl-health-and-safety-helmet-assignment'
DRIVE_DIR = Path('/content/drive/MyDrive/NFL')

def download_and_unzip_files(): 
    assert ENV == 'Colab'
    os.system("mkdir -p ~/.kaggle")
    os.system("cp '/content/drive/MyDrive/kaggle.json' ~/.kaggle/")
    os.system("cat ~/.kaggle/kaggle.json ")
    os.system("chmod 600 ~/.kaggle/kaggle.json")

    os.system("kaggle competitions download -c nfl-health-and-safety-helmet-assignment -p $TMP_DIR")
    os.system("unzip $TMP_DIR/'*.zip' -d $TMP_DIR")