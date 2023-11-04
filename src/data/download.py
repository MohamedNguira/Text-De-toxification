import os
from pathlib import Path
import zipfile
import gdown


def download_checkpoints():
    HOME = os.getcwd()
    current = HOME 
    while 'src' not in os.listdir(current):
        current = Path(current).parent
    checkpoint_dir = str(current) + '\\checkpoints'
    # download checkpoints
    url ="https://drive.google.com/file/d/1OzdC7oYtZoQlIEWYPz0Z0OBvzGNMwiTB/view?usp=sharing"
    gdown.download(url, checkpoint_dir + '\\checkpoints.zip', quiet=False, fuzzy=True, use_cookies=False)
    with zipfile.ZipFile(checkpoint_dir + 'checkpoints.zip', 'r') as zip_ref:
      zip_ref.extractall(checkpoint_dir)
    os.remove('checkpoints.zip')

if __name__ == '__main__':

    download_checkpoints()

   
    
        

