# To be used in colab:
# !curl -s -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/kechan/RealEstateML/master/colab_setup_small.py > colab_setup_small.py
# %run colab_setup_small.py

import os, shutil, sys
from pathlib import Path
from google.cloud import storage

bOnColab = Path('/content').exists()
bOnKaggle = Path('/kaggle/').exists()
bOnGCPVM = Path('/home/jupyter').exists()

if bOnColab and not Path('/content/drive').exists():
  from google.colab import drive
  drive.mount('/content/drive')

if bOnColab:
  home = Path('/content/drive/MyDrive')
elif bOnKaggle:
  home = Path('/kaggle/working')
elif bOnGCPVM:
  home = Path('/home/jupyter')
else:
  home = Path('/Users/kelvinchan/kelvin@jumptools.com - Google Drive/My Drive')

if bOnColab:
  os.system("pip -q install pyfarmhash")

if bOnKaggle:
  os.system("pip -q install gdown")

if (home/'Developer').exists():
  if len([p for p in sys.path if 'realestate-core' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-core'))
  if len([p for p in sys.path if 'realestate-nlp' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-nlp'))
  if len([p for p in sys.path if 'realestate-vision' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-vision'))
  if len([p for p in sys.path if 'realestate-vision-nlp' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-vision-nlp'))
  if len([p for p in sys.path if 'AVMDataAnalysis' in p]) == 0: sys.path.insert(0, str(home/'AVMDataAnalysis'/'monitoring'))
elif bOnKaggle:
  os.system('pip -q install git+https://github.com/kechan/realestate-core')
  os.system('pip -q install git+https://github.com/kechan/realestate-vision')
  os.system('pip -q install git+https://github.com/kechan/realestate-nlp')
  os.system('pip -q install git+https://github.com/kechan/realestate-vision-nlp')
elif bOnGCPVM:
  pass   # pip install manually, env is persistent 
else:
  # git clone all (except AVMDataAnalysis) 
  os.system('git clone https://github.com/kechan/realestate-core.git')
  os.system('git clone https://github.com/kechan/realestate-nlp.git')
  os.system('git clone https://github.com/kechan/realestate-vision.git')
  os.system('git clone https://github.com/kechan/realestate-vision-nlp.git')

  if len([p for p in sys.path if 'realestate-core' in p]) == 0: sys.path.insert(0, 'realestate-core')
  if len([p for p in sys.path if 'realestate-nlp' in p]) == 0: sys.path.insert(0, 'realestate-nlp')
  if len([p for p in sys.path if 'realestate-vision' in p]) == 0: sys.path.insert(0, 'realestate-vision')
  if len([p for p in sys.path if 'realestate-vision-nlp' in p]) == 0: sys.path.insert(0, 'realestate-vision-nlp')


if bOnColab:
  if len([p for p in sys.path if 'TFRecordHelper' in p]) == 0:
    try: 
      # !git clone https://github.com/kechan/TFRecordHelper.git      
      os.system("git clone https://github.com/kechan/TFRecordHelper.git")      
    except: pass
    sys.path.insert(0, 'TFRecordHelper')
elif bOnKaggle:
  os.system("pip -q install git+https://github.com/kechan/TFRecordHelper")
elif bOnGCPVM:
  pass   # pip install manually, env is persistent  
else:
  if len([p for p in sys.path if 'TFRecordHelper' in p]) == 0:
    sys.path.insert(0, '/Users/kelvinchan/Developer/TFRecordHelper')


import realestate_core.common.class_extensions
import realestate_vision.common.class_extensions
from realestate_vision.common.modules import *

if bOnColab:
  try: shutil.rmtree(Path('/content/sample_data'))
  except: print('/content/sample_data not found.')

def authenticate_user():
  from google.colab import auth
  auth.authenticate_user() 

def get_GOOGLE_APPLICATION_CREDENTIALS():
  import os; 
  return os.environ['GOOGLE_APPLICATION_CREDENTIALS']

os.environ["KERAS_SD_HOME"] = str(home/'Keras_SD')

# GCP Cloud Storage
gcp_storage_project_id = 'royallepage.ca:api-project-267497502775'
storage_client = storage.Client(project=gcp_storage_project_id)
try:
  bucket = storage_client.get_bucket('ai-tests')
except:
  bucket = None

def download_from_gcs(blob_path: str, dest_dir: str = None, debug=False):
  try:
    blob = bucket.blob(blob_path)
    if dest_dir is None:
      # download to local and use the same filename
      target_path = str(Path(blob_path).name)
    else:
      target_path = str(Path(dest_dir)/Path(blob_path).name)

    blob.download_to_filename(target_path)
  except Exception as ex:
    if debug:
      raise   
    else:
      print('Something has gone wrong. Please debug.')
    
