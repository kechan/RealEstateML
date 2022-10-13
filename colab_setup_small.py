# To be used in colab:
# !curl -s -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/kechan/RealEstateML/master/colab_setup_small.py > /tmp/colab_setup_small.py
# %run /tmp/colab_setup_small.py

import os, shutil, sys, argparse
from pathlib import Path

bOnColab = Path('/content').exists()
bOnKaggle = Path('/kaggle/').exists()
bOnGCPVM = Path('/home/jupyter').exists()

description = 'Script to setup custom ML environment for Colab, Kaggle, GCP VM, and local machine.'

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--transformers', action='store_true', help='pip install transformers. Needed in Colab')
parser.add_argument('--jax', action='store_true', help='pip install jax. Needed in Colab')
parser.add_argument('--flax', action='store_true', help='pip install flax. Needed in Colab')

parser.add_argument('--skip_pip_realestate', action='store_true', help='skip pip install for realestate')

args = parser.parse_args()

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

  if args.transformers:
    os.system("pip -q install transformers")
  if args.jax:
    # os.system("pip -q install --upgrade jax jaxlib")     # jaxlib may be for CPU only.
    os.system("pip -q install --upgrade jax")
  if args.flax:
    os.system("pip -q install flax")    

if bOnKaggle:
  os.system("pip -q install gdown")

if (home/'Developer').exists():
  if len([p for p in sys.path if 'realestate-core' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-core'))
  if len([p for p in sys.path if 'realestate-nlp' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-nlp'))
  if len([p for p in sys.path if 'realestate-vision' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-vision'))
  if len([p for p in sys.path if 'realestate-vision-nlp' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-vision-nlp'))
  if len([p for p in sys.path if 'AVMDataAnalysis' in p]) == 0: sys.path.insert(0, str(home/'AVMDataAnalysis'/'monitoring'))
elif bOnKaggle:
  if not args.skip_pip_realestate:
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

try:
  from google.cloud import storage
  storage_client = storage.Client(project=gcp_storage_project_id)
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
    
