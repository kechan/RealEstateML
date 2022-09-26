# To be used in colab:
# !curl -s -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/kechan/RealEstateML/master/colab_setup_small.py > colab_setup_small.py
# %run colab_setup_small.py

import os, sys
from pathlib import Path

bOnColab = Path('/content').exists()

if bOnColab and not Path('/content/drive').exists():
  from google.colab import drive
  drive.mount('/content/drive')

if bOnColab:
  home = Path('/content/drive/MyDrive')
else:
  home = Path('/Users/kelvinchan/kelvin@jumptools.com - Google Drive/My Drive')

if bOnColab:
  os.system("pip -q install pyfarmhash")

if len([p for p in sys.path if 'realestate-core' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-core'))
if len([p for p in sys.path if 'realestate-nlp' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-nlp'))
if len([p for p in sys.path if 'realestate-vision' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-vision'))
if len([p for p in sys.path if 'realestate-vision-nlp' in p]) == 0: sys.path.insert(0, str(home/'Developer'/'realestate-vision-nlp'))
if len([p for p in sys.path if 'AVMDataAnalysis' in p]) == 0: sys.path.insert(0, str(home/'AVMDataAnalysis'/'monitoring'))

if bOnColab:
  if len([p for p in sys.path if 'TFRecordHelper' in p]) == 0:
    try: 
      # !git clone https://github.com/kechan/TFRecordHelper.git      
      os.system("git clone https://github.com/kechan/TFRecordHelper.git")      
    except: pass
    sys.path.insert(0, 'TFRecordHelper')

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
  if bOnColab:
    from google.colab import auth
    auth.authenticate_user() 
