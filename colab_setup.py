import os, gc, sys, time, pickle, pytz, multiprocessing, h5py, glob, re, PIL, base64, shutil, random, urllib, hashlib
import tempfile
import string
import requests
from pathlib import *
from functools import partial
from datetime import date, datetime, timedelta
from IPython.display import HTML
from io import BytesIO

def onColab(): return os.path.exists('/content')
bOnColab = onColab()

if bOnColab:
  from google.colab import auth
  auth.authenticate_user()
  print('Authenticated')
  
if bOnColab and not os.path.exists('/content/drive'):       # presence of /content indicates you are on google colab
  from google.colab import drive
  drive.mount('/content/drive')
  print('gdrive mounted')

#try:
#  !pip install tf-nightly --quiet
#except:
#  pass

import pandas as pd
from functools import partialmethod
pd.DataFrame.q_py = partialmethod(pd.DataFrame.query, engine='python')
pd.DataFrame.defrag_index = partialmethod(pd.DataFrame.reset_index, drop=True, inplace=True)

import numpy as np

from google.cloud import storage

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.utils import Sequence

from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Reshape, Dropout, Activation, Dot
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import backend as K

from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import RandomFlip, RandomTranslation, RandomRotation
from tensorflow.keras.layers import RandomCrop, RandomContrast, RandomZoom

from tensorflow.keras.applications import ResNet50

AUTO = tf.data.experimental.AUTOTUNE

import sklearn
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

try:
  import cv2
  if bOnColab:
    from google.colab.patches import cv2_imshow
except Exception as e:
  print(e)
  print("Not importing cv2. Investigate this if this is needed. This can happen if huggingface is installed.")

from ipywidgets import interact, Checkbox, Button, Output, HBox, VBox, AppLayout, Label, Layout, Text, Textarea
from ipywidgets import HTML as IPyWidgetHTML     # conflict with "from IPython.display import HTML"

# from ipywidgets import ValueWidget, CoreWidget

from IPython.display import Image, Audio

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import seaborn as sns
# sns.set(style='darkgrid', context='talk', palette='Dark2')
sns.set(rc={'figure.figsize': (11, 4)})

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.size'] = 16

bJumptools = False

project_name = 'ListingImageClassification'
storage_project_id = 'royallepage.ca:api-project-267497502775'
bq_project_id = 'rlpdotca'

if bOnColab:
  home = Path('/content/drive/MyDrive')
  local = Path('/content')
  local_photos = local/'photos'

  if (home/'jumptools.com').exists(): bJumptools = True
  if bJumptools:
    data = home/project_name/'data'
    train_home = home/project_name/'training'
    photos = home/project_name/'photos'
    tmp = home/project_name/'tmp'
    labels_dir = home/project_name/'labels'
    utils_path = home/project_name/'utils'
    models_dir = home/project_name/'model'

    gs_data = Path('ai-tests')/project_name/'data'
  else:
    data = home/'rlp'/project_name/'data'
    train_home = home/'rlp'/project_name/'training'
    utils_path = home/'rlp'/project_name/'utils'
    tmp = home/'rlp'/project_name/'tmp'
    labels_dir = home/'rlp'/project_name/'labels'
    models_dir = home/'rlp'/project_name/'model'

else:
  # home = Path('/Users/kelvinchan/Google Drive (kelvin@jumptools.com)')
  home = Path('/Users/kelvinchan/kelvin@jumptools.com - Google Drive/My Drive')
  data = home/project_name/'data'
  train_home = home/project_name/'training'
  photos = home/project_name/'photos'
  tmp = home/project_name/'tmp'
  labels_dir = home/project_name/'labels'
  utils_path = home/project_name/'utils'
  models_dir = home/project_name/'model'
  local = Path('/Users/kelvinchan/Documents/RoyalLePage')
  local_photos = local/'photos'

  gs_data = Path('ai-tests')/project_name/'data'

try:
  sys.path.insert(0, str(utils_path))

  from rlp_data_loader import RLPDataLoader
  from rlp_listing_data_loader import RLPListingDataLoader
  from common_util import load_from_pickle, save_to_pickle, say_done
  from common_util import plot_training_loss, plot_loss_accuracy, plot_loss_and_metrics, combine_history
  #from common_util import ImageLabelWidget
  from small_fastai_utils import join_df 

  from common_util import ImageDataLoader, say
  from common_util import isNone_or_NaN
  from common_util import image_d_hash, tf_image_d_hash
  from common_util import get_listingId_from_image_name, count_photos, get_orig_image_name_from_cropped
  from common_util import join_filter_drop_df, tf_dataset_peek
except Exception as e:
  print(e)
  print("Not installing rlp_dataloader, common_util and small_fastai_utils")

print("\u2022 Using TensorFlow Version:", tf.__version__)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
  fig, axes = plt.subplots(1, 5, figsize=(20,20))
  axes = axes.flatten()
  for img, ax in zip( images_arr, axes):
      ax.imshow(img)
      ax.axis('off')
  plt.tight_layout()
  plt.show()

# For inline image visualization  

pd.set_option('display.max_colwidth', None)
listingid_regex = re.compile(r'.*/(\d+)_\d+\.jpg')

def get_listingId(row):
  idx = row.name
  return listingid_regex.match(str(dev_idx_filenames[idx])).group(1)

def get_thumbnail(row):
  idx = row.name
  
  img = PIL.Image.fromarray((test_x[idx]*255).astype('uint8'), 'RGB')
  img.thumbnail((150, 150), PIL.Image.LANCZOS)

  return img

def image_base64(im):
  if isinstance(im, str):
      im = get_thumbnail(im)
  with BytesIO() as buffer:
      im.save(buffer, 'jpeg')
      return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
  return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

def sigmoid(x):
  return 1./(1. + np.exp(-x))

def get_notebook_name():
  return requests.get('http://172.28.0.2:9000/api/sessions').json()[0]['name']


def tmp_subdir():
  tmpdir = tmp/('.'+Path(get_notebook_name()).stem)
  if not tmpdir.exists(): tmpdir.mkdir()

  return tmpdir
