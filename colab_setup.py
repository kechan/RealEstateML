import os, gc, sys, time, pickle, pytz, multiprocessing, h5py, glob, re, PIL, base64, shutil, random, urllib
import tempfile
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
import numpy as np

from google.cloud import storage

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.utils import Sequence

from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Reshape, Dropout, Activation, Dot
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import backend as K

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomTranslation, RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, RandomContrast, RandomZoom

import sklearn
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

import cv2

from ipywidgets import interact, Checkbox, Button, Output, HBox, VBox, AppLayout, Label, Layout, Text, Textarea
from ipywidgets import HTML as IPyWidgetHTML     # conflict with "from IPython.display import HTML"

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

if bOnColab:
  home = Path('/content/drive/My Drive')
  if (home/'jumptools.com').exists(): bJumptools = True
  if bJumptools:
    data = home/project_name/'data'
    train_home = home/project_name/'training'
    photos = home/project_name/'photos'
    tmp = home/project_name/'tmp'
    labels_dir = home/project_name/'labels'
    utils_path = home/project_name/'utils'
    local = Path('/content')
    local_photos = local/'photos'
  else:
    data = home/'rlp'/project_name/'data'
    train_home = home/'rlp'/project_name/'training'
    utils_path = home/'rlp'/project_name/'utils'
    local = Path('/content')
    local_photos = local/'photos'

sys.path.insert(0, str(utils_path))

from rlp_data_loader import RLPDataLoader
from common_util import load_from_pickle, save_to_pickle, say_done
from common_util import plot_training_loss, plot_loss_accuracy, plot_loss_and_metrics, combine_history
from common_util import ImageLabelWidget
from small_fastai_utils import join_df

from common_util import ImageDataLoader, say
from common_util import isNone_or_NaN
from common_util import get_listingId_from_image_name

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
