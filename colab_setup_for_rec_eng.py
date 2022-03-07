import os, sys, time, glob, re, urllib, smtplib, gc, pickle, random, requests, shutil
from functools import partial
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date, datetime, timedelta
from itertools import product, filterfalse, combinations

def onColab(): return os.path.exists('/content')
def onGCP(): return os.path.exists('/home/jupyter')

bOnColab = onColab()
bOnGCP = onGCP()

if bOnColab:
  from google.colab import auth
  auth.authenticate_user()
  print('Authenticated')
  
if bOnColab and not os.path.exists('/content/drive'):       # presence of /content indicates you are on google colab
  from google.colab import drive
  drive.mount('/content/drive')
  print('gdrive mounted')


import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype, is_string_dtype, is_numeric_dtype, is_bool_dtype


from google.cloud import storage

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.text import text_to_word_sequence

from tensorflow.keras.utils import Sequence

from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Reshape, Dropout, Activation, Dot
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import concatenate, add, Lambda, Add
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

AUTO = tf.data.experimental.AUTOTUNE

import sklearn
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

import cv2
if bOnColab:
  from google.colab.patches import cv2_imshow


from ipywidgets import interact, Checkbox, Button, Output, HBox, VBox, AppLayout, Label, Layout, Text, Textarea
from ipywidgets import HTML as IPyWidgetHTML     # conflict with "from IPython.display import HTML"

import IPython.display as display
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

project_name = 'ListingRecommendation'
storage_project_id = 'royallepage.ca:api-project-267497502775'
bq_project_id = 'rlpdotca'

if bOnColab:
  home = Path('/content/drive/MyDrive')
  local = Path('/content')
  local_photos = local/'photos'

  if (home/'jumptools.com').exists(): bJumptools = True
  if bJumptools:
    utils_path = home/project_name/'utils'
    scripts_path = home/project_name/'scripts'
    data = home/project_name/'data'
    train_home = home/project_name/'training'
    photos = home/project_name/'photos'
    tmp = home/project_name/'tmp'
    labels_dir = home/project_name/'labels'
    models_dir = home/project_name/'model'

    gs_data = Path('ai-tests')/project_name/'data'

else:
  home = Path('/Users/kelvinchan/Google Drive (kelvin@jumptools.com)')
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
  sys.path.insert(0, str(scripts_path)) 

  from rlp_data_loader import RLPDataLoader
  from rlp_listing_data_loader import RLPListingDataLoader
  from rlp_data_preprocess import proc_NoneLookingStr_as_None, translate_from_fr_to_en, proc_extract_postal_code, proc_extract_fsa
  from rlp_data_preprocess import print_df_duration, print_df_size, print_df_dtype
  from rlp_data_preprocess import Bot, FastClicker, UserIdManager, rebuild_main, add_datetime_part
  from rlp_ml import build_colab_filter_model, build_colab_filter_model_nn
  from rlp_ml import plot_train_and_val_metric, plot_train_and_val_loss

  from rlp_rec_eng_v2 import RLPKerasRecommendationModeller, RLPListingRecommendator, SimilarListings
  from rlp_data_transforms_dev import RLPDataExplorer, RLPDataTransformer
  from rlp_EDA import EDA
  from rlp_deltaTs import compute_aux_dt_session_info

  from common_util import load_from_pickle, save_to_pickle, say_done
  from common_util import plot_training_loss, plot_loss_accuracy, plot_loss_and_metrics, combine_history
  from small_fastai_utils import join_df 

  from mem_util import mem_usage

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

def to_float(x):
  try:
    y = float(x)
    return y
  except:
    return np.NaN


def to_int(x):
  try:
    y = int(x)
    return y
  except:
    return np.NaN


def get_listingId(row):
  idx = row.name
  return listingid_regex.match(str(dev_idx_filenames[idx])).group(1)


def aggregate_data(user_shards, table_name, chkpt_name):
  '''
  Concat all data for that table for users in user_shards

  user_shards is a list of integer (the user shard #)
  '''
  dfs = []
  for k in user_shards:
    dataset_id = "combined_user_{}_frac_0.01".format(k)
    df = rlp_dataloader.load_checkpt(dataset_id=dataset_id, chkpt=chkpt_name, table_name=table_name)[0]
    dfs.append(df)

  return pd.concat(dfs, axis=0)


def sigmoid(x):
  return 1./(1. + np.exp(-x))


def get_notebook_name():
  return requests.get('http://172.28.0.2:9000/api/sessions').json()[0]['name']


def tmp_subdir():
  tmpdir = tmp/('.'+Path(get_notebook_name()).stem)
  if not tmpdir.exists(): tmpdir.mkdir()

  return tmpdir


