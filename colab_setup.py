import os, gc, sys, time, pickle, pytz, multiprocessing, h5py, glob, re, PIL, base64, shutil, random
from pathlib import *
from datetime import date, datetime, timedelta
from IPython.display import HTML
from io import BytesIO

from google.colab import auth
auth.authenticate_user()
print('Authenticated')
  
from google.colab import drive
drive.mount('/content/drive')
print('gdrive mounted')

import pandas as pd
import numpy as np

from google.cloud import storage

from ipywidgets import interact, Checkbox, Button, Output, HBox, VBox, AppLayout, Label, Layout, Text, Textarea
from ipywidgets import HTML as IPyWidgetHTML     # conflict with "from IPython.display import HTML"

project_name = 'ListingImageClassification'

home = Path('/content/drive/My Drive')

if not (home/project_name).exists(): (home/project_name).mkdir()

if not (home/project_name/'data').exists(): (home/project_name/'data').mkdir()
if not (home/project_name/'training').exists(): (home/project_name/'training').mkdir()
if not (home/project_name/'photos').exists(): (home/project_name/'photos').mkdir()
if not (home/project_name/'photos'/'restructured_full_set').exists(): (home/project_name/'photos'/'restructured_full_set').mkdir()
if not (home/project_name/'tmp').exists(): (home/project_name/'tmp').mkdir()
if not (home/project_name/'labels').exists(): (home/project_name/'labels').mkdir() 
if not (home/project_name/'utils').exists(): (home/project_name/'utils').mkdir()

data = home/project_name/'data'
train_home = home/project_name/'training'
photos = home/project_name/'photos'
tmp = home/project_name/'tmp'
labels_dir = home/project_name/'labels'
utils_path = home/project_name/'utils'

if not (home/'jumptools.com').exists():   # don't do this if this is kelvin's jumptools.com gdrive
  if not Path('/content/photos').exists(): Path('/content/photos').mkdir()
    
  # setup GS, download and copy files
  storage_project_id = 'royallepage.ca:api-project-267497502775'
  storage_client = storage.Client(project=storage_project_id)
  storage_bucket = storage_client.get_bucket('ai-tests')

  # copy util python files
  blob = storage_bucket.blob('Labelling/utils/data_util.py')
  blob.download_to_filename(str(utils_path/'data_util.py'))

  blob = storage_bucket.blob('Labelling/utils/rlp_data_loader.py')
  blob.download_to_filename(str(utils_path/'rlp_data_loader.py'))

  blob = storage_bucket.blob('Labelling/utils/small_fastai_utils.py')
  blob.download_to_filename(str(utils_path/'small_fastai_utils.py'))

  blob = storage_bucket.blob('Labelling/utils/common_util.py')
  blob.download_to_filename(str(utils_path/'common_util.py'))

  blob = storage_bucket.blob('Labelling/utils/prefetch.py')
  blob.download_to_filename(str(utils_path/'prefetch.py'))

  # copy label and data
  blob = storage_bucket.blob('Labelling/labels/all_labels_df.csv')
  blob.download_to_filename(str(labels_dir/'all_labels_df.csv'))

  blob = storage_bucket.blob('Labelling/data/new_master_listing_df.csv')
  blob.download_to_filename(str(data/'new_master_listing_df.csv'))

  blob = storage_bucket.blob('Labelling/data/listing_extras.pickle')
  blob.download_to_filename(str(data/'listing_extras.pickle'))

  blob = storage_bucket.blob('Labelling/data/listing_df')
  blob.download_to_filename(str(data/'listing_df'))

  for shard in range(10):
    blob = storage_bucket.blob(f'Labelling/data/listing_es_{shard}_df')
    blob.download_to_filename(str(data/f'listing_es_{shard}_df'))

  # copy 2 photos: no image and loading 
  blob = storage_bucket.blob('Labelling/photos/no_image.jpg')
  blob.download_to_filename(str(photos/'no_image.jpg'))
  blob = storage_bucket.blob('Labelling/photos/loading_image.jpg')
  blob.download_to_filename(str(photos/'loading_image.jpg'))
  blob = storage_bucket.blob('Labelling/photos/404.jpg')
  blob.download_to_filename(str(photos/'404.jpg'))

  # copy prefetch.py to /content/
  shutil.copy(utils_path/'prefetch.py', '/content/prefetch.py')


sys.path.insert(0, str(utils_path))

from rlp_data_loader import RLPDataLoader
from common_util import load_from_pickle, save_to_pickle, say_done
from common_util import plot_training_loss, plot_loss_accuracy, plot_loss_and_metrics, combine_history
from common_util import ImageLabelWidget
from small_fastai_utils import join_df

from common_util import ImageDataLoader, say

