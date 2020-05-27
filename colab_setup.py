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
data = home/project_name/'data'
train_home = home/project_name/'training'
photos = home/project_name/'photos'
tmp = home/project_name/'tmp'
labels_dir = home/project_name/'labels'
utils_path = home/project_name/'utils'

sys.path.insert(0, str(utils_path))

from rlp_data_loader import RLPDataLoader
from common_util import load_from_pickle, save_to_pickle, say_done
from common_util import plot_training_loss, plot_loss_accuracy, plot_loss_and_metrics, combine_history
from common_util import ImageLabelWidget
from small_fastai_utils import join_df

from common_util import ImageDataLoader, say

