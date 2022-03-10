import os, sys, time, glob, re, urllib, smtplib, gc, pickle, random, requests, shutil
from functools import partial
from pathlib import Path

from datetime import date, datetime, timedelta

bOnColab = Path('/content').exists()
bOnGCP = Path('/home/jupyter').exists()
bLocal = Path('/Users').exists()

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

from functools import partialmethod
pd.DataFrame.q_py = partialmethod(pd.DataFrame.query, engine='python')

from google.cloud import storage

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import seaborn as sns


project_name = 'AVMDataAnalysis'
storage_project_id = 'royallepage.ca:api-project-267497502775'
bq_project_id = 'rlpdotca'

if bOnColab:
  home = Path('/content/drive/MyDrive')
  local = Path('/content')

  utils_path = home/project_name/'utils'
  data = home/project_name/'data'
  tmp = home/project_name/'tmp'
else:
  home = Path('/Users/kelvinchan/Google Drive (kelvin@jumptools.com)')
  utils_path = home/project_name/'utils'


try:
  sys.path.insert(0, str(utils_path))

  from common_util import load_from_pickle, save_to_pickle

  from common_util import isNone_or_NaN

  from common_util import get_listingId_from_image_name

  from small_fastai_utils import join_df

except Exception as e:
  print(e)
  
