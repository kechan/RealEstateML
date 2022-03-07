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


  
