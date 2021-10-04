import os
from nltk import data
import torch
data.path.append(os.environ["HOME"]+"/nltk_data")
import nltk
nltk.download("punkt", quiet=True)