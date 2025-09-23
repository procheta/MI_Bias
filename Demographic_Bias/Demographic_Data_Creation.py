import sys, os
import torch
from torch import Tensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Union, Optional, Tuple, Literal, Callable
from functools import partial
from IPython.display import Image, display
from tqdm import tqdm
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, ActivationCache
import plotly.io as pio

from transformers import pipeline
from pprint import pprint
from einops import einsum

pio.renderers.default = "colab"

Output_Pos_FileName="output_pos.csv"
Output_Neg_FileName="output_neg.csv"
Nationality_File_Name="/home/ubuntu/MI_Bias/Dataset/Nationalities.csv"
corr_string="As Expected, abc people are so"
input_string="people are so"


device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print("WARNING: Running on CPU. Did you remember to set your Colab accelerator to GPU?")

model_name = 'Qwen/Qwen2-0.5B'
model = HookedTransformer.from_pretrained(model_name, device=device)
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

"""# **Reading and Modifying Dataset**"""

s = corr_string
logits=model(s).squeeze(0).cpu()
probs = torch.softmax(logits, dim=-1)
probs, next_tokens = torch.topk(probs[-1], 5)
for prob, token_id in zip(probs, next_tokens):
  token = model.tokenizer.decode(token_id.item())

df_pre = pd.read_csv(Nationality_File_Name)

# Forming the "{Nationality} people are so" dataset.
df = df_pre
sentence_column = []
corrupted_column = []

for index, row in df.iterrows():
    sentence_column.append(f"As expected, {row['Nationality']} {input_string}")
    corrupted_column.append(corr_string)

df['sentence'] = sentence_column
df['corrupted'] = corrupted_column

df.drop(columns=['Nationality'], inplace=True)


def batch_dataset(df, batch_size=1):
    sentence, corrupted = [df[col].tolist() for col in ['sentence', 'corrupted']]
    sentence = [sentence[i:i+batch_size] for i in range(0, len(df), batch_size)]
    corrupted = [corrupted[i:i+batch_size] for i in range(0, len(df), batch_size)]
    return [(sentence[i], corrupted[i]) for i in range(len(sentence))]

dataset = batch_dataset(df, batch_size=1)

"""# **Creating Positive and Negative Datasets, Scores**"""

s_list = []
# Here k denotes the number of topk predictions.
k=10

topk_pred = np.empty((len(dataset), k), dtype=object)
predicted = np.empty((len(dataset), k), dtype=object)
Probabilities = np.zeros((len(dataset),k), dtype=float)

for i, (sentence,_) in enumerate(dataset):
  s_list.append(sentence)
  logits = model(s_list[i]).squeeze(0).cpu()
  probs = torch.softmax(logits, dim=-1)
  probs, next_tokens = torch.topk(probs[-1], k)
  for j, (prob, token_id) in enumerate(zip(probs, next_tokens)):
    token = model.tokenizer.decode(token_id.item())
    predicted[i,j] = s_list[i][0] + token  # Append the predicted token to the current text
    topk_pred[i,j] = token
    Probabilities[i,j] = prob.item()

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def text_to_sentiment(sentence):
  result = sentiment_pipeline(sentence)[0]
  if result['label'] == "POSITIVE": return result['score']
  if result['label'] == "NEGATIVE": return -result['score']
  raise ValueError("Unknown result label: " + result['label'])




Senti_Scores = np.empty((len(dataset), k), dtype=object)
Total_Positive_Score = np.zeros(len(dataset))
Total_Negative_Score = np.zeros(len(dataset))

Positive_Probs = np.zeros(len(dataset))
Negative_Probs = np.zeros(len(dataset))

positive_sentiment_labels = [[] for _ in range(len(dataset))]
negative_sentiment_labels = [[] for _ in range(len(dataset))]

for i in range(len(dataset)):
  for j in range(k):
    Senti_Scores[i,j] = text_to_sentiment(predicted[i,j])
    if Senti_Scores[i,j] >= 0:
      Total_Positive_Score[i] += Senti_Scores[i,j]
      positive_sentiment_labels[i].append(topk_pred[i,j])
      Positive_Probs[i] += Probabilities[i,j]
    else:
      Total_Negative_Score[i] += Senti_Scores[i,j]
      negative_sentiment_labels[i].append(topk_pred[i,j])
      Negative_Probs[i] += Probabilities[i,j]

pos_dataset = []
neg_dataset = []
for i in range(len(dataset)):
  if(Positive_Probs[i]>(Negative_Probs[i])):
    pos_dataset.append(dataset[i]) ### Countries where gpt2 shows more positive sentiment
  else:
    neg_dataset.append(dataset[i]) ### Countries where gpt2 shows more negative sentiment

df = pd.DataFrame([(item[0],item[1]) for item in pos_dataset], columns=["clean","corrupted"])
df.to_csv(Output_Pos_FileName, index=False)


df = pd.DataFrame([(item[0],item[1]) for item in neg_dataset], columns=["clean","corrupted"])
df.to_csv(Output_Neg_FileName, index=False)
