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

device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print("WARNING: Running on CPU. Did you remember to set your Colab accelerator to GPU?")

model_name = 'gpt2-large'
model = HookedTransformer.from_pretrained(model_name, device=device)
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True


orig_str=" did a mistake, and everyone scolded"
corr_str="The xyz did a mistake, and everyone scolded"


Output_FB_FileName=""
Output_MB_FileName=""
"""# **Reading and Modifying Dataset**"""


df_pre = pd.read_csv('Professions.csv')

df = df_pre
sentence_column = []
corrupted_column = []

for index, row in df.iterrows():
    sentence_column.append("The"+ row['profession'] + orig_str)
    corrupted_column.append(corr_str)

df['sentence'] = sentence_column
df['corrupted'] = corrupted_column

df.drop(columns=['profession'], inplace=True)

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

gender_terms = {
    " ms": -1,
    " mrs": -1,
    " miss": -1,
    " missus": -1,
    " her": -1,
    " hers": -1,
    " herself": -1,
    " she": -1,
    " lady": -1,
    " woman": -1,
    " women": -1,
    " girl": -1,
    " girls": -1,
    " madam": -1,
    " ma'am": -1,
    " daughter": -1,
    " sister": -1,
    " aunt": -1,
    " mother": -1,
    " mom": -1,
    " mommy": -1,
    " grandmother": -1,
    " grandma": -1,
    " niece": -1,
    " lass": -1,
    " maiden": -1,
    " femme": -1,
    " dame": -1,
    " mademoiselle": -1,
    " daughter-in-law": -1,
    " stepmother": -1,
    " stepsister": -1,
    " niece-in-law": -1,
    " mr": 1,
    " him": 1,
    " his": 1,
    " himself": 1,
    " he": 1,
    " gentleman": 1,
    " man": 1,
    " men": 1,
    " boy": 1,
    " boys": 1,
    " sir": 1,
    " son": 1,
    " brother": 1,
    " uncle": 1,
    " father": 1,
    " dad": 1,
    " daddy": 1,
    " grandfather": 1,
    " grandpa": 1,
    " nephew": 1,
    " lads": 1,
    " dude": 1,
    " chap": 1,
    " fellow": 1,
    " guy": 1,
    " monsieur": 1,
    " stepson": 1,
    " stepfather": 1,
    " stepbrother": 1,
    " nephew-in-law": 1
}

MB_Probs = np.zeros(len(dataset))
FB_Probs = np.zeros(len(dataset))
GN_Probs = np.zeros(len(dataset))

MB_labels = [[] for _ in range(len(dataset))]
FB_labels = [[] for _ in range(len(dataset))]
GN_labels = [[] for _ in range(len(dataset))]

for i in range(len(dataset)):
  for j in range(k):
    bias = gender_terms.get(topk_pred[i,j].lower(), 0)
    if bias == 1:
      MB_labels[i].append(topk_pred[i,j])
      MB_Probs[i] += Probabilities[i,j]
    elif bias == -1 :
      FB_labels[i].append(topk_pred[i,j])
      FB_Probs[i] += Probabilities[i,j]
    else:
      GN_labels[i].append(topk_pred[i,j])
      GN_Probs[i] += Probabilities[i,j]

MB_dataset = []
FB_dataset = []
for i in range(len(dataset)):
  if(MB_Probs[i]>(FB_Probs[i])):
    MB_dataset.append(dataset[i])
  else:
    FB_dataset.append(dataset[i])

print(MB_dataset[0])

print(FB_dataset[0])

df = pd.DataFrame([(item[0],item[1]) for item in MB_dataset], columns=["clean","corrupted"])
df.to_csv(Output_MB_FileName, index=False)

df = pd.DataFrame([(item[0],item[1]) for item in FB_dataset], columns=["clean","corrupted"])
df.to_csv(Output_FB_FileName, index=False)




# Dataset Statistics Computation

print(f"Length of Male-Biased Dataset: {len(MB_dataset)}")
print(f"Length of Female-Biased Dataset: {len(FB_dataset)}")

MB_len = 0
FB_len = 0
for i in range(len(MB_dataset)):
  MB_len += len(model(MB_dataset[i][0]).squeeze(0).cpu())
for i in range(len(FB_dataset)):
  FB_len += len(model(FB_dataset[i][0]).squeeze(0).cpu())
avg_pos_len = MB_len/len(MB_dataset)
avg_neg_len = FB_len/len(FB_dataset)
print(f"Average token length of Male Bias dataset: {avg_pos_len}")
print(f"Average token length of Female Bias dataset: {avg_neg_len}")

df['#Male Bias labels'] = MB_labels
df['#Female Bias labels'] = FB_labels
df['#Gender Neutral Labels'] = GN_labels
df['#MB_Probs'] = MB_Probs
df['#FB_Probs'] = FB_Probs
df['#GN_Probs'] = GN_Probs
df['Bias_Type'] = df.apply(lambda row: 'Male Bias' if row['#MB_Probs'] > row['#FB_Probs'] else 'Female Bias', axis=1)
df.to_csv('gpt2large_gss1_m2c1_score.csv', index=False)

df_pos = df[df['Bias_Type'] == 'Male Bias']
df_neg = df[df['Bias_Type'] == 'Female Bias']

total_pos_words_1 = df_pos['#Male Bias labels'].apply(len).sum()
total_neg_words_1 = df_pos['#Female Bias labels'].apply(len).sum()
num_rows_1 = len(df_pos)
average_pos_words_1 = total_pos_words_1 / num_rows_1
average_neg_words_1 = total_neg_words_1/ num_rows_1
print(f'Average number of Male Bias words per row for Male Bias DataSet: {average_pos_words_1}')
print(f'Average number of Female Bias words per row for Male Bias DataSet: {average_neg_words_1}')

total_pos_words_2 = df_neg['#Male Bias labels'].apply(len).sum()
total_neg_words_2 = df_neg['#Female Bias labels'].apply(len).sum()
num_rows_2 = len(df_neg)
average_pos_words_2 = total_pos_words_2 / num_rows_2
average_neg_words_2 = total_neg_words_2 / num_rows_2
print(f'Average number of Male Bias per row for Female Bias DataSet: {average_pos_words_2}')
print(f'Average number of Female Bias per row for Female Bias DataSet: {average_neg_words_2}')

prob_pos_words_1 = df_pos['#MB_Probs'].sum()
prob_neg_words_1 = df_pos['#FB_Probs'].sum()
num_rows_1 = len(df_pos)
avg_prob_pos_words_1 = prob_pos_words_1 / num_rows_1
avg_prob_neg_words_1 = prob_neg_words_1 / num_rows_1
norm_prob_pos_words_1 = avg_prob_pos_words_1 / (avg_prob_pos_words_1 + avg_prob_neg_words_1)
norm_prob_neg_words_1 = avg_prob_neg_words_1 / (avg_prob_pos_words_1 + avg_prob_neg_words_1)

print(f'Normalized probability of Male Bias words per row for Male Bias DataSet: {norm_prob_pos_words_1}')
print(f'Normalized probability of Female Bias words per row for Male Bias DataSet: {norm_prob_neg_words_1}')

prob_pos_words_2 = df_neg['#MB_Probs'].sum()
prob_neg_words_2 = df_neg['#FB_Probs'].sum()
num_rows_2 = len(df_neg)
avg_prob_pos_words_2 = prob_pos_words_2 / num_rows_2
avg_prob_neg_words_2 = prob_neg_words_2 / num_rows_2
norm_prob_pos_words_2 = avg_prob_pos_words_2 / (avg_prob_pos_words_2 + avg_prob_neg_words_2)
norm_prob_neg_words_2 = avg_prob_neg_words_2 / (avg_prob_pos_words_2 + avg_prob_neg_words_2)

print(f'Normalized probability of Male Bias words per row for Female Bias DataSet: {norm_prob_pos_words_2}')
print(f'Normalized probability of Female Bias words per row for Female Bias DataSet: {norm_prob_neg_words_2}')



