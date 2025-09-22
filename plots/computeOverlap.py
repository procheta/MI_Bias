import pandas as pd

df_edges = pd.read_csv('/home/ubuntu/MI_Bias/Dataset/topn_qwen_edges_updated-2.csv')

top400_overlap_matrix = []
for x in ([df_edges['qwen_dss1_pos'], df_edges['qwen_dss1_neg'], df_edges['qwen_dss2_pos'], df_edges['qwen_gss1_pos'], df_edges['qwen_gss1_neg'], df_edges['qwen_gss2_pos'], df_edges['qwen_gss2_neg']]):
  for y in ([df_edges['qwen_dss1_pos'], df_edges['qwen_dss1_neg'], df_edges['qwen_dss2_pos'], df_edges['qwen_gss1_pos'], df_edges['qwen_gss1_neg'], df_edges['qwen_gss2_pos'], df_edges['qwen_gss2_neg']]):
    overlapping_edges=[]
    for i, _ in enumerate(x):
      for j, _ in enumerate(y):
        if(x[i]==y[j]):
          overlapping_edges.append(x[i])
    overlap_percentage = len(overlapping_edges)*100/(6000-len(overlapping_edges))
    top400_overlap_matrix.append(overlap_percentage)

corr_matrix = np.zeros((8, 8))
n=0
for i in range(8):
  for j in range(8):
    corr_matrix[i][j] = top400_overlap_matrix[j+n]
  n+=8

import seaborn as sns
corr_matrix = np.array(corr_matrix)

labels = ['Pos_DSS1', 'Neg_DSS1', 'Pos_DSS2','Male_GSS1', 'Female_GSS1', 'Male_GSS2', 'Female_GSS2', ]
df = pd.DataFrame(corr_matrix, index=labels, columns=labels)

plt.figure(figsize=(12, 6))
sns.heatmap(df, cmap='Blues', fmt=".2f")
plt.title('Percentage Overlap: GPT2-Small (Top 400 edges)')
plt.savefig('Percentage Overlap M2C1 for GPT2-Small for all combinations(Top 400 edges).jpg')

