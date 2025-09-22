import pandas as pd
import re
df=pd.read_csv("/home/ubuntu/topn_qwen_edges.csv",sep=",")

def extract_numbers(text: str) -> str:
    """
    Extracts only digits from a given string and returns them as a single string.

    Args:
        text (str): Input string

    Returns:
        str: String containing only digits
    """
    if "." in text:
        text=text.split(".")[0]
    return ''.join(re.findall(r'\d', text))




edges=df["qwen_dss1_pos"]

edge_dict={}
max_edge=0


for i in range(400):
    edge=edges[i]
    st=edge.split("->")
    endpoint=extract_numbers(st[1])
    if endpoint =="":
        endpoint=extract_numbers(st[0])
    num=1
    if endpoint=="":
        endpoint="0"
    if max_edge < int(endpoint):
        max_edge=int(endpoint)
    if endpoint in edge_dict.keys():
        num=edge_dict[endpoint]+1
    edge_dict[endpoint]=num



print(edge_dict)
print(max_edge)



import matplotlib.pyplot as plt


categories=[]
values=[]
for i in range(max_edge+1):
    if str(i) in edge_dict.keys():
        categories.append(i)
        val=edge_dict[str(i)]-10
        if val < 0:
            val=0
        values.append(val)




# Create the bar chart
plt.bar(categories, values, color="purple")

# Add labels and title
plt.xlabel("Layers")
plt.xlim(0,24)
plt.ylim(0,30)
plt.xticks(ticks=categories,                    # positions
    labels=categories)
plt.ylabel("#Important Edges")

# Show the chart
plt.savefig("a.png")
