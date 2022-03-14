import requests
import json

import pandas as pd
import numpy as np

data = pd.read_csv("/content/dialect_dataset.csv")
ids = data['id'].tolist

left = 0
right = len(ids) - 1
list_ = []
data_ = []

while left < right:
    list_.extend([str(ids[left]),str(ids[right])])
    jsonStr = json.dumps(list_)
    data = requests.post("https://recruitment.aimtechnologies.co/ai-tasks", data = jsonStr)
    data_.append([data.json()])
    list_ = []
    left += 1
    right -= 1
    
with open('/content/drive/MyDrive/data_dialect.json', 'w') as f:
    json.dump(data_, f)  