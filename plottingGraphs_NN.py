import matplotlib.pyplot as plt
import numpy as np
import json

dataPath = './checkpoints/deepLabv3Plus/data.json'

iter = []
loss = []
acc_seg = []

with open(dataPath, 'r') as file:
    for line in file:
      try:
        # Parse the JSON object in each line
        data = json.loads(line)
        # Access the values in the JSON object
        iter.append(data['iter'])
        loss.append(data['loss'])
        acc_seg.append(data['decode.acc_seg'])
      except KeyError:
        pass

fig, ax = plt.subplots(1,2, figsize=(10,4))

ax[0].plot(np.array(iter), np.array(loss))
ax[0].set_title('Loss')

ax[1].plot(np.array(iter), np.array(acc_seg))
ax[1].set_title('Accuracy')
fig.supxlabel('Iterations')
plt.show()