import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import dill

names = ['dan', 'sahil', 'yilin']
labels = ['STDDev', 'Skew', 'Kurtosis', 'Cluster']
rows = []
for name in names:
    with open(f"../../pickles/{name}.pkl", "rb") as f:
        try:
            print(name)
            person = dill.load(f)
            for i in range(3):
                rows.append([person.stddev[i], person.skew[i], person.kurtosis[i], 0 if name=='yilin' else 1 if name == 'sahil' else 2])
        except:
            print('exiting')
            exit()

overall = pd.DataFrame(columns=labels)
for ind in range(len(rows)):
    overall.loc[len(overall.index)] = rows[ind]

print(overall)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(overall['STDDev'])
y = np.array(overall['Skew'])
z = np.array(overall['Kurtosis'])

ax.scatter(x[0:3],y[0:3],z[0:3], marker="x", c=overall["Cluster"][0:3], s=40, cmap="RdBu")
ax.scatter(x[3:6],y[3:6],z[3:6], marker="+", c=overall["Cluster"][3:6], s=40, cmap="RdBu")
ax.scatter(x[6:9],y[6:9],z[6:9], marker="2", c=overall["Cluster"][6:9], s=40, cmap="RdBu")
#
plt.show()