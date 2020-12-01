#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

data = np.array([
    [0, 0.9723],
    [1, 0.9740],
    [2, 0.9748],
    [3, 0.9747],
    [4, 0.9728],
    [5, 0.9739],
    [6, 0.9735],
    [7, 0.9729],
    [8, 0.9732],
    [9, 0.9739],
    [10, 0.9740],
    [11, 0.9740],
])
x, y1 = data.T

data = np.array([
    [0, 0.8059],
    [1, 0.8081],
    [2, 0.8088],
    [3, 0.8085],
    [4, 0.8080],
    [5, 0.8084],
    [6, 0.8083],
    [7, 0.8069],
    [8, 0.8073],
    [9, 0.8079],
    [10, 0.8078],
    [11, 0.8074],
])
x, y2 = data.T

data = np.array([
    [0, 0.6234],
    [1, 0.6258],
    [2, 0.6268],
    [3, 0.6258],
    [4, 0.6257],
    [5, 0.6261],
    [6, 0.6260],
    [7, 0.6241],
    [8, 0.6246],
    [9, 0.6249],
    [10, 0.6250],
    [11, 0.6242],
])
x, y3 = data.T

# plot average line
# y_mean = [np.mean(y) for i in y]
fig, axs = plt.subplots(3)
# plt.plot(x, y_mean, label='Mean of the 10 BLEU scores', linestyle='--')
fig.suptitle('Impact of gamma value on distinct-n score')
# plots the coordinates
# plt.plot([1,2,3,4,5,6,7,8,9,10],[12.1,14.7,16.0, 16.7,16.9,17.4,17.4,17.7,18.0,18.1], marker='o', color='red')
axs[0].plot(x, y1, marker='o', color='tomato', label='distinct-1')
axs[1].plot(x, y2, marker='o', color='mediumseagreen', label='distinct-2')
axs[2].plot(x, y3, marker='o', color='mediumslateblue', label='distinct-3')
#give a title and a label to x and y axis
# fig.subtitle('Impact of gamma value on distinct-n score')
plt.xlabel('GAMMA')
plt.ylabel('DISTINCT-N')
fig.legend()

# save the graph as png
# plt.savefig('bleu_beam.png')
plt.savefig('distinct-n.png', bbox_inches='tight') #remove whitespace around the image