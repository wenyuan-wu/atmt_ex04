#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

data = np.array([
    [1, 21.1],
    [2, 22.0],
    [3, 22.4],
    [4, 22.2],
    [5, 22.2],
    [6, 22.1],
    [7, 22.0],
    [8, 21.7],
    [9, 21.6],
    [10, 21.6],
    [11, 21.7],
    [12, 21.7],
    [13, 21.6],
    [14, 21.6],
    [15, 21.6]
])
x, y = data.T

# plot average line
# y_mean = [np.mean(y) for i in y]
fig, ax = plt.subplots()
# plt.plot(x, y_mean, label='Mean of the 10 BLEU scores', linestyle='--')

# plots the coordinates
# plt.plot([1,2,3,4,5,6,7,8,9,10],[12.1,14.7,16.0, 16.7,16.9,17.4,17.4,17.7,18.0,18.1], marker='o', color='red')
plt.plot(x, y, marker='o', color='red')

#give a title and a label to x and y axis
plt.title('Impact of beam size on BLEU score')
plt.xlabel('BEAM SIZE')
plt.ylabel('BLEU')
# plt.legend()

# save the graph as png
# plt.savefig('bleu_beam.png')
plt.savefig('bleu_beam.png', bbox_inches='tight') #remove whitespace around the image