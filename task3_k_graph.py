#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

data = np.array([
    [1, 21.1],
    [2, 22.4],
    [3, 22.9],
    [4, 22.8],
    [5, 23.2],
    [6, 23.3],
    [7, 23.3],
    [8, 23.1],
    [9, 23.3],
    [10, 23.3]
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
plt.title('Impact of beam size on BLEU score with alpha set to 0.1')
plt.xlabel('BEAM SIZE')
plt.ylabel('BLEU')
# plt.legend()

# save the graph as png
# plt.savefig('bleu_beam.png')
plt.savefig('bleu_alpha_beam.png', bbox_inches='tight') #remove whitespace around the image