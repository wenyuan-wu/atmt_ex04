#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

data = np.array([
    [0.0, 22.4],
    [0.1, 22.9],
    [0.2, 22.6],
    [0.3, 22.4],
    [0.4, 22.0],
    [0.5, 21.5],
    [0.6, 21.5],
    [0.7, 21.3],
    [0.8, 21.1],
    [0.9, 20.9],
    [1.0, 20.4]
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
plt.title('Impact of alpha value on BLEU score')
plt.xlabel('ALPHA')
plt.ylabel('BLEU')
# plt.legend()

# save the graph as png
# plt.savefig('bleu_beam.png')
plt.savefig('bleu_alpha.png', bbox_inches='tight') #remove whitespace around the image