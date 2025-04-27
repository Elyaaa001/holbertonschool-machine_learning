#!/usr/bin/env python3
'''THis function makes a scatter plot'''

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    '''shows a scattered plot with the y and x-axis labelled'''

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))
    pl
    plt.scatter(x, y, color='magenta')
    plt.show()