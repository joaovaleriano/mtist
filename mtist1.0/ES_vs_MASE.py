#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:47:47 2022

@author: valeriano
"""


#%%

import numpy as np
import matplotlib.pyplot as plt
from mtist import infer_mtist2 as im


#%%

np.random.seed(0)

truth = -1*np.eye(10)
inferred = np.random.normal(size=(10,10))
inferred[np.arange(10), np.arange(10)] = -1

plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.imshow(np.sign(truth), vmin=-1, vmax=1)
plt.subplot(1, 2, 2)
plt.imshow(np.sign(inferred), vmin=-1, vmax=1)

print(f"ES score = {im.calculate_es_score(truth, inferred)}")
print(f"MASE = {(np.sign(truth)==np.sign(inferred)).mean()}")