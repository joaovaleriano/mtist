import os
import glob

from mtist import mtist_utils as mu
from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am
from mtist import infer_mtist as im

import numpy as np
from matplotlib import pyplot as plt

###############################
# GENERATE MTIST              #
###############################

mu.GLOBALS.MASTER_DATASET_DIR = "master_datasets_food2"
mu.GLOBALS.MTIST_DATASET_DIR = "mtist_datasets_food2"
mu.GLOBALS.GT_DIR = "ground_truths_food"
mu.GLOBALS.GT_NAMES = [f"3_sp_gt_{i}" for i in range(1, 9)]

# am.ASSEMBLE_MTIST_DEFAULTS.N_TIMESERIES_PARAMS = np.arange(1, 30) 
# am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_FREQ_PARAMS = [15]
# am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_SCHEME_PARAMS = ["even"] 

# mdg.MASTER_DATASET_DEFAULTS.NOISE_SCALES = [0.01]

mdg.generate_mtist_master_datasets()

plt.close("all")

am.assemble_mtist()
