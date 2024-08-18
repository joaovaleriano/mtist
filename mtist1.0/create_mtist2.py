import os
import glob

from mtist import mtist_utils as mu
from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am
from mtist import infer_mtist as im

import numpy as np
import matplotlib.pyplot as plt

###############################
# GENERATE MTIST              #
###############################

mu.GLOBALS.MASTER_DATASET_DIR = "master_datasets3"
mu.GLOBALS.MTIST_DATASET_DIR = "mtist_datasets3"
# mu.GLOBALS.GT_NAMES = [f"3_sp_gt_{i}" for i in range(1, 9)]

mdg.generate_mtist_master_datasets(save_example_figures=False)

plt.close("all")

am.assemble_mtist()
