import os
import glob

from mtist import mtist_utils as mu
from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am
from mtist import infer_mtist as im

import numpy as np
from matplotlib import pyplot as plt

################################
# INFER MTIST 4 WAYS #
################################

mu.GLOBALS.MASTER_DATASET_DIR = "master_datasets_nofood"
mu.GLOBALS.MTIST_DATASET_DIR = "mtist_datasets_nofood"
mu.GLOBALS.GT_DIR = "ground_truths_nofood"
mu.GLOBALS.GT_NAMES = [f"3_sp_gt_{i}" for i in range(1, 9)]

am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_FREQ_PARAMS = [5, 10, 15]
am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_SCHEME_PARAMS = ["seq"]

# mdg.MASTER_DATASET_DEFAULTS.NOISE_SCALES = [0.01]

inference_names = [
    "default",
    "ridge_CV",
#    "lasso_CV",
#    "elasticnet_CV",
]

prefixes = [f"{name}_" for name in inference_names]

inference_fxn_handles = [
    im.infer_from_did,
    im.infer_from_did_ridge_cv,
#    im.infer_from_did_lasso_cv,
#    im.infer_from_did_elasticnet_cv,
]

for inference_type, prefix, handle in zip(
    inference_names, prefixes, inference_fxn_handles
):
    print(inference_type+"_ignore_food")
    im.INFERENCE_DEFAULTS.INFERENCE_PREFIX = prefix+"ignore_food_"
    im.INFERENCE_DEFAULTS.INFERENCE_FUNCTION = handle
    _ = im.infer_and_score_all(save_inference=True, save_scores=True, food_source=False)

    print(inference_type+"_ignore_food_arithm")
    im.INFERENCE_DEFAULTS.INFERENCE_PREFIX = prefix+"ignore_food_arithm_"
    im.INFERENCE_DEFAULTS.INFERENCE_FUNCTION = handle
    _ = im.infer_and_score_all(save_inference=True, save_scores=True, food_source=False, arithm_mean=True)

    print(inference_type+"_food")
    im.INFERENCE_DEFAULTS.INFERENCE_PREFIX = prefix+"food_"
    im.INFERENCE_DEFAULTS.INFERENCE_FUNCTION = handle
    _ = im.infer_and_score_all(save_inference=True, save_scores=True, food_source=True)

    print(inference_type+"_food_arithm")
    im.INFERENCE_DEFAULTS.INFERENCE_PREFIX = prefix+"food_arithm_"
    im.INFERENCE_DEFAULTS.INFERENCE_FUNCTION = handle
    _ = im.infer_and_score_all(save_inference=True, save_scores=True, food_source=True, arithm_mean=True)
