#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:05:20 2022

@author: valeriano
"""

from mtist import infer_mtist2 as im

inf_funcs = [im.infer_from_did, im.infer_from_did_ols_with_p, 
             im.infer_from_did_lasso, im.infer_from_did_ridge, 
             im.infer_from_did_elasticnet, im.infer_from_did_lasso_cv, 
             im.infer_from_did_ridge_cv, im.infer_from_did_elasticnet_cv]

for func in inf_funcs:
    im.INFERENCE_DEFAULTS.INFERENCE_FUNCTION = func
    print(func.__name__)
    im.infer_and_score_all()