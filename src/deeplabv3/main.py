# Main model training

from utils import calc_norm_parameters
from glob import glob
import os
import json
from config import SCALE_PARAM_PATH
import lightning.pytorch as pl


# check for precalculated normalization parameters

if os.path.exists(SCALE_PARAM_PATH):
    try:
        with open(SCALE_PARAM_PATH, 'r') as f:
            scale_params = json.load(f)
        for key in ['means', 'std']:
            assert(key in scale_params.keys())
    except AssertionError:
        print('Scale params file missing key elements. Recalculating imageset scale parameters - this may take a few minutes.')    
        params = calc_norm_parameters('./data/images')
        scale_params = dict(params)
        scale_params.pop('means_array')
        scale_params['means'] = list(scale_params['means'])
        scale_params['std'] = list(scale_params['std'])

        with open('./outputs/scale_params/scale_params.json', 'w') as f:
            json.dump(scale_params, f)
elif not os.path.exists(SCALE_PARAM_PATH):
    print('Scale params file missing. Calculating imageset scale parameters - this may take a few minutes.')    
    params = calc_norm_parameters('./data/images')
    scale_params = dict(params)
    scale_params.pop('means_array')
    scale_params['means'] = list(scale_params['means'])
    scale_params['std'] = list(scale_params['std'])

    with open('./outputs/scale_params/scale_params.json', 'w') as f:
        json.dump(scale_params, f)
print("Imageset scale parameters loaded.")



