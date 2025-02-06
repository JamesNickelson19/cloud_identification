"""
@author: James
Program to train the models. Note that only the most recent model trained will be shown here. 
"""

#############################################################################################################

import fits_processing as fp
import time

#############################################################################################################

start = time.time()

lgbm_model = fp.LGBM()

lgbm_model.retrieve_training_data("../processed_data/training_features.csv")
load_data = time.time() - start

lgbm_model.train_model()
lgbm_model.write_model(filename="../models/lgb_80_images.pickle")
train_lgbm = time.time() - start

print(f"It took: {load_data} seconds to load the training data, and {train_lgbm-load_data} seconds to train the lgbm model")

#############################################################################################################

start = time.time()

xgb_model = fp.XGB()

xgb_model.retrieve_training_data("../processed_data/training_features.csv")
load_data = time.time() - start

xgb_model.train_model()
xgb_model.write_model(filename="../models/xgb_80_images.pickle")
train_xgb = time.time() - start

print(f"It took: {load_data} seconds to load the training data, and {train_xgb-load_data} seconds to train the xgb model")
