"""
@author: James
Program to train the models. Note that only the most recent model trained will be shown here. 
"""

#############################################################################################################

import fits_processing as fp
import time

#############################################################################################################

# getting all the training data
training_docs = ["../processed_data/training_features_20.csv", "../processed_data/training_features_40.csv", 
                 "../processed_data/training_features_80.csv", "../processed_data/training_features_160.csv",
                 "../processed_data/training_features_320.csv", "../processed_data/training_features_450.csv"]

# just a list of the number of images in each .csv file
model_num = ["20", "40", "80", "160", "320", "450"]

# describing what type of test is being used
test_type = "learning_rate_0.02"

# setting up model
lgbm_model = fp.LGBM()

i = 0

# iterating through training docs, training, and then saving the model with the correct name
for trainer in training_docs:
    lgbm_model.retrieve_training_data(trainer)
    lgbm_model.train_model()

    filename = f"../models/lgb_{model_num[i]}_images_{test_type}.pickle"

    i+=1

    lgbm_model.write_model(filename=filename)

#############################################################################################################

# same as with lgbm model

i = 0

for trainer in training_docs:
    xgb_model = fp.XGB()

    xgb_model.retrieve_training_data(trainer)
    xgb_model.train_model()

    filename = f"../models/xgb_{model_num[i]}_images_{test_type}.pickle"

    i+=1 

    xgb_model.write_model(filename=filename)
