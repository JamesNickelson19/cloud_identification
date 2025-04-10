"""
@author: James
Program used to test the models
"""

#############################################################################################################

import fits_processing as fits
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

#############################################################################################################

def test(filepath, algo):
    """
    This function will test the loaded model. It takes the filepath to the model, and which
    algorithm is behind it. 
    """

    start = time.time()

    # getting right algorithm, and reading the trained model
    if algo == "lgb":
        model = fits.LGBM()
        model.read_model(filepath)

    if algo == "xgb":
        model = fits.XGB()
        model.read_model(filepath)
    
    # loading test data
    #model.load_data("../processed_data/test_data_17_subs.csv") # for 17 subregions
    model.load_data("../processed_data/test_data.csv") # for 9 subregions

    correct_predictions = 0

    # iterating through test data and making a prediction, before comparing this to the real value
    for n in range(450):
        
        prediction = model.predict(model.data_X.iloc[n].values.reshape(1, -1))[0]

        if prediction == model.data_y[n]:
            correct_predictions += 1

    print(f"Model being tested is: {filepath.split('/')[-1]}")

    time_taken = time.time() - start

    print(f"Time taken to test was {time_taken} seconds.")

    # finding the percentage accuracy
    percentage = (correct_predictions/450) * 100
        
    return percentage

#############################################################################################################

print("Starting")


# getting models to be tested

# for testing the models post-code breaking
#lgbm_paths = ["../models/lgb_20_images_test.pickle", "../models/lgb_40_images_test.pickle",
#              "../models/lgb_80_images_test.pickle", "../models/lgb_160_images_test.pickle",
#              "../models/lgb_320_images_test.pickle", "../models/lgb_450_images_test.pickle"]

#xgb_paths = ["../models/xgb_20_images_test.pickle","../models/xgb_40_images_test.pickle",
#             "../models/xgb_80_images_test.pickle", "../models/xgb_160_images_test.pickle",
#             "../models/xgb_320_images_test.pickle", "../models/xgb_450_images_test.pickle"]

# for testing the models trained on 17 subregions
#lgbm_paths = ["../models/lgb_20_images_test_more_subs.pickle","../models/lgb_40_images_test_more_subs.pickle",
#              "../models/lgb_80_images_test_more_subs.pickle"]

#xgb_paths = ["../models/xgb_20_images_test_more_subs.pickle","../models/xgb_40_images_test_more_subs.pickle",
#             "../models/xgb_80_images_test_more_subs.pickle"]

# testing only photometric and overcast images
#lgbm_paths = ["../models/lgb_20_images_test_no_greys.pickle","../models/lgb_40_images_test_no_greys.pickle",
#             "../models/lgb_80_images_test_no_greys.pickle"]

#xgb_paths = ["../models/xgb_20_images_test_no_greys.pickle","../models/xgb_40_images_test_no_greys.pickle",
#             "../models/xgb_80_images_test_no_greys.pickle"]

# testing augmented images version
#lgbm_paths = ["../models/lgb_20_images_test_augmented.pickle","../models/lgb_40_images_test_augmented.pickle",
#              "../models/lgb_80_images_test_augmented.pickle"]

#xgb_paths = ["../models/xgb_20_images_test_augmented.pickle","../models/xgb_40_images_test_augmented.pickle",
#             "../models/xgb_80_images_test_augmented.pickle"]

# testing n_estimators = 1200
#lgbm_paths = ["../models/lgb_20_images_test_1200_n_estimators.pickle", "../models/lgb_40_images_test_1200_n_estimators.pickle",
#              "../models/lgb_80_images_test_1200_n_estimators.pickle","../models/lgb_160_images_test_1200_n_estimators.pickle",
#              "../models/lgb_320_images_test_1200_n_estimators.pickle","../models/lgb_450_images_test_1200_n_estimators.pickle"]

#xgb_paths = ["../models/xgb_20_images_test_1200_n_estimators.pickle","../models/xgb_40_images_test_1200_n_estimators.pickle",
#             "../models/xgb_80_images_test_1200_n_estimators.pickle","../models/xgb_160_images_test_1200_n_estimators.pickle",
#             "../models/xgb_320_images_test_1200_n_estimators.pickle","../models/xgb_450_images_test_1200_n_estimators.pickle"]

# testing n_estimators = 100
#lgbm_paths = ["../models/lgb_20_images_test_100_n_estimators.pickle", "../models/lgb_40_images_test_100_n_estimators.pickle",
#              "../models/lgb_80_images_test_100_n_estimators.pickle","../models/lgb_160_images_test_100_n_estimators.pickle",
#              "../models/lgb_320_images_test_100_n_estimators.pickle","../models/lgb_450_images_test_100_n_estimators.pickle"]

#xgb_paths = ["../models/xgb_20_images_test_100_n_estimators.pickle","../models/xgb_40_images_test_100_n_estimators.pickle",
#             "../models/xgb_80_images_test_100_n_estimators.pickle","../models/xgb_160_images_test_100_n_estimators.pickle",
#             "../models/xgb_320_images_test_100_n_estimators.pickle","../models/xgb_450_images_test_100_n_estimators.pickle"]

# testing max_leaves = 30
#lgbm_paths = ["../models/lgb_20_images_test_60_leaves.pickle","../models/lgb_40_images_test_60_leaves.pickle",
#              "../models/lgb_80_images_test_60_leaves.pickle","../models/lgb_160_images_test_60_leaves.pickle",
#              "../models/lgb_320_images_test_60_leaves.pickle","../models/lgb_450_images_test_60_leaves.pickle"]

#xgb_paths = ["../models/xgb_20_images_test_60_leaves.pickle","../models/xgb_40_images_test_60_leaves.pickle",
#             "../models/xgb_80_images_test_60_leaves.pickle","../models/xgb_160_images_test_60_leaves.pickle",
#             "../models/xgb_320_images_test_60_leaves.pickle","../models/xgb_450_images_test_60_leaves.pickle"]


# testing max_leaves = 15
#lgbm_paths = ["../models/lgb_20_images_test_15_leaves.pickle","../models/lgb_40_images_test_15_leaves.pickle",
#              "../models/lgb_80_images_test_15_leaves.pickle","../models/lgb_160_images_test_15_leaves.pickle",
#              "../models/lgb_320_images_test_15_leaves.pickle","../models/lgb_450_images_test_15_leaves.pickle"]

#xgb_paths = ["../models/xgb_20_images_test_15_leaves.pickle","../models/xgb_40_images_test_15_leaves.pickle",
#             "../models/xgb_80_images_test_15_leaves.pickle","../models/xgb_160_images_test_15_leaves.pickle",
#             "../models/xgb_320_images_test_15_leaves.pickle","../models/xgb_450_images_test_15_leaves.pickle"]

# testing learning_rate = 0.16
#lgbm_paths = ["../models/lgb_20_images_learning_rate_0.16.pickle", "../models/lgb_40_images_learning_rate_0.16.pickle", 
#              "../models/lgb_80_images_learning_rate_0.16.pickle", "../models/lgb_160_images_learning_rate_0.16.pickle",
#              "../models/lgb_320_images_learning_rate_0.16.pickle", "../models/lgb_450_images_learning_rate_0.16.pickle"]

#xgb_paths = ["../models/xgb_20_images_learning_rate_0.16.pickle", "../models/xgb_40_images_learning_rate_0.16.pickle",
#             "../models/xgb_80_images_learning_rate_0.16.pickle", "../models/xgb_160_images_learning_rate_0.16.pickle",
#             "../models/xgb_320_images_learning_rate_0.16.pickle", "../models/xgb_450_images_learning_rate_0.16.pickle"]

# testing learning_rate = 0.02
#lgbm_paths = ["../models/lgb_20_images_learning_rate_0.02.pickle", "../models/lgb_40_images_learning_rate_0.02.pickle", 
#              "../models/lgb_80_images_learning_rate_0.02.pickle", "../models/lgb_160_images_learning_rate_0.02.pickle",
#              "../models/lgb_320_images_learning_rate_0.02.pickle", "../models/lgb_450_images_learning_rate_0.02.pickle"]

#xgb_paths = ["../models/xgb_20_images_learning_rate_0.02.pickle", "../models/xgb_40_images_learning_rate_0.02.pickle",
#             "../models/xgb_80_images_learning_rate_0.02.pickle", "../models/xgb_160_images_learning_rate_0.02.pickle",
#             "../models/xgb_320_images_learning_rate_0.02.pickle", "../models/xgb_450_images_learning_rate_0.02.pickle"]

# testing learning_rate = 5
#lgbm_paths = ["../models/lgb_20_images_learning_rate_5.pickle", "../models/lgb_40_images_learning_rate_5.pickle", 
#              "../models/lgb_80_images_learning_rate_5.pickle", "../models/lgb_160_images_learning_rate_5.pickle",
#              "../models/lgb_320_images_learning_rate_5.pickle", "../models/lgb_450_images_learning_rate_5.pickle"]

#xgb_paths = ["../models/xgb_20_images_learning_rate_5.pickle", "../models/xgb_40_images_learning_rate_5.pickle",
#             "../models/xgb_80_images_learning_rate_5.pickle", "../models/xgb_160_images_learning_rate_5.pickle",
#             "../models/xgb_320_images_learning_rate_5.pickle", "../models/xgb_450_images_learning_rate_5.pickle"]

# testing learning_rate = 10
#lgbm_paths = ["../models/lgb_20_images_learning_rate_10.pickle", "../models/lgb_40_images_learning_rate_10.pickle", 
#              "../models/lgb_80_images_learning_rate_10.pickle", "../models/lgb_160_images_learning_rate_10.pickle",
#              "../models/lgb_320_images_learning_rate_10.pickle", "../models/lgb_450_images_learning_rate_10.pickle"]

#xgb_paths = ["../models/xgb_20_images_learning_rate_10.pickle", "../models/xgb_40_images_learning_rate_10.pickle",
#             "../models/xgb_80_images_learning_rate_10.pickle", "../models/xgb_160_images_learning_rate_10.pickle",
#             "../models/xgb_320_images_learning_rate_10.pickle", "../models/xgb_450_images_learning_rate_10.pickle"]

# testing alpha = 1000
#lgbm_paths = ["../models/lgb_20_images_alpha_1000.pickle", "../models/lgb_40_images_alpha_1000.pickle",
#              "../models/lgb_80_images_alpha_1000.pickle", "../models/lgb_160_images_alpha_1000.pickle",
#              "../models/lgb_320_images_alpha_1000.pickle", "../models/lgb_450_images_alpha_1000.pickle"]

#xgb_paths = ["../models/xgb_20_images_alpha_1000.pickle", "../models/xgb_40_images_alpha_1000.pickle",
#             "../models/xgb_80_images_alpha_1000.pickle", "../models/xgb_160_images_alpha_1000.pickle",
#             "../models/xgb_320_images_alpha_1000.pickle", "../models/xgb_450_images_alpha_1000.pickle"]

# testing alpha = 0.1
#lgbm_paths = ["../models/lgb_20_images_alpha_0.1.pickle", "../models/lgb_40_images_alpha_0.1.pickle",
#              "../models/lgb_80_images_alpha_0.1.pickle", "../models/lgb_160_images_alpha_0.1.pickle",
#              "../models/lgb_320_images_alpha_0.1.pickle", "../models/lgb_450_images_alpha_0.1.pickle"]

#xgb_paths = ["../models/xgb_20_images_alpha_0.1.pickle", "../models/xgb_40_images_alpha_0.1.pickle",
#             "../models/xgb_80_images_alpha_0.1.pickle", "../models/xgb_160_images_alpha_0.1.pickle",
#             "../models/xgb_320_images_alpha_0.1.pickle", "../models/xgb_450_images_alpha_0.1.pickle"]

# testing lambda = 10000
#lgbm_paths = ["../models/lgb_20_images_lambda_10000.pickle", "../models/lgb_40_images_lambda_10000.pickle",
#              "../models/lgb_80_images_lambda_10000.pickle", "../models/lgb_160_images_lambda_10000.pickle",
#              "../models/lgb_320_images_lambda_10000.pickle", "../models/lgb_450_images_lambda_10000.pickle"]

#xgb_paths = ["../models/xgb_20_images_lambda_10000.pickle", "../models/xgb_40_images_lambda_10000.pickle",
#             "../models/xgb_80_images_lambda_10000.pickle", "../models/xgb_160_images_lambda_10000.pickle",
#             "../models/xgb_320_images_lambda_10000.pickle", "../models/xgb_450_images_lambda_10000.pickle"]

# testing lambda = 1
#lgbm_paths = ["../models/lgb_20_images_lambda_1.pickle", "../models/lgb_40_images_lambda_1.pickle",
#              "../models/lgb_80_images_lambda_1.pickle", "../models/lgb_160_images_lambda_1.pickle",
#              "../models/lgb_320_images_lambda_1.pickle", "../models/lgb_450_images_lambda_1.pickle"]

#xgb_paths = ["../models/xgb_20_images_lambda_1.pickle", "../models/xgb_40_images_lambda_1.pickle",
#             "../models/xgb_80_images_lambda_1.pickle", "../models/xgb_160_images_lambda_1.pickle",
#             "../models/xgb_320_images_lambda_1.pickle", "../models/xgb_450_images_lambda_1.pickle"]

#############################################################################################################

y_vals_lgb = []
y_vals_xgb = []

# iterating through filepaths for both models, testing, then printing the results
for path in lgbm_paths:

    results = test(path, "lgb")

    print(f"The percentage accuracy was {results}%\n")

    # appending the results to an empty set for creating a graph
    y_vals_lgb.append(results)

for path in xgb_paths:

    results = test(path, "xgb")

    print(f"The percentage accuracy was {results}%\n")

    y_vals_xgb.append(results)

#############################################################################################################
# making the graph

# x values to be set to the graph
x_vals = [20,40,80,160,320,450] # [20, 40, 80] when testing methods of improving accuracy that involved changing training dataset

# this section was only necessary before the models had been fully trained, so that the full graph could be seen
#while len(y_vals_xgb) != len(x_vals):
#    y_vals_lgb.append(None)
#    y_vals_xgb.append(None)


default_x_ticks = range(len(x_vals))

plt.figure()

# plotting
plt.plot(default_x_ticks, y_vals_lgb, color="b", label="LGB model", marker="o", linewidth=3)
plt.plot(default_x_ticks, y_vals_xgb, color="r", label="XGB model", marker="o")

# getting axis ranges set up, and putting the right values along the x_axis
plt.xticks(default_x_ticks, x_vals)
plt.ylim(0, 100)
plt.xlim(-0.25, 5.25)

# setting labels
plt.xlabel("Number of Images")
plt.ylabel("Percentage accuracy (%)")

plt.grid()
plt.legend()

# setting the title and saving the figure. these are from the last tested models
plt.title("Percentage Accuracy with lambda = 1")
plt.savefig("../processed_data/results_graph_lambda_1.png")

print("Done")