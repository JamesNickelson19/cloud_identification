"""
@author: James
Program used to test the models
"""

#############################################################################################################

import fits_processing as fp
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

    if algo == "lgb":
        model = fp.LGBM()
        model.read_model(filepath)

    if algo == "xgb":
        model = fp.XGB()
        model.read_model(filepath)
    
    model.load_data("../processed_data/test_data.csv")

    correct_predictions = 0

    for n in range(450):
        
        prediction = model.predict(model.data_X.iloc[n].values.reshape(1, -1))[0]

        if prediction == model.data_y[n]:
            correct_predictions += 1

    time_taken = time.time() - start

    print(f"Time taken to test {filepath.split('/')[-1]} was {time_taken} seconds.")

    percentage = (correct_predictions/450) * 100
        
    return percentage

#############################################################################################################

print("Starting")

lgbm_paths = ["../models/lgb_20_images.pickle", "../models/lgb_40_images.pickle",]
             #"../models/lgb_80_images.pickle", "../models/lgb_160_images.pickle",
             #"../models/lgb_320_images.pickle", "../models/lgb_450_images.pickle"]

xgb_paths = ["../models/xgb_20_images.pickle", "../models/xgb_40_images.pickle",]
            #"../models/xgb_80_images.pickle",] #"../models/xgb_160_images.pickle",
            #"../models/xgb_320_images.pickle", "../models/xgb_450_images.pickle"]

y_vals_lgb = []
y_vals_xgb = []

for path in lgbm_paths:

    results = test(path, "lgb")

    print(f"The percentage accuracy of the aformentioned model was {results}%\n")

    y_vals_lgb.append(results)

for path in xgb_paths:

    results = test(path, "xgb")

    print(str(results) + " XGB")

    y_vals_xgb.append(results)

#############################################################################################################

x_vals = [20,40,80,160,320,450]

while len(y_vals_xgb) != len(x_vals):
    y_vals_lgb.append(None)
    y_vals_xgb.append(None)


default_x_ticks = range(len(x_vals))

plt.figure()

plt.plot(default_x_ticks, y_vals_lgb, color="b", label="LGB model", marker="o")
plt.plot(default_x_ticks, y_vals_xgb, color="k", label="XGB model", marker="o")

plt.xticks(default_x_ticks, x_vals)
plt.ylim(0, 100)

plt.xlabel("Number of Images")
plt.ylabel("Percentage accuracy (%)")

plt.grid()
plt.legend()

plt.title("Percentage Accuracy of ML Models During Training")

plt.savefig("../processed_data/results_graph.png")

print("Done")