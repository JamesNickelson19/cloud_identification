"""
@author: James
Program to get training data from all selected images, and save it to a csv file
"""
#############################################################################################################

import fits_processing as fp
import os

#############################################################################################################

def list_files(dir):
    """List all FIT files in directory and sub-directories"""
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name[-4:] == ".FIT" and name[:3] == "All": # making sure all files are FIT and AllSkyImages
                r.append(os.path.join(root, name))
    return r

#############################################################################################################
# creating the csv file, and giving it the relevent headers

print("Starting")

outfile = open("../processed_data/training_features.csv", "w")
outfile.write("subregion,moonalt,sunalt,moonphase,exp_time,srcdens,bkgmean,bkgmedian,bkgstd,cloudy\n")

#############################################################################################################

# getting the list of files
files = list_files('../FIT_images')

for file in files:

    # setting up the image for feature extraction
    image = fp.processedFIT(file)
    image.read_single_fit()
    image.crop_and_mask()

    # generating subregions
    subregions = image.generate_subregions()
    #image.save_subregions() # this part only needs to be run when first generating subregions, which I've already done
        
    # extracting features
    features = image.exctract_features(subregions)

    # writing to the file, making sure to place the bkg and srcdens features in the appropriate subregion
    for sub in features['Subregion']:

        # setting it so I can index through the features with an array of data properly
        index = sub-1
        
        # actually writing to the file
        outfile.write(f"{sub},{float(features['moon_alt'])},{float(features['sun_alt'])},{float(features['moon_phase'])},{float(features['exp_time'])},\
                      {float(features['srcdens'][index])},{float(features['bkgmean'][index])},{float(features['bkgmedian'][index])},{float(features['bkgstd'][index])},\n")

    # getting the filename for the png version of the image
    split_file = file.split(".")
    split_file[-1] = "png"
    file = ".".join(split_file)

    # sending the png version to the correct folder
    split_file = file.split("\\")
    split_file[0] = "../png_images"
    filename = "/".join(split_file)

    # creating the subregion overlay, and saving the overlayed image as a png
    overlay = image.create_overlay(overlaytype="subregions",  regions=[1,2,3,4,5,6,7,8,9])
    image.write_image(filename=filename, overlay=overlay)

#############################################################################################################

outfile.close()
print("Done")
