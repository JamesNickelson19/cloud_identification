"""
@author: James
Program to process fits files for the next steps. It should separate the important
information out from the header (exposure time, date and time of image taking)
and process the data in such a way as to be usable for either algorithm
"""

import os
import datetime
from joblib import dump, load
from collections import OrderedDict

import numpy as np
from scipy.signal import convolve2d
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import uniform, randint

import sep
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from skimage import measure
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (train_test_split, cross_validate, RandomizedSearchCV)
from sklearn.metrics import f1_score, confusion_matrix

import astropy.units as u
from astropy.coordinates import EarthLocation
from pytz import timezone
from astroplan import Observer

import remove_outliers as rmo

#############################################################################################################

# Please note: any sections marked as "taken from cloudynight" came from the github page linked below. 
# https://github.com/mommermi/cloudynight/blob/master/cloudynight/cloudynight.py

#############################################################################################################
# setting up observatory location for finding sun and moon altitudes at time of image
# set up using information on https://astroplan.readthedocs.io/en/stable/getting_started.html

# setting up coordinates and elevation
longitude = "-0d5m40s"
latitude = "+51d46m30s"
elevation = 66 * u.m

# establishing location
location = EarthLocation.from_geodetic(longitude, latitude, elevation)

# defining the actual observatory location
observatory = Observer(name = "Bayfordbury", location = location, timezone = timezone("UTC"), description = "Bayfordbury observatory, university of hertfordshire")







#############################################################################################################







class processedFIT():
    """
    Class for processing FIT files into their constituent parts
    """

    def __init__(self, filename):
        """
        __init__ taken from cloudynight, Mommert, M. 2020
        """
        self.filename = filename  # raw image filename
        self.datetime = None      # date and time of observation
        self.thumbfilename = None # filename for thumbnail image
        self.data = None          # image data array
        self.header = None        # image header
        self.subregions = None    # subregion arrays
        self.features = None      # extracted features
        self.imgdata = None       # data for all images in a directory
        
#############################################################################################################

    def read_single_fit(self):
        """
        Read FIT file, and separate the header and data, and save to self
        """

        hdu = fits.open(self.filename) # opening FIT

        # saving data
        self.header = hdu[0].header
        self.data = hdu[0].data

        # testing to ensure data has been gathered correctly
        #print(self.header)
        #print(self.data)

        hdu.close() # closing file

        return self

#############################################################################################################

    def save_image(self, filename):
        """
        Test to see if cropped range is correct by saving cropped image
        """

        hdu = fits.PrimaryHDU(data=self.data, header=self.header) # getting data and the header into a FIT format

        hdu.writeto(filename, overwrite=True) # saving to a file, overwrite is set to true for testing

#############################################################################################################

    def crop(self):
        """
        Cropping the image. This function only exists for generating the mask
        """
        # setting crop ranges
        y_croprange = (0,480)
        x_croprange = (80, 560)

        # cropping image
        self.data = self.data[y_croprange[0]:y_croprange[1], x_croprange[0]:x_croprange[1]]

#############################################################################################################

    def generate_mask(self, mask_lwr, convolve=None ,gaussian_blur=None, median = False, filename="../mask/mask.FIT"):
        """
        Generate an image mask to cover the background

        PARAMETERS
        -----------
        mask_lwr : int
                 masks all pixel values lower than this
        convolve : int, None by default
                 convolves image if value is given, can be ignored if not. value doesn't have to be given
        gaussian_blur : int, None by default
                      applies a gaussian blur to the image using the provided value. ideally something is given else code won't run properly
        median : Bool, false by default
               used to help figure out what mask_lwr should be. means only the median image is returned
        filename : str, "mask.FIT" by default
                 filename to save the mask as. for the actual mask no changes need to be made, but for the median image could be changed to "median.FIT"
        """

        if self.imgdata is not None:
            mask = np.median([img.data for img in self.imgdata], axis=0) # generating an image based off the median of multiple images (not included)

        else:
            mask = np.ones(self.data.shape) # generating blank mask if just one image is supplied

        if gaussian_blur is not None: # checking to make sure a gaussian blur value was provided
            mask = gaussian_filter(mask, gaussian_blur) # applying blur if one was. note this won't do anything if there wasnt a median image created

            if median is False: # checking to see if just the median image is wanted. this would be the case for finding a good pixel value to mask all values below

                mask_new = np.ones(self.data.shape) # creating another ones mask, just in case a median image was generated
        
                line_val = 0 # starting an iterator for indexing the mask line

                if self.imgdata is None: # checking to make sure there isnt a median image 

                    for line in self.data: # iterating through single image lines
                        value_val = 0 # starting an iterator for indexing the value in the mask line

                        for value in line: # iterating through values in the lines

                            if value <= mask_lwr: # checking to see if its lower than the threshold

                                for n in range(5): # starting an iterator to mask more than should be needed
                                    if value_val + 5 >= 480 or line_val + 5 >= 480: # checking to make sure its not going to be outside of range
                                        pass

                                    else: # masking pixels around the pixel value
                                        mask_new[line_val+n][value_val+n] = 0
                                        mask_new[line_val+n][value_val-n] = 0
                                        mask_new[line_val-n][value_val+n] = 0
                                        mask_new[line_val-n][value_val-n] = 0

                                value_val += 1 

                        line_val += 1

                else: # going through same process as before, just with the median image instead of self.data

                    for line in mask:
                        value_val = 0

                        for value in line:

                            if value <= mask_lwr:

                                for n in range(5):
                                    if value_val + 5 >= 480 or line_val + 5 >= 480:
                                        pass
                                    else:
                                        mask_new[line_val+n][value_val+n] = 0
                                        mask_new[line_val+n][value_val-n] = 0
                                        mask_new[line_val-n][value_val+n] = 0
                                        mask_new[line_val-n][value_val-n] = 0

                mask = mask_new # setting the mask equal to this new one

            if convolve is not None: # checking to see if a convolve value was given, then convolving
                mask = np.clip(convolve2d(mask, np.ones((convolve, convolve)), mode='same'), 0, 1)

        self.maskdata = mask

        hdu = fits.PrimaryHDU(data=mask, header=None) # getting data into a FIT-useable format. header is not necessary information for this

        hdu.writeto(filename, overwrite=True) # writing to file

        return mask

#############################################################################################################

    def crop_and_mask(self):
        """Crop image and mask"""

        mask = fits.open("../mask/mask.FIT")

        # setting crop ranges
        y_croprange = (0,480)
        x_croprange = (80, 560)

        # cropping image
        self.data = self.data[y_croprange[0]:y_croprange[1], x_croprange[0]:x_croprange[1]]
        
        # applying mask
        self.data = self.data * mask[0].data

        # saving to self for generating subregions
        self.maskdata = mask[0].data 

        mask.close()

#############################################################################################################

    def generate_subregions(self):
        """
        Create subregions array. This array consists of N_subregions
        arrays, each with the same dimensions as self.maskdata. Taken from cloudynight.py
        """

        # slight change from the cloudynight version of this, as I'm not using the conf document
        # all places where conf was used was replaced with these two terms
        N_RINGS = 2
        N_RINGSEGMENTS = 4

        shape = np.array(self.maskdata.shape)
        center_coo = shape/2
        radius_borders = np.linspace(0, min(shape)/2, N_RINGS + 2)
        azimuth_borders = np.linspace(-np.pi, np.pi, N_RINGSEGMENTS + 1)
        n_subregions = N_RINGS * N_RINGSEGMENTS+1 # plus 1 for the middle section 

        # build templates for radius and azimuth
        y, x = np.indices(shape)
        r_map = np.sqrt((x-center_coo[0])**2 + (y-center_coo[1])**2).astype(int)
        az_map = np.arctan2(y-center_coo[1], x-center_coo[0])

        # subregion maps
        subregions = np.zeros([n_subregions, *shape], dtype=bool)
        
        # polygons around each source region in original image dimensions
        polygons = []
        
        for i in range(N_RINGS+1):
            for j in range(N_RINGSEGMENTS):
                if i == 0 and j == 0:
                    subregions[0][(r_map < radius_borders[i+1])] = True
                    # find contours
                    contours = measure.find_contours(subregions[0], 0.5)
                elif i == 0 and j > 0:
                    break
                else:
                    subregions[(i-1)*N_RINGSEGMENTS+j+1][
                        ((r_map > radius_borders[i]) &
                         (r_map < radius_borders[i+1]) &
                         (az_map > azimuth_borders[j]) &
                         (az_map < azimuth_borders[j+1]))] = True
                    contours = measure.find_contours(subregions[(i-1) * N_RINGSEGMENTS + j+1], 0.5)
                # downscale number of vertices
                polygons.append((contours[0][:,0][::10], contours[0][:,1][::10]))
                
        self.subregions = subregions
        #self.polygons = np.array(polygons) # this part keeps throwing up an error, and its not important enough for me to fix it

        return subregions
    
#############################################################################################################

    def save_subregions(self, subregions = None):
        """
        Save subregions so that they can be veiwed as png files

        Params:
        subregions : Array of subregions, None by default
        Array of subregions, only needs to have a value if generate_subregions isn't run beforehand
        """

        # check to see if there are subregions available, if not setting the subregions equal to self.subregions
        if subregions == None:
            subregions = self.subregions

        for subi in range(len(subregions)):
            plt.imshow(subregions[subi], origin='lower', vmin=0, vmax=1)
            plt.savefig(f"../subregions/subregion_{subi}.png")

            plt.close()
            
#############################################################################################################

    def remove_outliers(self):
        """
        Check weather data (if available) and the exposure time of the image to see wether or not its worth
        running the algorithm. algorithm will only be run if both parts indicate that its worthwhile

        returns: [weather, expos], bool
                 both bool values, both must be True for algorithm to run.

                 [reason_w, reason_e], str
                 reasons for algorithm not running if either of them are the issue, 
                 saved here for later decisions about whether or not the domes should open
        """

        # getting data out of the header
        exp_time = float(self.header["EXPTIME"])
        time_unprocessed = self.header["DATE-OBS"]

        # splitting the datetime part into two separate sections for later part
        date_part, time_part = time_unprocessed.split('T')

        # processing date and time into datetime objects, as removeOutliers relies on that
        date = datetime.datetime.strptime(date_part, "%Y-%m-%d").date()
        time = datetime.datetime.strptime(time_part, "%H:%M:%S.%f").time()

        # initialising removeOutliers object
        data = rmo.removeOutliers(date, time, exp_time)

        # checking wether the weather and exposure time information indicate the algorithm should be run or not
        weather, reason_w = data.check_clouds()
        expos, reason_e = data.check_expsure_time()

        # this section was just to test to make sure it was actually working, and will be how the decision is made in practice
        #if [weather, expos] == [True, True]:
        #    print("Run algo")

        #else:
        #    if weather == False:
        #        print(reason_w)
            
        #    if expos == False:
        #        print(reason_e)

        # returning wether or not the algorithm should run. both must be true
        return [weather, expos], [reason_w, reason_e]

#############################################################################################################

    def extract_features(self, subregions, mask = None):
        """
        Extract image features for each subregion. Image should be cropped
        and masked. Taken from cloudynight

        PARAMETERS
        -----------
        subregions: subregion array, generated in generate_subregions
        subregions to be used
        mask: mask, contained in self
        mask to be applied in source extraction, optional

        return: None
        """
        # set internal pixel buffer
        sep.set_extract_pixstack(10000000)

        # extract time from header and derive frame properties

        time = Time(self.header['DATE-OBS'], format='isot')
        features = OrderedDict([('time', time.isot),
            ('exp_time', float(self.header['EXPTIME'])),
            ('filename', self.filename.split(os.path.sep)[-1]),
            ('Subregion', []),
            ('moon_alt', observatory.moon_altaz(time).alt.deg),
            ('sun_alt', observatory.sun_altaz(time).alt.deg),
            ('moon_phase', 1-observatory.moon_phase(time).value/np.pi),
        ])

        # derive and subtract sky background
        bkg = sep.Background(self.data.astype(np.float64),bw=15, bh=15,fw=3, fh=5)
        data_sub = self.data - bkg.back()

        # if mask is provided, it is applied in the proper derivation of
        # source brightness thresholds
        if mask is not None:

            mask = mask + 1
            mask[mask==2] = 0

            threshold = (np.ma.median(np.ma.array(data_sub, mask=(1-mask))) + np.median(bkg.rms())*1.5)

            src = sep.extract(data_sub, threshold, minarea=3, mask=(1-mask), deblend_nthresh=32, deblend_cont=0.005)

        else:
            threshold = (np.median(data_sub) + np.median(bkg.rms())*1.5)
            src = sep.extract(data_sub, threshold, minarea=3, mask=mask, deblend_nthresh=32, deblend_cont=0.005)

        # apply max_flag cutoff (reject flawed sources)
        src = src[src['flag'] <= 7]

        # feature extraction per subregion
        features['srcdens'] = []
        features['bkgmedian'] = []
        features['bkgmean'] = []
        features['bkgstd'] = []

        for i, sub in enumerate(subregions):

            features['Subregion'].append(i+1)

            #print(sum((sub[src['y'].astype(int), src['x'].astype(int)]).astype(int))) # gives number of sources in each subregion
            #print(sum((sub[src['y'].astype(int), src['x'].astype(int)]).astype(int)) / np.sum(sub.astype(int))) # gets number of sources divided by total sources in region. roughly matches expected output
            #print(src[sub[src['y'].astype(int), src['x'].astype(int)]] / np.sum(sub[mask== 0])) # previous way of doing it, broken due to divide by 0 errors

            features['srcdens'].append(sum((sub[src['y'].astype(int), src['x'].astype(int)]).astype(int)) / np.sum(sub.astype(int)))
            
            features['bkgmedian'].append(np.median(bkg.back()[sub]))
            features['bkgmean'].append(np.mean(bkg.back()[sub]))
            features['bkgstd'].append(np.std(bkg.back()[sub]))

        self.features = features

        return features
    
#############################################################################################################

    def create_overlay(self, overlaytype='srcdens', regions=None):
        """
        Create overlay for thumbnail image. Requires self.subregions to be
        initialized. An overlay is an array with the same dimensions as
        self.data` in which certain subregions get assigned certain values as
        defined by `overlaytype`. Taken from cloudynight.py

        :param overlaytype: define data source from `self.features` from which
                            overlay should be generated, default: 'srcdens'
        :param regions: list of length=len(self.subregions), highlights
                        subregions with list element value > 0; requires
                        `overlaytype='subregions'`, default: None

        :return: overlay array
        """

        map = np.zeros(self.data.shape)

        for i, sub in enumerate(self.subregions):
            if overlaytype == 'srcdens':
                map += sub*self.features['srcdens'][i]
            elif overlaytype == 'bkgmedian':
                map += sub*self.features['bkgmedian'][i]
            elif overlaytype == 'bkgmean':
                map += sub*self.features['bkgmean'][i]
            elif overlaytype == 'bkgstd':
                map += sub*self.features['bkgstd'][i]
            elif overlaytype == 'subregions':
                if regions[i]:
                    map += sub

                    
        map[map == 0] = np.nan
        return map

#############################################################################################################

    def write_image(self, filename, overlay=None, mask=None,
                    overlay_alpha=0.3, overlay_color='Reds'):
        """
        Write image instance as scaled png thumbnail image file.
        Taken from CLoudynight.py

        :param filename: filename of image to be written, relative to cwd
        :param overlay: provide overlay or list of overlays, optional
        :param mask: apply image mask before writing image file
        :param overlay_alpha: alpha value to be applied to overlay
        :param overlay_color: colormap to be used with overlay

        :return: None
        """
        
        data = self.data

        THUMBNAIL_SCALE = ZScaleInterval()
        THUMBNAIL_WIDTH = 6
        THUMBNAIL_HEIGHT = 6

        # derive image scaling and stretching
        if mask is not None:
            norm = ImageNormalize(data[mask.data == 1], THUMBNAIL_SCALE, stretch=LinearStretch())
            data[mask.data == 0] = 0
        else:
            norm = ImageNormalize(data=data, interval=THUMBNAIL_SCALE, stretch=LinearStretch())

        # create figure
        f, ax = plt.subplots(figsize=(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))

        # plot image
        img = ax.imshow(data, origin='lower', norm=norm, cmap='gray', extent=[0, self.data.shape[1], 0, self.data.shape[0]])

        # plot overlay(s)
        if overlay is not None:
            if not isinstance(overlay, list):
                overlay = [overlay]
                overlay_color = [overlay_color]
            overlay_img = []
            for i in range(len(overlay)):
                overlay_img.append(ax.imshow(overlay[i], cmap=overlay_color[i],
                                             origin='lower', vmin=0, alpha=overlay_alpha,
                                             extent=[0, overlay[i].shape[1], 0, overlay[i].shape[0]]))
                
                overlay_img[i].axes.get_xaxis().set_visible(False)
                overlay_img[i].axes.get_yaxis().set_visible(False)

        # remove axis labels and ticks
        plt.axis('off')
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)

        # save thumbnail image
        plt.savefig(filename, bbox_inches='tight', dpi=150, pad_inches=0)
        plt.close()







#############################################################################################################







class LGBM():
    """
    Class for handling LightGBM model
    """
    
    def __init__(self):
        """
        __init__ taken from cloudynight
        """
        self.data_X = None      # pandas DataFrame
        self.data_y = None      # pandas DataFrame
        self.model = None       # model implementation
        self.filename = None    # model pickle filename
        self.train_score = None # model training score
        self.test_score = None  # model test score
        self.val_score = None   # model validation sample score
        self.f1_score_val = None  # model validation sample f1 score

#############################################################################################################

    def retrieve_training_data(self, data_location):
        """
        Retrieves processed training data from its location

        param : data_location, str to where the training data csv file is located
        """

        n_subregions = 9 # number of subregions in the images

        raw = pd.DataFrame(pd.read_csv(data_location)) # creating a pd dataframe with the raw data

        # taken from cloudynight, processes the data. any unused information will be removed from this list and marked below
        data = pd.DataFrame()
        for j in range(len(raw['moonalt'])):
            frame = pd.DataFrame(OrderedDict(
                (('moonalt', [raw['moonalt'][j]]*n_subregions), # note to self, *n_subregions is there to make all arrays same length and prevent breaking. that it leads to 2.1 million lines of data is apparently just what has to happen
                 ('sunalt', [raw['sunalt'][j]]*n_subregions),
                 ('moonphase', [raw['moonphase'][j]]*n_subregions),
                 ('subid', range(n_subregions)),
                 ('srcdens', raw['srcdens'][j]),
                 ('bkgmean', raw['bkgmean'][j]),
                 ('bkgmedian', raw['bkgmedian'][j]),
                 ('bkgstd', raw['bkgstd'][j]),
                 ('exposure_time', raw['exp_time'][j]),
                 ('cloudy', raw['cloudy'][j]))))
            data = pd.concat([data, frame]) 

        # again taken from cloudynight
        self.data_X = data.drop(['cloudy'], axis=1)

        train_data = np.ravel(data.loc[:, ['cloudy']].values)

        self.data_y = train_data
        self.data_X_featurenames = data.drop(['cloudy'], axis=1).columns.values

#############################################################################################################

    def load_data(self, filename):
        """
        Load feature data. Taken from cloudynight
        """
        data = pd.read_csv(filename, index_col=0)

        # split features and target
        self.data_X = data.drop(['cloudy'], axis=1)
        self.data_y = np.ravel(data.loc[:, ['cloudy']].values).astype(int)
        self.data_X_featurenames = data.drop(['cloudy'], axis=1).columns.values

#############################################################################################################

    def train_model(self, parameters = {'max_depth': 5,'n_estimators': 500,
                                        'learning_rate': 0.25,'num_leaves': 30,
                                        'min_child_samples': 100,'reg_alpha': 10,
                                        'reg_lambda': 100}, cv = 5):
        """
        Train lightgbm model. Content taken from cloudynight

        PARAMETERS
        -----------
        parameters : dict
        parameters for training the model, has a default value taken from cloudynight

        cv : int 
        for cross validation, default value taken from cloudynight
        """
        # split data into training and validation sample
        X_cv, X_val, y_cv, y_val = train_test_split(self.data_X, self.data_y, test_size=0.1, random_state=42)

        # define model
        lgb = LGBMClassifier(objective='binary', random_state=42, n_jobs=-1, **parameters)

        # train model
        lgb.fit(X = X_cv, y = y_cv)
        self.model = lgb

        # derive cv scores
        cv_results = cross_validate(lgb, X_cv, y_cv, cv=cv,return_train_score=True)

        # saving to self
        self.train_score = np.max(cv_results['train_score'])
        self.test_score = np.max(cv_results['test_score'])
        self.parameters = parameters
        self.val_score = self.model.score(X_val, y_val)
        self.f1_score_val = f1_score(y_val, self.model.predict(X_val))

        return self.val_score
    
#############################################################################################################
    
    def train_randomised_search_cv(self, n_iter = 100, distributions = {
                'max_depth': randint(low=3, high=47),
                'n_estimators': randint(low=100, high=1400),
                'learning_rate': uniform(loc=0.1, scale=0.9),
                'feature_fraction': uniform(loc=0.1, scale=0.9),
                'num_leaves': randint(low=3, high=97),
                'min_child_samples': randint(low=10, high=190),
                'reg_alpha': [1, 5, 10, 50, 100],
                'reg_lambda': [1, 5, 10, 50, 100, 500, 1000]}, 
                cv = 3 , scoring = "accuracy"):
        """
        Train the model with a random cross-validation search. taken from cloudynight

        PARAMETERS
        -----------
        n_iter : int
        number of iterations, default of 100

        distributions : dict
        distributions to use, default value taken from cloudynight

        cv : int
        cross validation, default value of 3

        scoring : str
        defines scoring type for RandomizedSearchCV, default of accuracy

        """

        # split data into training and validation sample
        X_grid, X_val, y_grid, y_val = train_test_split(self.data_X, self.data_y, test_size=0.1, random_state=42)

        # initialize model
        lgb = LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)

        # initialize random search + cross-validation
        lgbrand = RandomizedSearchCV(lgb, distributions, cv=cv, scoring=scoring, n_iter=n_iter, return_train_score=True)

        # fit model
        lgbrand.fit(X_grid, y_grid)

        self.cv_results = lgbrand.cv_results_
        self.model = lgbrand.best_estimator_

        # derive scores
        self.train_score = lgbrand.cv_results_['mean_train_score'][lgbrand.best_index_]
        self.test_score = lgbrand.cv_results_['mean_test_score'][lgbrand.best_index_]
        self.parameters = lgbrand.cv_results_['params'][lgbrand.best_index_]
        self.val_score = self.model.score(X_val, y_val)
        self.f1_score_val = f1_score(y_val, self.model.predict(X_val))

        return self.val_score
    
#############################################################################################################

    def write_model(self, filename = "../models/lgb_model.pickle"):
        """Write trained model to file."""
        self.filename = filename
        dump(self.model, filename)

#############################################################################################################

    def read_model(self, filename = "../models/lgb_model.pickle"):
        """Read trained model from file."""
        self.filename = filename
        self.model = load(filename)
        
#############################################################################################################

    def predict(self, X):
        """Predict cloud coverage for feature data."""
        return self.model.predict(X)







#############################################################################################################







class XGB():
    """
    Class for handling XGBoost model. Most of this class is reused from the LGBM class
    """

    def __init__(self):
        """
        __init__ taken from cloudynight
        """
        self.data_X = None      # pandas DataFrame
        self.data_y = None      # pandas DataFrame
        self.model = None       # model implementation
        self.filename = None    # model pickle filename
        self.train_score = None # model training score
        self.test_score = None  # model test score
        self.val_score = None   # model validation sample score
        self.f1_score_val = None  # model validation sample f1 score

#############################################################################################################

    def retrieve_training_data(self, data_location):
        """
        Retrieves processed training data from its location

        param : data_location, str to where the training data csv file is located
        """

        n_subregions = 33 # number of subregions in the images

        raw = pd.DataFrame(pd.read_csv(data_location)) # creating a pd dataframe with the raw data

        # taken from cloudynight, processes the data. any unused information will be removed from this list and marked below
        data = pd.DataFrame()
        for j in range(len(raw['moonalt'])):
            frame = pd.DataFrame(OrderedDict(
                (('moonalt', [raw['moonalt'][j]]*n_subregions), # note to self, *n_subregions is there to make all arrays same length and prevent breaking. that it leads to 2.1 million lines of data is apparently just what has to happen
                 ('sunalt', [raw['sunalt'][j]]*n_subregions),
                 ('moonphase', [raw['moonphase'][j]]*n_subregions),
                 ('subid', range(n_subregions)),
                 ('srcdens', raw['srcdens'][j]),
                 ('bkgmean', raw['bkgmean'][j]),
                 ('bkgmedian', raw['bkgmedian'][j]),
                 ('bkgstd', raw['bkgstd'][j]),
                 ('exposure_time', raw['exp_time'][j]),
                 ('cloudy', raw['cloudy'][j]))))
            data = pd.concat([data, frame]) 

        # again taken from cloudynight
        self.data_X = data.drop(['cloudy'], axis=1)
        self.data_y = np.ravel(data.loc[:, ['cloudy']].values).astype(int)
        self.data_X_featurenames = data.drop(['cloudy'], axis=1).columns.values

#############################################################################################################

    def load_data(self, filename):
        """
        Load feature data. Taken from cloudynight
        """
        data = pd.read_csv(filename, index_col=0)

        # split features and target
        self.data_X = data.drop(['cloudy'], axis=1)
        self.data_y = np.ravel(data.loc[:, ['cloudy']].values).astype(int)
        self.data_X_featurenames = data.drop(['cloudy'], axis=1).columns.values

#############################################################################################################

    def train_model(self, parameters = {'max_depth' : 5, 'lambda' : 100,
                                        'learning_rate' : 0.25, 'max_leaves' : 30,
                                        'min_child_weight' : 100, 'alpha' : 10, }, cv = 5):
        """
        Train XGBoost model. Most of the actual content was taken from the LGBM class

        PARAMETERS
        -----------
        parameters : dict 
        Parameters are as close to those used in the LGBM class as was possible to get

        cv : int, default 5
        Cross validation score
   

        """

        # split data into training and validation sample
        X_cv, X_val, y_cv, y_val = train_test_split(self.data_X, self.data_y, test_size=0.2, random_state=42)

        # initialise the model, as much as possible I used the same parameters here as I did with the lgb model
        xgb = XGBClassifier(objective = 'multi:softmax', num_class = 3, n_estimators = 500, random_state = 42, **parameters) 

        #training the model
        xgb.fit(X_cv, y_cv)
        self.model = xgb

        # derive cv scores
        cv_results = cross_validate(xgb, X_cv, y_cv, cv=cv, return_train_score=True)

        # saving to self
        self.train_score = np.max(cv_results['train_score'])
        self.test_score = np.max(cv_results['test_score'])
        self.parameters = parameters
        self.val_score = self.model.score(X_val, y_val)
        self.f1_score_val = f1_score(y_val, self.model.predict(X_val))

        return self.val_score

#############################################################################################################

    def write_model(self, filename = "../models/xgb_model.pickle"):
        """Write trained model to file."""
        self.filename = filename
        dump(self.model, filename)

#############################################################################################################

    def read_model(self, filename = "../models/xgb_model.pickle"):
        """Read trained model from file."""
        self.filename = filename
        self.model = load(filename)
        
#############################################################################################################

    def predict(self, X):
        """Predict cloud coverage for feature data."""
        return self.model.predict(X)







#############################################################################################################
"""
This section contains the process for processing the example image that is seen in the main body of my report.
It also contains the code used in generating the mask.FIT file
"""



#mask_image = processedFIT(../mask/AllSkyImage007537912.FIT)
#mask_image.read_single_fit()
#mask_image.crop()
#mask_image.generate_mask(mask_lwr=5500, gaussian_blur=10)


# testing functions in processedFIT class

#example_image = processedFIT("../example_images/AllSkyImage007553175.FIT")

#example_image.read_single_fit()

# applying mask to test image. shouldn't be run at the same time as crop and generate_mask
#mask = example_image.crop_and_mask()
#example_image.save_image(filename="../example_images/masked_example.FIT")

# generating subregions
#subregions = example_image.generate_subregions()
#example_image.save_subregions()

# testing the subregion overlay, and saving 
#overlay = example_image.create_overlay(overlaytype="subregions", regions=[1,2,3,4,5,6,7,8,9])
#example_image.write_image(filename="../example_images/overlay_example.png", overlay=overlay)

# testing the remove outliers function
#run_algo = example_image.remove_outliers()

#features = example_image.extract_features(subregions)

#print(features)







#############################################################################################################
"""
These two sections are literally just testing to make sure the models can be trained using the already available data
"""




# training the lightgbm model using data from mommert, m. 2020 to just test to see if its working

#start = time.time()

#lgbm_model = LGBM()

# commented out as model has been trained
#lgbm_model.retrieve_training_data("../processed_data/fulltrainingsample_features.dat")

#load_data = time.time() - start

#lgbm_model.train_model()
#lgbm_model.write_model(filename = "../models/model_test.pickle")

#train_lgbm = time.time() - start

# commented out for not working
#lgbm_model.train_randomised_search_cv()
#lgbm_model.write_model(filename = "randomised_search_test.pickle")
#train_randomised_search = time.time() - start

# testing model, taken from https://github.com/mommermi/cloudynight/blob/master/scripts/model_lightgbm.py

# apply model to predict presence of clouds in a random subregion from the training data set
#i = 12345
#print('Is there a cloud in training example {}? {}.'.format(i, lgbm_model.data_y[i] == 1))

#print('The lightgbm model finds {} cloud in this subregion.'.format({1: 'a', 0: 'no'}[lgbm_model.predict(lgbm_model.data_X.iloc[i].values.reshape(1, -1))[0]]))

#predict_clouds = time.time() - start

# build confusion matrix 
#print('confusion matrix:')
#cm = confusion_matrix(lgbm_model.data_y, lgbm_model.predict(lgbm_model.data_X.values),normalize='true')
#tn, fp, fn, tp = cm.ravel()
#print(('true positives: {}\nfalse positives: {}\n'
#       'false negatives: {}\ntrue negatives: {}').format(tn, fp, fn, tp))

#confusion_matrix_time = time.time() - start

#print(f"Times taken to load data/train model/randomised search/predict clouds/build and apply confusion matrix:\n{load_data}\n{train_lgbm}\n{predict_clouds}\n{confusion_matrix_time}")







#############################################################################################################







# training xgboost model, using the same process as before

#start = time.time()

#xgb_model = XGB()

#commented out as model has been trained
#xgb_model.retrieve_training_data("../processed_data/fulltrainingsample_features.dat")

#load_data = time.time() - start

#xgb_model.train_model()
#xgb_model.write_model(filename = "../models/xgb_model_test.pickle")

#train_xgb = time.time() - start

# testing model, taken from https://github.com/mommermi/cloudynight/blob/master/scripts/model_lightgbm.py

# apply model to predict presence of clouds in a random subregion from the training data set
#i = 12345
#print('Is there a cloud in training example {}? {}.'.format(i, xgb_model.data_y[i] == 1))

#print('The xgboost model finds {} cloud in this subregion.'.format({1: 'a', 0: 'no'}[xgb_model.predict(xgb_model.data_X.iloc[i].values.reshape(1, -1))[0]]))

#predict_clouds = time.time() - start

# build confusion matrix 
#print('confusion matrix:')
#cm = confusion_matrix(xgb_model.data_y, xgb_model.predict(xgb_model.data_X.values),normalize='true')
#tn, fp, fn, tp = cm.ravel()
#print(('true positives: {}\nfalse positives: {}\n'
#      'false negatives: {}\ntrue negatives: {}').format(tn, fp, fn, tp))

#confusion_matrix_time = time.time() - start

#print(f"Times taken to load data/train model/randomised search/predict clouds/build and apply confusion matrix:\n{load_data}\n{train_xgb}\n{predict_clouds}\n{confusion_matrix_time}")
