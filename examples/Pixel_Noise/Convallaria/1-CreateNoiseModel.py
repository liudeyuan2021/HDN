if __name__ == '__main__':
    # Generate a Noise Model using Calibration Data 
    # We will use pairs of noisy calibration observations $x_i$ and clean signal $s_i$ (created by averaging these noisy, calibration images) to estimate the conditional distribution $p(x_i|s_i)$. Histogram-based and Gaussian Mixture Model-based noise models are generated and saved and are later used for training.
    # __Note:__ Noise model can also be generated if calibration data is not available. In such a case, we use an approach called ```Bootstrapping```. Take a look at the bootstrapping notebook [here](https://github.com/juglab/PPN2V/blob/master/examples/Convallaria/PN2V/1b_CreateNoiseModel_Bootstrap.ipynb). To understand more about the ```Bootstrapping``` procedure, take a look at the readme [here](https://github.com/juglab/PPN2V).

    import warnings
    warnings.filterwarnings('ignore')
    import torch
    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from torch.distributions import normal
    import matplotlib.pyplot as plt, numpy as np, pickle
    from scipy.stats import norm
    from tifffile import imread
    import sys
    sys.path.append('../../../')
    #from pn2v import *
    from lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
    from lib import histNoiseModel
    from lib.utils import plotProbabilityDistribution
    import os
    import urllib
    import zipfile

    # Download the data from https://zenodo.org/record/5156913/files/Convallaria_diaphragm.zip?download=1. Here we show the pipeline for Convallaria dataset also used in this [paper](https://ieeexplore.ieee.org/abstract/document/9098612/). Save the dataset in an appropriate path. For us, the path is the ```data``` folder which exists at ```./```.
    if not os.path.isdir('./data'):
        os.mkdir('./data')

    zipPath="./data/Convallaria_diaphragm.zip"
    if not os.path.exists(zipPath):  
        data = urllib.request.urlretrieve('https://zenodo.org/record/5156913/files/Convallaria_diaphragm.zip?download=1', zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall("./data")
    
    # The noise model is a characteristic of your camera. The downloaded data folder contains a set of calibration images (For the Convallaria dataset, it is ```20190726_tl_50um_500msec_wf_130EM_FD.tif``` and the data to be denoised is named ```20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif```). We can either bin the noisy - GT pairs (obtained from noisy calibration images) as a 2-D histogram or fit a GMM distribution to obtain a smooth, parametric description of the noise model.
    # Specify ```path```, ```dataName```,  ```n_gaussian```, ```n_coeff```.
    # The default choices for these values generally work well for most datasets. 

    path="./data/Convallaria_diaphragm/"
    dataName = 'convallaria' # Name of the noise model 
    n_gaussian = 3 # Number of gaussians to use for Gaussian Mixture Model
    n_coeff = 2 # No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components.

    observation= imread(path+'20190726_tl_50um_500msec_wf_130EM_FD.tif') # Load the appropriate data
    nameHistNoiseModel ='HistNoiseModel_'+dataName+'_'+'calibration'
    nameGMMNoiseModel = 'GMMNoiseModel_'+dataName+'_'+str(n_gaussian)+'_'+str(n_coeff)+'_'+'calibration'

    # The data contains 100 images of a static sample.
    # We estimate the clean signal by averaging all images.

    signal=np.mean(observation[:, ...],axis=0)[np.newaxis,...]

    # Let's look the raw data and our pseudo ground truth signal
    print(signal.shape)
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 2)
    plt.title(label='average (ground truth)')
    plt.imshow(signal[0],cmap='gray')
    plt.subplot(1, 2, 1)
    plt.title(label='single raw image')
    plt.imshow(observation[0],cmap='gray')
    plt.show()

    # Creating the Histogram Noise Model
    # Using the raw pixels $x_i$, and our averaged GT $s_i$, we are now learning a histogram based noise model. It describes the distribution $p(x_i|s_i)$ for each $s_i$. 

    # We set the range of values we want to cover with our model.
    # The pixel intensities in the images you want to denoise have to lie within this range.
    # The dataset is clipped to values between 0 and 255.
    minVal, maxVal = 234, 7402
    bins = 256

    # We are creating the histogram.
    # This can take a minute.
    histogram = histNoiseModel.createHistogram(bins, minVal, maxVal, observation,signal)

    # Saving histogram to disc.
    np.save(path+nameHistNoiseModel+'.npy', histogram)
    histogramFD=histogram[0]

    # Let's look at the histogram-based noise model.
    plt.xlabel('Observation Bin')
    plt.ylabel('Signal Bin')
    plt.imshow(histogramFD**0.25, cmap='gray')
    plt.show()

    # Creating the GMM noise model
    # Using the raw pixels $x_i$, and our averaged GT $s_i$, we are now learning a GMM based noise model. It describes the distribution $p(x_i|s_i)$ for each $s_i$.
 
    min_signal=np.min(signal)
    max_signal=np.max(signal)
    print("Minimum Signal Intensity is", min_signal)
    print("Maximum Signal Intensity is", max_signal)

    # Iterating the noise model training for `n_epoch=2000` and `batchSize=250000` works the best for `Convallaria` dataset. 

    gaussianMixtureNoiseModel = GaussianMixtureNoiseModel(min_signal = min_signal,
                                                                                max_signal =max_signal,
                                                                                path=path, weight = None, 
                                                                                n_gaussian = n_gaussian,
                                                                                n_coeff = n_coeff,
                                                                                min_sigma = 50, 
                                                                                device = device)
    
    gaussianMixtureNoiseModel.train(signal, observation, batchSize = 250000, n_epochs = 2000, learning_rate=0.1, name = nameGMMNoiseModel)

    plotProbabilityDistribution(signalBinIndex=25, histogram=histogramFD, gaussianMixtureNoiseModel=gaussianMixtureNoiseModel, min_signal=minVal, max_signal=maxVal, n_bin= bins, device=device)