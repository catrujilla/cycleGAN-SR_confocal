'''
This Python script implements several functions for evaluating image quality metrics and processing using machine learning models. 
Developed by Ana Doblas and Carlos Trujillo, the script leverages libraries like NumPy, TensorFlow, OpenCV, and SciPy. 
Key functionalities include generating fake and real image samples, computing metrics such as PSNR, SSIM, SNR, and cutoff frequency. 
These metrics are essential for image resolution quantification and quality assessment in various applications.

Functions:

generate_fake_samples: Generates fake images using a given model.
generate_real_samples: Retrieves real images from a dataset.
psnr: Computes Peak Signal-to-Noise Ratio between real and generated images.
avg_SSIM: Calculates Average Structural Similarity Index between real and generated images.
snr: Computes Signal-to-Noise Ratio of an image.
snr_batch: Computes average Signal-to-Noise Ratio for a batch of generated images.
bgstd_batch: Computes background standard deviation from generated images compared to ground truth.
tsm: Calculates Total Sum of the modified image.
cutoff_metric: Computes cutoff frequency metric for image resolution assessment.
cutoff_batch: Computes mean and standard deviation of cutoff frequencies for a batch of images.
Initial Release: February 22, 2023
Last Modification: March 29, 2023

Authors: Dr. Ana Doblas and Dr. Carlos Trujillo
Affiliations: University of Massachussets Dartmouth, Universidad EAFIT
'''

import numpy
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randint
import tensorflow as tf
import cv2

from scipy import fftpack
from scipy.interpolate import UnivariateSpline


def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create ✬fake✬ class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X

def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate ✬real✬ class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2]

def std_window(img, g_model, window):
    a = window[0]
    b = window[1]
    c = window[2]
    d = window[3]
    img_std = generate_fake_samples(g_model, img, 1)
    img_std = img_std[0]
    img_std = (img_std + 1) / 2.0
    return np.std(img_std[a:b, c:d])


def psnr(g_model, dataset, image_shape, n_samples=15):

    [X_realA, X_realB] = generate_real_samples(dataset, n_samples, 1)
    X_fakeB = generate_fake_samples(g_model, X_realA, 1)

    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    #psnr = 0
    #mse_tot = 0
    mse_tot = [0] * image_shape[0]
    psnr_tot = [0] * image_shape[0]
    for i in range(n_samples):
        mse = np.sum((X_realB[i] - X_fakeB[i])**2)/np.prod(image_shape)
        mse_tot[i] = mse
        psnr_tot[i] = 20*np.log10(1) - 10*np.log10(mse)
        
    psnr_mean = np.mean(psnr_tot)
    psnr_std = np.std(psnr_tot)

    mse_mean = np.mean(mse_tot)
    mse_std = np.std(mse_tot)
    
    return psnr_mean, psnr_std, mse_mean, mse_std

def avg_SSIM(g_model, dataset, image_shape, n_samples=15):
    [X_realA, X_realB] = generate_real_samples(dataset, n_samples, 1)
    X_fakeB = generate_fake_samples(g_model, X_realA, 1)

    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    
    #new adds
    X_fakeB = tf.convert_to_tensor(X_fakeB)
    X_realB = tf.convert_to_tensor(X_realB)
    
    ssim = tf.image.ssim(X_fakeB, X_realB, 1)

    mean_ssim = tf.reduce_sum(ssim) / n_samples
    std_ssim = tf.math.reduce_std(ssim)

    return tf.keras.backend.eval(mean_ssim), tf.keras.backend.eval(std_ssim)

    
def snr(img):
    mean = np.mean(img)
    std = np.sqrt(np.var(img))
    return mean/std

def snr_batch(g_model, dataset, n_samples=15):
    [X_realA, _] = generate_real_samples(dataset, n_samples, 1)
    X_fakeB = generate_fake_samples(g_model, X_realA, 1)

    X_fakeB = (X_fakeB + 1) / 2.0

    snr_tot = 0

    for i in range(n_samples):
        snr_img = snr(X_fakeB[i])
        snr_tot += snr_img
    return snr_tot/n_samples

def bgstd_batch(g_model, dataset, groundtruthset, n_samples=15):
    [X_realA, _] = generate_real_samples(dataset, n_samples, 1)
    [_, X_realB] = generate_real_samples(groundtruthset, n_samples, 1)
    X_fakeB = generate_fake_samples(g_model, X_realA, 1)

    X_realB = 127.5 * X_realB + 127.5
    X_fakeB = 127.5 * X_fakeB + 127.5

    std = np.zeros((n_samples,))

    for i in range(n_samples):
        g_truth = X_realB[i]
        g_truth = g_truth[:, :, 0]
        g_truth = g_truth.astype("uint8")
        img = X_fakeB[i]
        img = img[:, :, 0]
        img = img.astype("uint8")
        blur1 = cv2.GaussianBlur(g_truth, (151, 151), 0)
        blur2 = cv2.GaussianBlur(g_truth, (5, 5), 0)

        ret, otsu1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, otsu2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closing1 = cv2.morphologyEx(otsu1, cv2.MORPH_CLOSE, kernel)
        closing2 = cv2.morphologyEx(otsu2, cv2.MORPH_CLOSE, kernel)

        orimg = np.logical_or(closing1, closing2)
        bg_img = np.logical_not(orimg)

        bg_flat = bg_img.flatten()
        img_flat = img.flatten()

        bg_int = [0] * img_flat.shape[0]
        for j in range(img_flat.shape[0]):
            if bg_flat[j] == 1:
                bg_int[j] = img_flat[j]

        std[i] = np.std(bg_int)
        
    mean = np.mean(std)
    std_t = np.std(std)
    
    return mean, std_t

def tsm(g_model, dataset, image_shape, n_samples=15):

    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    X_fakeB = (X_fakeB + 1) / 2.0

    thresh_XfakeB = X_fakeB < .02
    total_black = np.sum(thresh_XfakeB)
    return 1 - total_black/np.prod(image_shape)

"""
Code developed by Ana Doblas to compute a cutoff frequency metric for
image resolution quantification. The higher the value of the resulting metric, 
the better the resolution of the inspected image.

Initial release: February 22, 2023.
Last modification: March 29, 2023.

Authors: Ana Doblas, Carlos Trujillo

Affiliations: The University of Memphis, Universidad EAFIT
"""

def cutoff_metric(file_name):

    I_lr = file_name

    #Convert the image to double-precision floating-point.
    I_lr = I_lr.astype(float)
    
    #print (I_lr.dtype, I_lr.shape)

    #Compute the 2-D FFT and shift the zero-frequency component to the center of the array.
    lr_fft = fftpack.fftshift(fftpack.fft2(I_lr))

    #Compute the magnitude spectrum.
    lr_fft = np.abs(lr_fft)

    #Display the magnitude spectrum in log scale.
    #plt.imshow(np.log(lr_fft), cmap='gray')
    #plt.show()

    #Determine the size of the magnitude spectrum.
    img_size = lr_fft.shape

    #Store the size of the magnitude spectrum along each dimension.
    Nx = img_size[0]
    Ny = img_size[1]

    #Determine the coordinates of the center pixel in the magnitude spectrum.
    x0 = Nx/2 + 1
    y0 = Ny/2 + 1

    #Define a set of radii to crop the magnitude spectrum.
    radius = np.arange(1, Nx, 5)

    # For each radius, define a binary circular mask and compute the sum of the
    # magnitude spectrum within the mask.

    # Fast version

    # Create arrays of x and y coordinates
    x = np.arange(Nx)
    y = np.arange(Ny)
    xx, yy = np.meshgrid(x, y)

    # Compute distances from center point to all pixels
    distances = np.sqrt((xx-x0)**2 + (yy-y0)**2)

    # Create mask as a boolean array using distances
    mask = np.logical_and(distances >= radius[:-1][:, np.newaxis, np.newaxis], distances < radius[1:][:, np.newaxis, np.newaxis])

    # Convert boolean mask to integer array
    mask = mask.astype(int)

    # Multiply Fourier transform with mask and sum along x and y axes
    ft_profile = np.sum(lr_fft*mask, axis=(1,2))

    #plt.plot(ft_profile) # Plot the magnitude spectrum (commented out)
    #plt.show()

    cutoff_energy = 0.95*np.sum(ft_profile) # Compute the cutoff energy as 95% of the total energy

    energy = np.zeros(len(ft_profile))
    for x in range(len(ft_profile)):
        energy[x] = np.sum(ft_profile[:x+1]) # Compute the cumulative energy up to each frequency

    # Prepare the cumulative energy data for curve fitting
    xData, yData = np.arange(len(energy)), energy

    # Fit model to data.
    #fitresult = UnivariateSpline(xData, yData, s=0.1*np.sum(ft_profile))
    fitresult = UnivariateSpline(xData, yData, s=0.01, k=3) # Fit the smoothing spline to the cumulative energy data

    vect_frequency = np.arange(1, 300, 0.1) # Define a range of frequencies to evaluate the fitted curve at

    a = fitresult(vect_frequency) # Evaluate the fitted curve at each frequency
    #plt.plot(vect_frequency,a,'b') # Plot the fitted curve
    #plt.show()

    n = np.argwhere(a > cutoff_energy) # Find the index of the frequency where the fitted curve equals the cutoff energy

    # Estimate the cutoff frequency doing a linear interpolation between two values
    y1, y0, y = a[n[0]][0], a[n[0]-1][0], cutoff_energy # Define variables for linear interpolation
    #print (y1, y0, y)
    x0, x1 = vect_frequency[n[0]-1], vect_frequency[n[0]] # Define variables for linear interpolation
    #print (x0, x1)
    x = x0 + (y-y0)/(y1-y0)*(x1-x0) # Perform linear interpolation to estimate the cutoff frequency

    cutoff_freq = x # Output the estimated cutoff frequency

    #print (cutoff_freq)
    
    return cutoff_freq

def cutoff_batch(g_model, dataset, image_shape, n_samples=15):

    [X_realA, _] = generate_real_samples(dataset, n_samples, 1)
    X_fakeB = generate_fake_samples(g_model, X_realA, 1)

    X_fakeB = (X_fakeB + 1) / 2.0
    
    cutoff = np.zeros((n_samples,))
    
    for i in range(n_samples):
        #print (X_fakeB[i,:,:,0].shape)
        cutoff[i] = cutoff_metric(X_fakeB[i,:,:,0])
    
    mean = np.mean(cutoff)
    std = np.std(cutoff)

    return mean, std