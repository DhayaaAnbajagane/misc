'''
This code extracts profiles for Planck, Spt, and Act clusters.
(i)  Stacks them to obtain the mean result
(ii) Take a log derivative as well
'''

#Import all necessary packages
#Import all packages as needed
import matplotlib.pyplot as plt
%pylab inline
from astropy import units as u
import astropy.io.fits as pf
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
import pandas as pd
from colossus.halo import mass_so, mass_defs, mass_adv
from colossus.cosmology import cosmology
from scipy import signal, interpolate, stats
from tqdm import tqdm
import h5py

#Plotting setup just to make ticks big enough
#Setup plotting environment

import matplotlib as mpl
mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
mpl.rcParams['xtick.major.size'], mpl.rcParams['xtick.minor.size'] = 14, 8
mpl.rcParams['xtick.major.width'], mpl.rcParams['xtick.minor.width'] = 1.2, 0.8
mpl.rcParams['xtick.major.pad'], mpl.rcParams['xtick.minor.pad'] = 10, 10
mpl.rcParams['ytick.major.size'], mpl.rcParams['ytick.minor.size'] = 14, 8
mpl.rcParams['ytick.major.width'], mpl.rcParams['ytick.minor.width'] = 1.2, 0.8

plt.rc('xtick',labelsize=22)
plt.rc('ytick',labelsize=22)

#Functions for logder

def logder_linear(x, data, N=100, xrange = None, window_length=5, polyorder=3):
    '''
    Chihway's code.
    Function that takes x, y (linear values),
    smooths the y values using Savitsky-Golay,
    fits a linear spline to the profile,
    and returns the log derivative dlny/dlnx
    '''
    data_sm = signal.savgol_filter(np.log10(data), window_length=window_length, polyorder=polyorder)

    f = interpolate.interp1d(np.log10(x), data_sm, kind='linear') #'cubic'

    if xrange == None:
        xrange = (np.min(x), np.max(x))

    # Evaluate spline across a very fine grid or radii
    lnrad_fine = np.linspace(np.log10(xrange[0]), np.log10(xrange[1]), num=N)

    lnsigma_fine = f(lnrad_fine)
    # Calculate derivative using finite differencing

    dlnsig_dlnr_fine = (lnsigma_fine[1:] - lnsigma_fine[:-1])/(lnrad_fine[1:] - lnrad_fine[:-1])

    return 10**((lnrad_fine[1:]+lnrad_fine[:-1])/2), dlnsig_dlnr_fine

'''
NOTE: Normally we'd need to compute the profiles here.
      But I already saved the computed profiles so we don't
      have to redo them all over again. Saved profiles include
      100 realizations for each cluster with 100 M500c values
      drawn from gaussian distribution in linear M500c
'''

#Define which datasets we want to look into
keys = ['Planck', 'Spt', 'Act']

#Dictionary whose keys will be Dataset name
#And whose values will be the jackknife stacked
#profiles.
stacked_profile = {}

# Load bins (R/R500c) from the HDF5 file
# There are the values in the center of each bin
# So no need to post-process this array any further
bins_plot = h5py.File('/project2/chihway/dhayaa/Cluster_Profiles.hdf5', 'r')['bins']

for Dataset in keys:

    # Load the data from HDF5 file. This array has three axes.
    # Axis 0 -> Indexes into the individual cluster data, so have length Ncluster
    #           and this value varies per dataset.
    # Axis 1 -> Profile of a single cluster but with 100 values for M500c
    #           Marginalizing over axis 1 marginalizes over the mass variance.
    #           This always has length nBootstrap = 100
    # Axis 2 -> The radial bins (R/R500c) that the profile is computed in.
    #           This always has length nbins = 59

    Data = h5py.File('/project2/chihway/dhayaa/Cluster_Profiles.hdf5', 'r')['%s_Profiles'%Dataset]

    # Create masked_array numpy object which handles the NaN values in the array.
    # This is similar to using np.nansum/nanpercentile etc. on a regular arrays
    # but a masked_array object is way more versatile eg. there's no np.nanaverage()
    # function but with the masked array class we can use np.ma.average()
    Data = np.ma.masked_array(Data, np.isnan(Data))

    # Compute the mean profile and variance in the profile for a single cluster
    # using the 100 realizations with different M500c values
    # NOTE: Both the linear mean and variance of a log-normally distributed variable
    #       don't estimate the true mean and variance. So even the mean might have issues?
    mean_profile = np.ma.mean(Data, axis = 1)
    var_profile  = np.ma.var(Data, axis = 1)

    # Create an index of [1, 2, ...., Ncluster].
    # We will use this for leave-one-out stacking
    cluster_indices = np.arange(Data.shape[0], dtype = int)

    # Initialize the output array in our dictionary
    stacked_profile[Dataset] = np.empty([Data.shape[0], bins_plot.size])

    #Loop over each individual cluster in a dataset
    for i in range(Data.shape[0]):

        stacked_profile[Dataset][i, :] = np.ma.average(mean_profile[np.delete(cluster_indices, i), :],
                                                       axis = 0)

        # If you wanted to weight the stacking, you'd use the following

        # stacked_counts[Dataset][i, :] = np.ma.average(mean_profile[np.delete(cluster_indices, i), :],
        #                                               weights = var_profile[np.delete(cluster_indices, i), :]),
        #                                               axis = 0)


#Once we finish stacking, plot the profiles

plt.figure(figsize=(12,8))
plt.grid()
plt.yscale('log')
plt.xscale('log')

window_l = 9 #Window length of SG filter

#Loop over datasets
for Dataset, color in zip(keys, ['C0', 'C1', 'C2']):

    # Extract the stacked profiles from the dictionary we
    # computed before
    Profiles = stacked_profile[Dataset]

    # Estimate the mean, and 68% bound behavior
    # Smooth profiles in log-space and then transform back to linear
    mean_counts = np.e**signal.savgol_filter(np.log(np.nanpercentile(Profiles, 50, axis = 0)), window_l, 2, mode='nearest')
    high_counts = np.e**signal.savgol_filter(np.log(np.nanpercentile(Profiles, 84, axis = 0)), window_l, 2, mode='nearest')
    low_counts  = np.e**signal.savgol_filter(np.log(np.nanpercentile(Profiles, 16, axis = 0)), window_l, 2, mode='nearest')

    plt.plot(bins_plot, mean_counts, lw = 3, color = color, label = Dataset)
    plt.fill_between(bins_plot, high_counts, low_counts, alpha = 0.4, color = color)

plt.xlabel('R/R500c', size = 30)
plt.ylabel('<y>', size = 35)
plt.legend(fontsize = 25)
plt.show()

#Now plot the derivative
plt.figure(figsize=(12,8))
plt.grid()
plt.xscale('log')

window_l = 9
size = 100 #Sets how many points we compute derivative at

Max_R = {'Planck': 5, 'Spt': 4, 'Act': 18} #Sets the maximum R/R500c value we compute derivative to

#Loop over data again
for Dataset, color in zip(keys, ['C0', 'C1', 'C2']):

    # Pull stacked profiles and setup output array
    Profiles   = stacked_profile[Dataset]
    derivative = np.empty([Profiles.shape[0], size])

    #Loop over data
    for i in range(Profiles.shape[0]):

        # Create mask to throw away all negative and zero values
        # This is needed because we take the log of the profiles
        # and SG filter doesn't handle Infs and NaNs.
        mask = Profiles[i, :] > 0

        # Compute and store derivative for each individual cluster
        # Need N = size + 1, because N sets the number of bin edges
        # and not the number of bins, and we have N_bins = size
        new_bins, derivative[i,:] = logder_linear(bins_plot[mask], Profiles[i, :][mask],
                                                  window_length = window_l,
                                                  xrange = (0.4, Max_R[Dataset]), N = size + 1)

    # We then perform additional "smoothing" onto the log derivative
    # to remove any sharp features due to noise.
    # The for loop is just to apply the filter N = 5 number of times
    # Our main result is not affected by using different values of N
    for i in range(5):

        # Apparently this is called a three-point Hanning filter?
        # Found this form in https://arxiv.org/pdf/1804.10199.pdf, page 6
        # But can't find it when I google though
        derivative[:, 1:-1] = (derivative[:, 2:] + 2*derivative[:, 1:-1] + derivative[:, :-2])/4

    plt.plot(new_bins, np.nanpercentile(derivative, 50, 0),
             lw = 3, color = color, label = Dataset)
    plt.fill_between(new_bins,
                     np.nanpercentile(derivative, 16, 0),
                     np.nanpercentile(derivative, 84, 0), alpha = 0.4, color = color)


plt.ylim(bottom = -5)
plt.xlabel('R/R500c', size = 30)
plt.ylabel('dlnY/dlnR', size = 30)
plt.legend(fontsize = 25, loc = 'lower left')
plt.show()
