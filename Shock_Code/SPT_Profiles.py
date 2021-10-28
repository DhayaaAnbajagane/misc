import sys
import astropy.io.fits as pf
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo

import pandas as pd
from colossus.halo import mass_so, mass_defs, mass_adv
from colossus.cosmology import cosmology

from scipy import signal, interpolate, stats
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages')

'''
Part 1: Store cutouts for each cluster

It's just faster to do this so you don't
compute separations for all pixels every time
you need to recompute profiles
(eg. after binning change, radius change etc.)
'''

spt_data = pd.read_hdf('/project2/chihway/dhayaa/Cluster_Catalog.hdf5', key = 'SPT')

sz_map = hp.read_map('/project2/chihway/dhayaa/SPT_maps/Planck_SPT_100_353_ymapsept15_min_var_nside_8192.fitsmap.fits')

Mask = hp.read_map('/project2/chihway/dhayaa/SPT_maps/final_sptsz_point_source_mask_nside_8192_binary_mask.fits')
Mask *= hp.read_map('/project2/chihway/dhayaa/SPT_maps/dust_mask__top_5percent_8192.fits')


nside = 8192
pix = np.arange(hp.nside2npix(nside))[Mask.astype(bool)]
SZ  = sz_map[Mask.astype(bool)]
theta, phi = hp.pix2ang(nside, pix)
c2 = SkyCoord(phi, np.pi/2 - theta, unit = u.rad, frame = 'icrs')


import warnings
import tables

# Need this since cluster names are not proper python identifiers
# but still usable in our context
warnings.simplefilter('ignore', tables.NaturalNameWarning)

#Setting N = 2 just as an example
N = 2 #len(spt_data)

for i in range(N):

    # Need cluster_ID when storing data
    cluster_ID = spt_data.SPT_ID.values[i]

    c1 = SkyCoord(spt_data.RA.values[i], spt_data.DEC.values[i], unit = u.degree, frame = 'icrs')

    R500c = spt_data['R500c'].values[i] #in units of physical Mpc

    # Convert maximum radius from units of R500c to radians
    Max = R500c/cosmo.angular_diameter_distance(spt_data['REDSHIFT'].values[i]).value*200

    sep = c1.separation(c2)
    Mask = sep.degree < Max*180/np.pi

    #Store data
    Data = pd.DataFrame()
    Data['SZ']    = SZ[Mask]
    Data['sep']   = sep[Mask]

    # Data.to_hdf(some path file, key = 'Cluster_' + cluster_ID)

    # If you want to view the actual file I use for SPT, it's at:
    # '/project2/chihway/dhayaa/Postprocessed/Spt_Maps_Subset_Masked_Massive.hdf5'

'''
Part 2: Compute profiles
'''

bins = np.geomspace(1/60, 10, 70) #In degrees

#Once again, setting N = 2 just as an example
N = 2 #len(spt_data) #Number of clusters

counts = np.zeros([N, bins.size - 1])

for i in tqdm(range(N)):

    Data = pd.read_hdf('/project2/chihway/dhayaa/Postprocessed/Spt_Maps_Subset_Masked_Massive.hdf5',
                       key = 'Cluster_' + data.name.values[i])

    # If for whatever reason (mostly masking)
    # the cluster is fully in a hole, with
    # no usable pixels, then skip it
    if len(Data) == 0:
        counts[Dataset][i, :, :] = np.NaN
        continue

    # Read out data and mask it just so
    # computation is easier. Our maximum
    # separation is 10 degrees, so don't
    # need data beyond that.
    sep, SZ = Data.sep.values, Data.SZ.values
    sep, SZ = sep[sep < 10], SZ[sep < 10]
    counts[i, :]  = stats.binned_statistic(sep, SZ, 'mean', bins)[0]

print("Finished with", Dataset)
