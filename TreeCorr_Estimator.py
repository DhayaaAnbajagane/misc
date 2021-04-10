#Need to add chihway directory to PATH variable
#Jupyter doesn't pick it up automagically

import sys
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages')

import treecorr
import healpy as hp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Just plotting setup
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

'''
Planck
'''

#Read cluster catalog and SZ map (and Mask)
planck_data = pd.read_hdf('/project2/chihway/dhayaa/Cluster_Catalog.hdf5', key = 'Planck')
sz_map = hp.read_map('/project2/chihway/dhayaa/Planck_maps/COM_CompMap_YSZ_R2.02/milca_ymaps.fits')
Mask   = hp.read_map('/project2/chihway/dhayaa/Planck_maps/COM_CompMap_YSZ_R2.02/COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
                     field = 4).astype(bool)

#Get phi, theta for healpix pixels and apply mask as well
nside = 2048
pix = np.arange(hp.nside2npix(nside))[Mask]
theta, phi = hp.pix2ang(nside, pix)
SZ = sz_map[Mask]

#Set-up treecorr objects
Clusters = treecorr.Catalog(ra=planck_data.GLON.values, dec=planck_data.GLAT.values,
                            ra_units='deg', dec_units='deg', npatch = 1)

CMB_y    = treecorr.Catalog(ra=phi, dec=np.pi/2 - theta,
                            k = SZ, ra_units='rad', dec_units='rad')

#Compute correlation
planck_halo_y = treecorr.NKCorrelation(nbins = 70, min_sep = 1, max_sep = 10 * 60, sep_units = 'arcmin')
planck_halo_y.process(Clusters, CMB_y)


'''
SPT
'''

#Read cluster catalog and SZ map (and Masks)
spt_data = pd.read_hdf('/project2/chihway/dhayaa/Cluster_Catalog.hdf5', key = 'SPT')
sz_map   = hp.read_map('/project2/chihway/dhayaa/SPT_maps/Planck_SPT_100_353_ymapsept15_min_var_nside_8192.fitsmap.fits')

Mask = hp.read_map('/project2/chihway/dhayaa/SPT_maps/final_sptsz_point_source_mask_nside_8192_binary_mask.fits')
Mask *= hp.read_map('/project2/chihway/dhayaa/SPT_maps/dust_mask__top_5percent_8192.fits')

#Get phi, theta for healpix pixels and apply mask as well
nside = 8192
pix = np.arange(hp.nside2npix(nside))[Mask.astype(bool)]
SZ  = sz_map[Mask.astype(bool)]
theta, phi = hp.pix2ang(nside, pix)

#Set-up treecorr objects
Clusters = treecorr.Catalog(ra=spt_data.RA.values, dec=spt_data.DEC.values,
                            ra_units='deg', dec_units='deg', npatch = 1)

CMB_y    = treecorr.Catalog(ra=phi, dec=np.pi/2 - theta, k = SZ,
                            ra_units='rad', dec_units='rad')

#Compute correlation
spt_halo_y = treecorr.NKCorrelation(nbins = 70, min_sep = 1, max_sep = 10 * 60, sep_units = 'arcmin')
spt_halo_y.process(Clusters, CMB_y)



# Plot the correlations
plt.figure(figsize = (12,12))
plt.grid()
plt.loglog()


bins_plot = np.exp(planck_halo_y.logr)
Xi, err   = planck_halo_y.calculateXi()

plt.plot(bins_plot, Xi, lw = 6, color = 'C0')
plt.fill_between(bins_plot, Xi + np.sqrt(err), Xi - np.sqrt(err),
                 lw = 6, color = 'C0', alpha = 0.3)

bins_plot = np.exp(spt_halo_y.logr)
Xi, err   = spt_halo_y.calculateXi()

plt.plot(bins_plot, Xi, lw = 6, color = 'C1')
plt.fill_between(bins_plot, Xi + np.sqrt(err), Xi - np.sqrt(err),
                 lw = 6, color = 'C1', alpha = 0.3)


plt.xlabel(r'R [arcmin]', size = 30)
plt.ylabel(r'$\langle y\rangle$', size = 35)
plt.ylim(1e-8)
plt.show()
