#Need to add chihway directory to PATH variable
#Jupyter doesn't pick it up automagically

import sys
sys.path.append('/project2/chihway/virtualenvs/midway2_python3/lib/python3.7/site-packages')

#Import all packages as needed
import astropy.io.fits as pf
import healpy as hp
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
import numpy as np
import treecorr
import pandas as pd
from tqdm import tqdm
import h5py

from colossus.halo import mass_so, mass_defs, mass_adv
from colossus.cosmology import cosmology
cosmology.setCosmology('planck13')

#Dictionary that converts
#snap number for The300 to redshift
snap_to_z = {'123' : 0.117194,
             '120' : 0.193887,
             '116' : 0.304461,
             '113' : 0.394117,
             '110' : 0.489869,
             '107' : 0.592357,
             '104' : 0.701838,
             '101' : 0.818843,
             '99'  : 0.901141,
             '96'  : 1.031694}

#Get snapshot number that is inputted to script
snap = int(sys.argv[1])
z    = snap_to_z[str(snap)]

#Function for rotation + interpolation of noise
def get_noise(Noise_map, theta, phi, theta1, theta2, phi1, phi2):
    """
    This function (i) rotates the sim cutout to some location in
    noise map, (ii) interpolates to get the noise vals at new locations,
    of sim map and; (iii) returns those noise vals

    --------
    Inputs
    --------

    Noise_map:
        full healpy map

    theta, phi:
        Coordinates of the SIMULATION map

    theta1, phi1:
        Coordinates of the sim map center

    theta2, phi2:
        New coordinates the sim map should be
        centered on after rotation.
    """

    # Get new theta, phi of the rotated sim map via 2-step procedure
    # Dhayaa : I found this two-step rotation by trial and error
    #        : Confirmed visually that it rotates to the right position
    trot, prot = hp.Rotator(deg=False, rot=[phi1, theta2 - theta1])(theta, phi, deg = False)
    trot, prot = hp.Rotator(deg=False, rot=[-phi2, 0])(trot, prot, deg = False)

    # Interpolate map onto these coordinates
    Noise = hp.get_interp_val(Noise_map, trot, prot)

    return Noise

# load SPT maps
# Not using a Mask because point sources/dust
# are "signals" and so are removed in differencing below.
# Don't know how many point sources are transient though..?

NSIDE     = 8192
spt_half1 = hp.read_map('/project2/chihway/dhayaa/SPT_maps/Planck_SPT_100_353_ymapsept15_min_var_half1_nside_8192.fitsmap.fits')
spt_half2 = hp.read_map('/project2/chihway/dhayaa/SPT_maps/Planck_SPT_100_353_ymapsept15_min_var_half2_nside_8192.fitsmap.fits')

# Take difference to get noise
# Need sqrt(2) since difference integrates down noise
Noise_map = (spt_half1 - spt_half2)/2 * np.sqrt(2)

# Delete variables just to free up memory
# Don't need these anymore
del spt_half1
del spt_half2

# Generate randoms for ROTATIONS (not 2pt background estimation)
# Start by generating many many points on the sphere
N = 20000000
phi_rand   = np.random.uniform(0, 2*np.pi, N)
theta_rand = np.arccos(1 - 2*np.random.uniform(0, 1, N))

# Remove all points that are outside the SPT map.
# "pix_mask" is smaller than SPT map so that
# we don't select any randoms that are "half-in, half-out"
# in the map.
pix_mask = (theta_rand < 2.52) & (theta_rand > 2.45) & ((phi_rand > 5.5 ) | (phi_rand < 1.5))
phi_rand, theta_rand = phi_rand[pix_mask], theta_rand[pix_mask]

print("Using %d Randoms"%phi_rand.size)

'''
Generate profiles
'''

h = 0.6777
n_clusters = 324
path = '/project2/chihway/dhayaa/Postprocessed/'

# n_noise sets how many noise realizations are used per cluster
bin_min, bin_max, nbins, n_noise = 0.1, 20, 50, 200

# Array of bins in units r/R200m
bins_plot   = np.geomspace(bin_min, bin_max, nbins, endpoint = True)

#Setup output arrays
Profile     = np.zeros([n_noise, 324, nbins])
npairs      = np.zeros([n_noise, 324, nbins])

# Index for the random locations in noise map
# Will use this to make sure no location is used
# twice
Noise_indices = np.arange(phi_rand.size).astype(int)

#Loop over each cluster
for i in tqdm(range(1, n_clusters + 1)):

    #Edge case: The300 don't have the file for
    #           this cluster (at this snapshot alone)
    if (i == 228) & (snap == 110): continue

    #Setup string to use when reading in files
    name = '0'*(4 - len(str(i))) + str(i)

    #Read in healpy map of cluster (I generated this from Weiguang's original maps)
    Data = pd.read_hdf(path + '/The300_Maps_snap_' + str(snap) + '.hdf5', key = 'Spt_Cluster_' + str(i))

    #Get angular position + SZ val
    theta_The300, phi_The300 = hp.pix2ang(NSIDE, Data.hpix.values)
    SZ = Data['SZ'].values.copy()/(1 + z) #Need 1/(1 + z) factor to correct bug in map

    # Loop to generate the N = n_noise noise realizations
    # for each cluster
    for j in range(n_noise):

        #Pick a random location in the noise map from the list of
        #random locations we generated before. By using Noise_indices,
        #we make sure we never select same location twice
        rand_j = np.random.randint(0, Noise_indices.size, 1)
        ind    = Noise_indices[rand_j]

        #theta/phi1 are cluster location.
        #theta/phi2 are random's location
        theta1, theta2 = (90 - 27.98)*np.pi/180, theta_rand[ind]
        phi1,   phi2   = 194.95*np.pi/180, phi_rand[ind]

        #Get Noise for each pixel of The300's cluster map
        rotated_Noise = get_noise(Noise_map, theta_The300, phi_The300, theta1, theta2, phi1, phi2)

        # Remove the index, ind, from Noise_indices so this specific
        # location isn't picked again in future loops
        Noise_indices = np.delete(Noise_indices, rand_j)

        #Get new "noise-added" SZ
        Contaminated_SZ = SZ + rotated_Noise
        DEC = 90. - theta_The300*180/np.pi
        RA  = phi_The300*180/np.pi

        #Construct y-map TreeCorr catalog
        CMB_y   = treecorr.Catalog(ra = RA, dec = DEC, k = Contaminated_SZ, ra_units = 'deg', dec_units = 'deg')

        # Now get relevant halo params from file
        Relaxation_Params = pd.read_csv('/project2/chihway/dhayaa/The300_maps/The300_relaxation_params/GadgetX_ALL_NewMDCLUSTER_' + name + '.dat',
                                            delimiter = ' ', skiprows = 0, skipinitialspace = True)

        Relaxation_Params = Relaxation_Params[Relaxation_Params.snapnum.values == snap]

        CRVAL1, CRVAL2 = 194.95, 27.98

        #Find angular position of cluster center in map
        #(Maps are centered on sim-volume, not cluster center)
        Xc_ang = (Relaxation_Params.Xc.values[0] * 1e-3 - 500)/h/(1 + z) / cosmo.angular_diameter_distance(z).value
        Yc_ang = (Relaxation_Params.Yc.values[0] * 1e-3 - 500)/h/(1 + z) / cosmo.angular_diameter_distance(z).value + CRVAL2 *np.pi/180

        center = (Xc_ang/np.pi*180 / np.cos(Yc_ang) + CRVAL1, Yc_ang/np.pi*180)

        #Get R200m for each cluster (assuming c200c-M200c
        # relation from Diemer&Joyce2019)
        M200c = Relaxation_Params.M200c.values[0]
        R200m = mass_adv.changeMassDefinitionCModel(M200c, z, '200c', '200m')[1]*1e-3/h #in units of physical Mpc

        D_A = cosmo.angular_diameter_distance(z).value
        theta_min = bin_min * R200m/D_A * 180/np.pi * 60
        theta_max = bin_max * R200m/D_A * 180/np.pi * 60

        #Generate 2pt function
        Clusters = treecorr.Catalog(ra  = np.atleast_1d(center[0]),
                                    dec = np.atleast_1d(center[1]),
                                    ra_units = 'deg', dec_units = 'deg')

        two_pt_corr = treecorr.NKCorrelation(nbins = nbins, min_sep = theta_min, max_sep = theta_max,
                                             sep_units = 'arcmin', bin_slop = 0)
        two_pt_corr.process(Clusters, CMB_y)

        #Note that I DON'T do background subtraction here.
        #The sim profiles don't have a background given their zoom-in
        #nature (sim regions extend to around 5-6R200m at best).
        #Noise map by definition will have the background subtracted out as well.

        Profile[j, i - 1, :] = two_pt_corr.raw_xi
        npairs[j, i - 1, :]  = two_pt_corr.npairs

#Save data in hdf5 file
with h5py.File('The300_Noise_snap%s_TreeCorr_Profiles.hdf5'%snap, 'w') as f:

    f.create_dataset('Profiles', data = Profile)
    f.create_dataset('Npairs',   data = npairs)
    f.create_dataset('bins',     data = bins_plot)
