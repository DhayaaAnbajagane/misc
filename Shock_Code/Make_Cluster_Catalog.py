#Import all packages as needed
import matplotlib.pyplot as plt, numpy as np
import pandas as pd
from scipy import signal, interpolate, stats
import h5py

import astropy.io.fits as pf
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
import joblib

import pandas as pd
from colossus.halo import mass_so, mass_defs, mass_adv
from colossus.cosmology import cosmology
from colossus.lss import peaks
cosmology.setCosmology('planck15')

'''
--------------------------------
Fiducial Planck catalog
--------------------------------
'''

planck = pf.open('/project2/chihway/dhayaa/Planck_maps/HFI_PCCS_SZ-union_R2.08.fits')
planck = Table(planck[1].data).to_pandas()

#Throw out all z < 0 objects
planck = planck.loc[planck.REDSHIFT.values > 0]

#Make MSZ log10 in units of Msun
planck['M500c'] = np.log10(planck.MSZ.values*1e14*1.28) #The x1.28 is a necessary correction factor
planck['M_ERR'] = (planck.MSZ_ERR_UP.values + planck.MSZ_ERR_LOW.values)/2*1e14

#Get an R500c (in physical Mpc)
planck['R500c'] = mass_so.M_to_R(10**planck.M500c.values*cosmo.h,
                                      planck.REDSHIFT.values, '500c')/cosmo.h*1e-3

#Get theta and phi (use celestial coordinates RA/DEC, not galactic)
planck['theta'] = (90.- planck['DEC'].values)/180*np.pi
planck['phi']   = planck['RA'].values/180*np.pi

planck['name']  = planck['NAME']

'''
--------------------------------
SPT-SZ
--------------------------------
'''

spt = pf.open('/project2/chihway/dhayaa/SPT_maps/2500d_cluster_sample_Bocquet19.fits')
spt = Table(spt[1].data).to_pandas()

#Throw out all z < 0 objects
spt = spt.loc[spt.REDSHIFT.values > 0]

spt['R500c'] = mass_so.M_to_R(10**14*spt.M500.values*cosmo.h,
                                   spt.REDSHIFT.values, '500c')/cosmo.h*1e-3

spt['M500c'] = np.log10(10**14*spt.M500)
spt['M_ERR'] = (spt.M500_uerr.values + spt.M500_lerr.values)/2*10**14
#Get theta and phi
spt['theta'] = (90.- spt['DEC'].values)/180*np.pi
spt['phi']   = spt['RA'].values/180*np.pi

spt['SNR']  = spt['XI']
spt['name'] = spt['SPT_ID']


'''
--------------------------------
SPT-ECS
--------------------------------
'''

spt_ecs = pf.open('/project2/chihway/dhayaa/SPT_maps/sptecs_catalog_oct919.fits')
spt_ecs = Table(spt_ecs[1].data).to_pandas()

#Throw out all z < 0 objects
spt_ecs = spt_ecs.loc[spt_ecs.REDSHIFT.values > 0]

spt_ecs['R500c'] = mass_so.M_to_R(10**14*spt_ecs.M500.values*cosmo.h,
                                   spt_ecs.REDSHIFT.values, '500c')/cosmo.h*1e-3

spt_ecs['M500c'] = np.log10(10**14*spt_ecs.M500)
spt_ecs['M_ERR'] = (spt_ecs.M500_UERR.values + spt_ecs.M500_LERR.values)/2*10**14
#Get theta and phi
spt_ecs['theta'] = (90.- spt_ecs['DEC'].values)/180*np.pi
spt_ecs['phi']   = spt_ecs['RA'].values/180*np.pi

spt_ecs['SNR']  = spt_ecs['XI']
spt_ecs['name'] = spt_ecs['SPT_ID']

'''
--------------------------------
SPTpol 100d
--------------------------------
'''

spt_100d = pf.open('/project2/chihway/dhayaa/SPT_maps/sptpol100d_catalog_huang19.fits')
spt_100d = Table(spt_100d[1].data).to_pandas()
spt_100d['REDSHIFT'] = spt_100d['redshift']

#Throw out all z < 0 objects
spt_100d = spt_100d.loc[spt_100d.REDSHIFT.values > 0]

spt_100d['R500c'] = mass_so.M_to_R(10**14*spt_100d.M500.values*cosmo.h,
                                   spt_100d.REDSHIFT.values, '500c')/cosmo.h*1e-3

spt_100d['M500c'] = np.log10(10**14*spt_100d.M500)
spt_100d['M_ERR'] = (spt_100d.M500_uerr.values + spt_100d.M500_lerr.values)/2*10**14

#Get theta and phi
spt_100d['DEC']   = spt_100d['Dec']
spt_100d['theta'] = (90.- spt_100d['DEC'].values)/180*np.pi
spt_100d['phi']   = spt_100d['RA'].values/180*np.pi

spt_100d['SNR']  = spt_100d['xi']
spt_100d['name'] = spt_100d['SPT_ID']


'''
--------------------------------
ACT DR5
--------------------------------
'''


act = pf.open('/project2/chihway/dhayaa/ACT_maps/DR5_cluster-catalog_v1.0.fits')
act = Table(act[1].data).to_pandas()

act['REDSHIFT'] = act['redshift']

#Throw out all z < 0 objects
act = act.loc[act.REDSHIFT.values > 0]

act['R500c'] = mass_so.M_to_R(10**14*act.M500c.values*cosmo.h,
                                   act.REDSHIFT.values, '500c')/cosmo.h*1e-3

act['M500c'] = np.log10(10**14*act.M500cCal)
act['M_ERR'] = (act.M500c_errPlus.values + act.M500c_errMinus.values)/2*10**14

act['Theta500c'] = act['R500c']/cosmo.angular_diameter_distance(act.REDSHIFT.values).value

#Get theta and phi
act['theta'] = (90.- act['decDeg'].values)/180*np.pi
act['phi']   = act['RADeg'].values/180*np.pi

act['RA']  = act['RADeg']
act['DEC'] = act['decDeg']


'''
--------------------------------
DES
--------------------------------
'''


tmp = pf.open('/project2/chihway/dhayaa/DES_Clusters/'
              'y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl50_catalog.fit')[1].data

des = pd.DataFrame()

des['name'] = tmp['MEM_MATCH_ID'].byteswap().newbyteorder()
des['RA']   = tmp['RA'].byteswap().newbyteorder()
des['DEC']  = tmp['DEC'].byteswap().newbyteorder()

des['LAMBDA_CHISQ']   = tmp['LAMBDA_CHISQ'].byteswap().newbyteorder()
des['LAMBDA_CHISQ_E'] = tmp['LAMBDA_CHISQ_E'].byteswap().newbyteorder()

des['LAMBDA_RAW'] = des['LAMBDA_CHISQ']/tmp['SCALEVAL'].byteswap().newbyteorder()
des['SNR']        = des['LAMBDA_CHISQ']/des['LAMBDA_CHISQ_E']
des['REDSHIFT']   = tmp['Z_LAMBDA'].byteswap().newbyteorder()

des['phi']   = des['RA'] * np.pi/180
des['theta'] = (90 - des['DEC']) * np.pi/180

#Do some preliminary quality cuts
des = des.loc[(des.REDSHIFT.values > 0.1) & (des.REDSHIFT.values < 0.8) &
              (des.LAMBDA_RAW.values > 10)]

#Implement McClintock mass-richness relation
#Use montecarlo to propogate uncertainties in
#a (semi) meaningful way
M0, dM0 = 14.489, 0.02
F, dF   = 1.356, 0.05
G, dG   = -0.3, 0.3

N_MonteCarlo = 1000

M0 = np.random.normal(M0, dM0, N_MonteCarlo)
F  = np.random.normal(F,  dF,  N_MonteCarlo)
G  = np.random.normal(G,  dG,  N_MonteCarlo)

M200m = 10**M0 * (des.LAMBDA_CHISQ.values[:, None]/40)**F * ((1 + des.REDSHIFT.values[:, None])/(1 + 0.35))**G

des['M200m'] = np.mean(np.log10(M200m), axis = 1)

des['R200m'] = mass_so.M_to_R(10**des.M200m.values*cosmo.h,
                              des.REDSHIFT.values, '200m')/cosmo.h*1e-3

for data in [planck, spt, spt_ecs, spt_100d, act]:


    R200m = np.zeros(len(data))
    M200m = np.zeros(len(data))

    def change_mass_def(i):
        
        cosmology.setCosmology('planck15')

        R200m = mass_adv.changeMassDefinitionCModel(10**data.M500c.values[i]*cosmo.h,
                                                       data.REDSHIFT.values[i],
                                                       '500c', '200m')[1]*1e-3/cosmo.h
        M200m = mass_so.R_to_M(R200m*cosmo.h*1e3, data.REDSHIFT.values[i], '200m')

        return i, R200m, M200m

    jobs = [joblib.delayed(change_mass_def)(i) for i in range(len(data))]

    with joblib.parallel_backend("loky"):
        outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)

    for result in outputs:
        if result is not None:

            i = result[0]
            R200m[i] = result[1]
            M200m[i] = result[2]

    data['R200m'] = R200m
    data['M200m'] = np.log10(M200m/cosmo.h)

    print("Done")
    
    
'''
For DES, we need to get M500c
'''


R500c = np.zeros(len(des))
M500c = np.zeros(len(des))

def change_mass_def(i):

    cosmology.setCosmology('planck15')

    R500c = mass_adv.changeMassDefinitionCModel(10**des.M200m.values[i]*cosmo.h,
                                                   des.REDSHIFT.values[i],
                                                   '200m', '500c')[1]*1e-3/cosmo.h
    M500c = mass_so.R_to_M(R500c*cosmo.h*1e3, des.REDSHIFT.values[i], '500c')

    return i, R500c, M500c

jobs = [joblib.delayed(change_mass_def)(i) for i in range(len(des))]

with joblib.parallel_backend("loky"):
    outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)

for result in outputs:
    if result is not None:

        i = result[0]
        R500c[i] = result[1]
        M500c[i] = result[2]

des['R500c'] = R500c
des['M500c'] = np.log10(M500c/cosmo.h)

print("Done")
    
    
'''
Get Peak heights
'''

for data in [planck, spt, spt_ecs, spt_100d, act, des]:


    v200m = np.zeros(len(data))

    def get_peak_height(i):
        
        cosmology.setCosmology('planck15')

        return i, peaks.peakHeight(10**data.M200m.values[i]*cosmo.h, data.REDSHIFT.values[i])

    jobs = [joblib.delayed(get_peak_height)(i) for i in range(len(data))]

    with joblib.parallel_backend("loky"):
        outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)

    for result in outputs:
        if result is not None:

            i = result[0]
            v200m[i] = result[1]

    data['v200m'] = v200m

path = '/home/dhayaa/Desktop/Accretion_Shock_Project'

planck.to_hdf(path + '/Cluster_Catalog.hdf5',   key = 'Planck', mode = 'w')
spt.to_hdf(path + '/Cluster_Catalog.hdf5',      key = 'SPT')
spt_ecs.to_hdf(path + '/Cluster_Catalog.hdf5',  key = 'SPT_ECS')
spt_100d.to_hdf(path + '/Cluster_Catalog.hdf5', key = 'SPT_100d')
act.to_hdf(path + '/Cluster_Catalog.hdf5',      key = 'ACT')
des.to_hdf(path + '/Cluster_Catalog.hdf5',      key = 'DES')
