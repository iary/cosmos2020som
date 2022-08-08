#TRAINING WITH COLORS + I-BAND. CUT IN MASS, Z and K-BAND.
#LABELLING WITH IR SAMPLE CUT IN MASS, Z

### LOADING MODULES ###

#standard modules
import numpy as np
import os,sys
#astropy
from astropy.io import ascii,fits
from astropy import table,stats
from astropy.coordinates import SkyCoord,match_coordinates_sky
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
#matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
#SOMPY
from sompy import sompy,umatrix
#our own functions
import som_utils
import som_predict
from som_io import OutputToHDF5
#miscellaneous
from itertools import combinations



### SETTING PARAMETERS ###

vname = 'classic' 
flux = False   # input catalog is in flux or mag?
FOLDOUT = './'  # folder to store output files
# cosmology used for absolute mag, M* estimates, etc
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# SOM parameters
# reference band for scaling, and corresponding zero point
refbd = "HSC_i_MAG"
reflev = 22.5 
zupp = 1.8  # upper limit in redshift
# rectangular grid dimensions
dsz0 = 80; dsz1 = 80




### INPUT DATASET ###

# COSMOS2020 photometric and SED fitting catalog
data0 = table.Table.read("/Users/iary/OneDrive - Københavns Universitet/catalogs/cosmos2020_classic_v1.8.1.fits") 

# Filter names in the same order as in LePhare
filtname = ['CFHT_ustar', 'CFHT_u', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'HSC_y', 'VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_Ks', 'SC_IB427', 'SC_IB464', 'SC_IA484', 'SC_IB505', 'SC_IA527', 'SC_IB574', 'SC_IA624', 'SC_IA679', 'SC_IB709', 'SC_IA738', 'SC_IA767', 'SC_IB827', 'SC_NB711', 'SC_NB816', 'VISTA_NB118','SC_B','SC_V','SC_R','SC_I','SC_Z','IRAC_CH1', 'IRAC_CH2','GALEX_FUV','GALEX_NUV']

# SFG/QG separation: NUVrK, as in Ilbert+15 
color1 = data0['HSC_r_ABSMAG'].data - data0['VISTA_Ks_ABSMAG'].data
color2 = data0['GALEX_NUV_ABSMAG'].data - data0['HSC_r_ABSMAG'].data 
corr = -0.17*(cosmo.age(data0['lp_zBEST']) - cosmo.age(2.)).to(u.Gyr).value
sfg = (color2+corr<2.6) | (color2+corr<2*color1+1.7)
data0['SFG'] = list(map(int,sfg))  # convert True/False into 0/1
  
# we use a sub-set of filters:
filtused = ['CFHT_ustar', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'HSC_y', 
            'VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_Ks',
            'IRAC_CH1','IRAC_CH2']   # 'CFHT_u' is not in SXDF 
            
# list all possible colors
#! col_list = list(combinations(filtused, 2))
# but we use only pair-wise colors
col_list = [(filtused[i],filtused[i+1]) for i in range(len(filtused)-1)] 
  
# Possible selections to clean the sample
sel1 = data0['SFG']==1. #Star forming galaxies 
sel2 = (data0['lp_zBEST']>0.01) & (data0['lp_zBEST']<zupp) #Redshift cut
sel3 = (data0['VISTA_Ks_MAG']>17.) & (data0['VISTA_Ks_MAG']<24.8) #Cut in K band
sel4 = data0['lp_mass_med']>8.5 #Cut in stellar mass
# We implement only the followin selection cuts
data = data0[sel2 & sel3 & sel4]         
# Some columns need to be renamed
data.rename_column("lp_mass_med","M_SED")
data.rename_column("lp_SFR_med","SFR_SED")
data.rename_column("lp_zBEST","z_SED")
data.rename_column("ALPHA_J2000","RA")
data.rename_column("DELTA_J2000","DEC")

# Replace non-detection with model mag
for f in filtname:
    nd = (data[f+'_MAG']<0.) | (data[f+'_MAG']>30.)
    data[f+'_MAG'][nd] = data[f+'_MODMAG'][nd]

# Input color matrix for building the SOM
datin = som_utils.sominput(data,col_list,flux=flux,cname='{}_MAG',verbose=False)
# Quality selection for colors
for i in range(datin.shape[1]):
    good1 = np.isfinite(datin[:,i]) # in case there were NaN in the catalog 
    good2 = (datin[:,i]>-10.) & (datin[:,i]<10.)  # extreme colors
    good1 &= good2
    datin = datin[good1]
    data = data[good1]
     
#Add i-band magnitudes as an extra dimension
datin = np.c_[datin, data[refbd].data-reflev] 

print("We work with the color matrix with shape",datin.shape)
ngalt = datin.shape[0]  # n of galaxies used to rain the SOM



for r in range(10):
    version = '{}_v4cRR{}'.format(vname,r)

    # Load the IR catalog (MIPS detections)
    IdIR, zIR, raIR, decIR, sfrIR = np.loadtxt("SFR_All.dat", dtype="float", usecols=(0,1,2,3,6), unpack=True)
    det = (zIR>0.01) & (zIR<zupp)
    raIR = raIR[det]; decIR = decIR[det]; zIR = zIR[det]; sfrIR = sfrIR[det]; IdIR = IdIR[det]
    # Match IR sample to C2020 
    coorIR = SkyCoord(ra = raIR*u.degree, dec = decIR*u.degree)
    coorC20 = SkyCoord(ra = data['RA'].data*u.degree, dec = data['DEC'].data*u.degree)
    idx, d2d, _ = match_coordinates_sky(coorIR,coorC20)
    sep_limit = d2d < 1.*u.arcsec
    idx = idx[sep_limit]
    idy = sep_limit
    # Keep IR galaxies with logM*>10
    mass_cut = data['M_SED'][idx].data > 10.
    ir = np.zeros(datin.shape[0], dtype=bool) 
    ir[idx] = True #index in the COSMOS catalogue that corresponds to the IR galaxies 
    irv = np.zeros(datin.shape[0], dtype=bool) 
    irc = np.zeros(datin.shape[0], dtype=bool) 
    irv[idx[np.random.randint(0,high=len(idx),size=int(np.round(len(idx)*0.15)))]] = True 
    irc[idx[mass_cut]] = True 
    irc *= ~irv 
    # Add SFR_IR in the table
    data['SFR_IR'] = -999.
    data['SFR_IR'][ir] = sfrIR[idy] 
    data['IR_CBR'] = 0
    data['IR_CBR'][irc] = 1 
    data['IR_VLD'] = 0
    data['IR_VLD'][irv] = 1
    print("---")
    print(np.count_nonzero(ir))
    print(np.count_nonzero(irv))
    print(np.count_nonzero(irc))
    print("validation IR gal with logM>10:",np.count_nonzero((irv)&(data['M_SED']>10)))
    
    
    ### BUILD THE SOM WITH C2020 GALAXIES ###
    
    # Training
    somsp = sompy.SOMFactory.build(datin[irc], mapsize=(dsz0,dsz1), mapshape='planar',lattice='rect',initialization='pca')
    somsp.train(n_job=4,shared_memory='no')
    # extract coordinates X and Y on the SOM grid, and BMU
    ac = somsp.bmu_ind_to_xy(somsp.project_data(datin))
    data["SOM_X"] = ac.T[0]; data["SOM_Y"] = ac.T[1]; data["SOM_BMU"] = ac.T[2]
    # QF, i.e. fraction of quiescent gal per cell
    cellcnt = np.zeros([dsz0,dsz1])
    cellsfg = cellcnt.copy()
    for i in range(ngalt):
        cellcnt[ac[i][0],ac[i][1]] += 1
        cellsfg[ac[i][0],ac[i][1]] += data['SFG'][i]
    cellQF = 1.-cellsfg/cellcnt # this will be NaN in the empty cells 
    # each galaxy in the data table gets the QF of its cell
    data['QF_CELL'] = som_predict.predict_from_bmu(somsp, cellQF, data["SOM_BMU"].data, use_bmu=True, log=False, scale=None, fill_nan=False) 
    # Distance between a galaxy and its BMU weight
    dist, _ = somsp.find_k_nodes(datin,k=1)
    data['DIST'] = dist[:,0]
    
    
    
    ###  LABELLING  ###
    
    
    # Redshift and Stellar mass with the full training sample
    
    # i-band mag offsets
    del_i = data[refbd]-reflev
    # Redshift estimates 
    cellmed_z,cellvar_z = som_utils.paint_cell(somsp,data['SOM_BMU'].data,data['z_SED'].data)
    data['z_SOM'] = som_predict.predict_from_nn(somsp, cellmed_z, datin, log=False, scale=None, nn=5)
    # Stellar mass estimates
    mass_nrm = data['M_SED'] + 0.4*del_i # renormalized stellar mass
    cellmed_mass,cellvar_mass = som_utils.paint_cell(somsp,data['SOM_BMU'].data,mass_nrm)
    data['M_SOM'] = som_predict.predict_from_nn(somsp, cellmed_mass, datin, log=True, scale=del_i, nn=5)
     
    
    #Project IR/non-IR galaxies on trained SOM
    
    sfrIR_nrm = data['SFR_IR'][irc]+ 0.4*del_i[irc] #rescaling the SFR
    cellmed_sfr,cellvar_sfr = som_utils.paint_cell(somsp,data['SOM_BMU'][irc].data,sfrIR_nrm)
    data['SFR_SOM'] = som_predict.predict_from_nn(somsp, cellmed_sfr, datin, log=True, scale=del_i, nn=5)
    
    #Save the SOM:
    
    data.write(FOLDOUT+"som2020_cosmos2020_{}.dat".format(version),include_names=["ID","RA","DEC","SOM_X","SOM_Y","SOM_BMU","DIST","z_SOM", "M_SOM", "SFR_SOM", "z_SED","M_SED", "SFR_SED", "SFR_IR", "QF_CELL", "SFG","IR_CBR","IR_VLD"],format='ascii.commented_header',overwrite=True)
    # formats={}
    
   
    print(r,"th realization completed")


