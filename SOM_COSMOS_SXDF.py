#TRAINING WITH COLORS + I-BAND. CUT IN MASS, Z and K-BAND.
#LABELLING WITH IR SAMPLE CUT IN MASS, Z


# standard modules
import numpy as np
import os,sys
#astropy
from astropy.io import ascii,fits
from astropy import table
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
#SOMPY
from sompy import sompy,umatrix
#sklearn
from sklearn import metrics
#our own functions
import som_utils
import som_predict
from som_io import InputFromHDF5
from itertools import combinations

### SETTING PARAMETERS ###

vname = 'classic' 
version = '{}_v4d'.format(vname)
flux = False   # input catalog is in flux or mag?
FOLDOUT = './'  # folder to store output files
# cosmology used for absolute mag, M* estimates, etc
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# SOM parameters
# reference band for scaling, and corresponding zero point
refbd = "HSC_i_MAG"
reflev = 22.5 
# rectangular grid dimensions
dsz0 = 80; dsz1 = 80

#############################



# Input table 

data_sxdf = table.Table.read("/Users/iary/OneDrive - Københavns Universitet/catalogs/photoz_sxdf2018_v1.6.2_raw_classic.fits", format="fits")
filtused_sxdf = ['CFHT_u', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'HSC_y',
            'VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_Ks',
            'IRAC_CH1','IRAC_CH2']  
for f in filtused_sxdf:
    nd = (data_sxdf[f+'_MAG']<0.) | (data_sxdf[f+'_MAG']>28.)
    data_sxdf[f+'_MAG'][nd] = data_sxdf[f+'_MODMAG'][nd]
data_sxdf = data_sxdf[data_sxdf['HCSDR2_MASK']==1] # HSC-DR2 stellar mask

dat_cos = table.Table.read("som2020_cosmos2020_{}.fits".format(version)) 
  

# Select subsample 

sel1 = (data_sxdf['zBEST_M18']>0.01) & (data_sxdf['zBEST_M18']<1.8) #Redshift cut
sel2 = (data_sxdf['VISTA_Ks_MAG']>17.) & (data_sxdf['VISTA_Ks_MAG']<24.23) #Cut in K band
#sel3 = (data_sxdf['MASS_BEST']>8.) #mass cut
data_sxdf_inp = data_sxdf[sel1 & sel2]

     
# Input color matrix

# list all possible colors
#! col_list = list(combinations(filtused_sxdf, 2))
# but we use only pair-wise colors
col_list = [(filtused_sxdf[i],filtused_sxdf[i+1]) for i in range(len(filtused_sxdf)-1)] 
datin_sxdf_inp = som_utils.sominput(data_sxdf_inp,col_list,flux=flux,cname='{}_MAG',verbose=False)
for i in range(datin_sxdf_inp.shape[1]):
    good1 = np.isfinite(datin_sxdf_inp[:,i])
    good2 = (datin_sxdf_inp[:,i]>-20.) & (datin_sxdf_inp[:,i]<20.)
    good1 &= good2
    datin_sxdf_inp = datin_sxdf_inp[good1]
    data_sxdf_inp = data_sxdf_inp[good1]
# datin_sxdf = np.c_[datin_sxdf_inp, data_sxdf_inp[refbd].data-reflev]


### LOAD THE COSMOS SOM ###

HS = InputFromHDF5("som_80x80_{}.hdf5".format(version))
somsp = HS.createSOM()
somsp = HS()

 
### PROJECT THE SXDF CATALOG ON COSMOS SOM ###

data_sxdf_inp.rename_column("MASS_MED","M_SED")
data_sxdf_inp.rename_column("SFR_MED","SFR_SED")
data_sxdf_inp.rename_column("zBEST_M18","z_SED")
 
ac_sxdf = somsp.bmu_ind_to_xy(somsp.project_data(datin_sxdf_inp))
data_sxdf_inp["SOM_X"] = ac_sxdf.T[0]; data_sxdf_inp["SOM_Y"] = ac_sxdf.T[1]; data_sxdf_inp["SOM_BMU"] = ac_sxdf.T[2]
dst,_ = somsp.find_k_nodes(datin_sxdf_inp,k=50)
data_sxdf_inp["DIST"] = dst #[:,0]

# redshift, SFR, M* estimates
data_sxdf_inp["z_SOM"] = som_predict.predict_from_nn(somsp, somsp._dlabelz, datin_sxdf_inp, log=False, scale=None, nn=5)
del_i_IR = data_sxdf_inp['HSC_i_MAG'] - reflev
data_sxdf_inp["SFR_SOM"] = som_predict.predict_from_nn(somsp, somsp._dlabelsfr, datin_sxdf_inp, log=True, scale=del_i_IR,  nn=5)
data_sxdf_inp["M_SOM"] = som_predict.predict_from_nn(somsp, somsp._dlabelmass, datin_sxdf_inp, log=True, scale=del_i_IR,  nn=5)
data_sxdf_inp["QF_NN"] = som_predict.predict_from_nn(somsp, somsp._dlabelqf, datin_sxdf_inp, scale=None,  nn=5)

data_sxdf_out = table.join(data_sxdf_inp,table.unique(dat_cos["SOM_BMU","QF_CELL"]),keys="SOM_BMU",join_type='left')
 
# data_sxdf_out.write(FOLDOUT+"som2020_sxdf2018_{}.dat".format(version),include_names=["IDENT","RA","DEC","SOM_X","SOM_Y","SOM_BMU","DIST","z_SOM", "M_SOM", "SFR_SOM", "z_SED","M_SED", "SFR_SED", "SFR_IR", "QF_CELL", "SFG","IR_CBR","IR_VLD"],format='ascii.commented_header',overwrite=True)
data_sxdf_out.write(FOLDOUT+"som2020_sxdf2018_{}.fits".format(version),overwrite=True)
 
print("SXDF done")
