import numpy as np
import xarray as xr
import cftime
from dataloader import *
import warnings
from scipy import integrate, signal

from haversine import haversine_vector, Unit

def linear_psm(c, prior, pp_y_all):
    """
    Linear regression psm. The regression parameters are provided in pp_y_all.
    One can also define a global slope and intercept in cfg['linear']['uniform']
    The proxy error is set in the beginning when the proxies are loaded.
    """
    def linear_reg(x,a,b):
        """
        Input:
            x: (time,site) model time series for each proxy site
            a: (site) linear regression slope for each site
            b: (site) linear regresssion intercept for each site
        """
    return x*a +b 
    
    #get names of regression parameters
    var=c.linear['variable']
    var_type=c.linear['type']
    var_time=c.linear['anom_time']
    slope=c.linear['slope']
    intercept=c.linear['intercept']
    uniform=c.linear['uniform']
        
    #get obs from model for variable
    lats=pp_y_all.lat.values
    lons=pp_y_all.lon.values

    HXfull=psm_pseudoproxy.obs_from_model(prior[var],lats,lons,interpol=c.interpol)
    #eventually compute anomaly
    if var_type=='anom':
        if var_time is not None:
            HXfull=HXfull-HXfull.sel(time=slice(var_time[0],var_time[1])).mean('time',skipna=True)
        else:
            HXfull=HXfull-HXfull.mean('time',skipna=True)
    
    #apply function
    if uniform is not None:
        HXfull=linear_reg(HXfull,pp_y_all[slope],pp_y_all[intercept])  
    else:
        HXfull=HXfull*uniform[0]+uniform[1]
    
    return HXfull

def infilt_weighting(d18,pr,evap,slice_unfinished=True,check_nan=False):
    """
    Takes yearly mean of d18O values taking into account infiltration (pr-evap)
    Input needs to be monthly xarrays. The evaporation values are 'positive'.
 
    Use pr and evap that has been loaded via load_prior(path,var), because there correct conversion to mm/month has already
    been done (Including correct month length)

    Slice unfinished: We do not use the first and/or last model year if it does not start at least from February/ends in November.
    """
    #Check equal dimensions and units between pr and evap, same time_series
    assert pr.dims==evap.dims==d18.dims, "Input xarrays have wrong dimensions"
    assert (pr.time==evap.time).all() and (evap.time==d18.time).all(), "Input xarrays have unequal timeseries"

    d18=d18.copy()
    pr=pr.copy()
    evap=evap.copy()

    #eventually adjust evap units
    ev_u=evap.attrs['units']
    pr_u=pr.attrs['units']

    if pr_u=='m/s':
        #assuming 30 days per month convert to mm/month
        pr=pr*1000*60*60*24*30
        pr_u='mm/month'

    if (ev_u != pr_u):
        #unequal dimensions
        if ev_u=='mm/month' and pr_u=='kg/m^2s':
            pr=kg_to_mm(pr)
            #convert pr to mm/month
        elif ev_u=='kg/m^2s' and pr_u=='mm/month':
            #convert evap to mm/month
            evap=kg_to_mm(evap)
        else:
            print('evaporation and precipitation units unknown, conversion not possible')

    #make evap values positive for later subtraction
    #evap=np.abs(evap)

    #set nan values to zero in all arrays (kind of a mask)
    cond=(d18.isnull() | pr.isnull() | evap.isnull())
    ones = xr.where(cond, 0.0, 1.0)
    
    d18 = d18 * ones
    pr  = pr * ones
    evap = evap * ones
    
    inf=pr- evap
    
    #just keep inf where inf >0
    inf=xr.where(inf<0,0,inf)
    
    #weighted d18 year mean
    # Calculate the weights
    #wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum('time')
    wgts = inf.groupby("time.year") / inf.groupby("time.year").sum('time')
    
    # Calculate the numerator, resample instead of group by to keep time dimension correctly
    inf_weighted = (wgts*d18).resample(time='AS').sum(dim='time')
    
    if slice_unfinished:
        #check beginning/end, full year?    
        xarray_time_copy=d18.time
        fm=xarray_time_copy.time.dt.month[0] #first month
        lm=xarray_time_copy.time.dt.month[-1] #last month
        if fm!=1 and fm!=2:
            inf_weighted=inf_weighted.isel(time=slice(1,None))
        if lm!=12 and lm!=11:
            inf_weighted=inf_weighted.isel(time=slice(None,-1))
   
    #####FALL-BACK MECHANISM:
    #SOME VALUES ARE SET TO ZERO DUE TO NO-AVAILABLE RECORDS
    # 0. TRY PRECIPITATION WEIGHTING
    
    pr=prec_weighting(d18,pr,slice_unfinished=True)
    inf_weighted=xr.where(inf_weighted==0,pr,inf_weighted)
    
    # 1. RECUR TO UNWEIGHTED D18=0 where inf_weighted is 0.
    
    d18_unweighted=annual_mean(d18)
    inf_weighted=xr.where(inf_weighted==0,d18_unweighted,inf_weighted)
    
    #check nan option due to ihadcm3. the d18_unweighted will introduce nans
    if check_nan:
        print('Checking prior for nans')
        for t in inf_weighted.time:
            x=inf_weighted.sel(time=t)
            nans=np.count_nonzero(np.isnan(x))
            if nans>0:
                print('Dropped year', t.values, 'due to nans')
                inf_weighted=inf_weighted.where(inf_weighted.time!=t, drop=True)
    
    return inf_weighted

def kg_to_mm(inp):
    #convert kg/m^2s to mm/month
    #kg/m^2s -> 10^3 m^3/m^2/s -> 10^3*10^(-3)mm/month/(60*60*24*30)-> mm/month*(60*60*24*30)
    #assume 30 days in month
    return inp*(60*60*24*30)

def mm_to_kg(inp):
    #convert mm/month to kg/m^2s
    #assume 30 days in month
    return inp/(60*60*24*30)

def lhf_to_evap(lhf,temp):
    #divide latent heat flux by latent heat of vaporization
    #e.g. https://rdrr.io/cran/bigleaf/man/LE.to.ET.html
    #returns et in kg m-2 s-1
    lam=(2.501-0.00237*(temp-273.15))*10**6
    return lhf/lam


### PDB conversion formulas

def pdb_friedman():
    #returns smow delta18O from pdb-value
    #following Friedman, O'Neil 1977
    #return 1.03086*delta_pdb + 30.86
    
    #just return the pre-factor, because that's the one number needed
    return 1.03086#*delta_pdb + 30.86

def pdb_coplen():
    #returns smow delta18O from pdb-value
    #following Coplen 1983
    #used in Comas-Bru 2019
    #return 1.03091 * delta_pdb + 30.91
    #just return the pre-factor, because that's the one number needed
    return 1.03091 #*delta_pdb + 30.91


## Fractionation formulas
#Normally given as 1000*ln(alpha)= ... , here formulated as alpha=...

### Calcite conversion formulas
#all found in Tremaine 2011

def alpha_calc_trem(T):
    return np.exp((16.1*10**3/T - 24.6)/1000)

def alpha_calc_fried(T):
    return np.exp((2.78*10**6/(T**2)-2.89)/1000)

def alpha_calc_kim(T):
    return np.exp((18.03*10**3/T-32.17)/1000)

### Aragonite conversion formulas
#as formulated in Lachniet 2015

def alpha_arag_grossman(T):
    #used in Comas-Bru 2019
    return np.exp((18.34*10**3/T - 31.954)/1000)

def alpha_arag_patterson(T):
    return np.exp((18.56*10**3/T - 33.49)/1000)

def alpha_arag_thorrold(T):
    return np.exp((18.56*10**3/T - 32.54)/1000)

def alpha_arag_kim(T):
    return np.exp((17.88*10**3/T - 30.76)/1000)

def frac(d18,T,pdb_conversion,exp_alpha):
    #calculates frac for numpy arrays/scalars
    
    #d18O: in permil
    #T: in Kelvin
    #pdb_conversion: formula
    return ((d18+1000)/pdb_conversion())*(exp_alpha(T))-1000

def prec_weighting(d18,pr,slice_unfinished=False,check_nan=False):
    """
    Takes yearly mean of d18O values taking into account precipipitation
    (old version had month correction, was wrong, do precipitation weighting correctly!)
    
    Input needs to be monthly xarrays.

    pr,d18: xarray DataArrays (not Dataset!, for Dataset you need to explictely slice)

    
    """
    #Check equal dimensions and units between pr and evap, same time_series
    assert pr.dims==d18.dims, "Input xarrays have wrong dimensions"
     
    #set nan values to zero in all arrays (kind of a mask)
    cond=(d18.isnull() | pr.isnull())
    ones = xr.where(cond, 0.0, 1.0)
    
    d18 = d18 * ones
    pr  = pr * ones
    
    #weigh infiltration values by month length (automatically also weights d18O)
    #month_length = pr.time.dt.days_in_month

    #mask relevant month lengths (has shape time*lat*lon)
    #month_length = month_length * ones

    #weighted d18 year mean
    # Calculate the weights
    #wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum('time')
    wgts = pr.groupby("time.year") / pr.groupby("time.year").sum('time')

    # Calculate the numerator, resample instead of group by to keep time dimension correctly
    pr_weighted = (wgts*d18).resample(time='AS').sum(dim='time')

    # Calculate the denominator
    #denom=(pr*wgts).resample(time='AS').sum(dim='time')

    #prw=pr_weighted/denom
    
    #if slice_unfinished:
    #check beginning/end, full year?    
    xarray_time_copy=d18.time
    fm=xarray_time_copy.time.dt.month[0] #first month
    lm=xarray_time_copy.time.dt.month[-1] #last month
    if fm!=1 and fm!=2:
        pr_weighted=pr_weighted.isel(time=slice(1,None))
    if lm!=12 and lm!=11:
        pr_weighted=pr_weighted.isel(time=slice(None,-1))

    #####FALL-BACK MECHANISM:
    #SOME VALUES ARE SET TO ZERO DUE TO NO-AVAILABLE RECORDS
    # 0. IF value==0 set replace it bei unweighted d18O
    
    d18_unweighted=annual_mean(d18)
    pr_weighted=xr.where(pr_weighted==0,d18_unweighted,pr_weighted)
    
    #check nan option due to ihadcm3. there should be nans for sure.
    if check_nan:
        print('Checking prior for nans')
        for t in pr_weighted.time:
            x=pr_weighted.sel(time=t)
            nans=np.count_nonzero(np.isnan(x))
            if nans>0:
                print('Dropped year', t.values, 'due to nans')
                pr_weighted=pr_weighted.where(pr_weighted.time!=t, drop=True)
    
    return pr_weighted

#direct interpolation from
def obs_from_model(prior,lat,lon,interpol=None):
    '''
    Compute observations from model simulation at proxy-locations. Can either take nearest location or compute a distance weighted mean. Also works on the orography for model-height calculation.
    
    Input:
        - Prior: (#Lat*#Lon,#nb_ensemble)),xarray that also contains coords (Also works on orography)
        - lat: Latitudes of proxy locations
        - lon: Longitude of proxy locations
        - interpol='dw' inverse distance weighting: https://en.wikipedia.org/wiki/Inverse_distance_weighting
        
    
    Output:
        - Prior at proxy_loc [#loc,#nb_ensemble]

    '''
    if interpol=='dw':
        #bring proy and model
        loc=np.array([[lat,lon] for lat,lon in zip(lat,lon)])
        stacked=prior.stack(z=('lat','lon'))
        if 'oro'!=prior.name:
            stacked=stacked.transpose('z','time')
        coords=[list(z) for z in stacked.z.values]
        
        dist=haversine_vector(loc,coords, Unit.KILOMETERS,comb=True)
        args=np.argsort(dist,axis=0)[:4].astype(int)
        dist2=np.take_along_axis(dist, args, axis=0)
        weight0=1/dist2

        weight=(weight0/(np.sum(weight0,axis=0)))
        pubu=stacked.values[args]
        if 'oro'==prior.name:
            prior_vals=np.sum(pubu*weight,axis=0)
            data_vars={prior.name:(['site'],prior_vals)}
            coords=dict(lon=(('site'),lon),lat=(('site'),lat))
            output = xr.Dataset(data_vars,coords,attrs=prior.attrs)[prior.name].T
        else:
            prior_vals=np.einsum('ij,ijm->jm',weight,pubu)
            data_vars={prior.name:(['site','time'],prior_vals)}

            coords=dict(lon=(('site'),lon),lat=(('site'),lat),time=(["time"],prior.time.data))
            output = xr.Dataset(data_vars,coords,attrs=prior.attrs)[prior.name].T
    else:
    
    	#sel method needs some tweaking, else we'd get all the lats/lons (eg 163x163 instead of 163.2)
    	#WARNING: Make sure that both prior and proxy site longitudes are in the same system! (with negativ lons it will give bs-results!
   	    output=prior.sel(lat=xr.DataArray(lat,dims='site'),lon=xr.DataArray(lon,dims='site'),method='nearest')
        

    return output

def pseudoproxies(prior_vals,SNR=5,noisetype='w',a=0.32,seed=None):
    """
    Perturbes timeseries for given Signal to noise variance ratio (SNR=ts_std/(ts-ts_perturbed).std)
    Some tweaking to get the correc axis and do the matrix calculation with np.einsum.
    
    Input:
        prior_vals: time series (locations,time)
        SNR: signal to noise variance ratio
        SNR= STD(True timeseries)/STD(Noise)
        
        
        noisetype: 'w'- white noise or 'r' - red noise
        Red Noise config:
            a: mean autocorellation for red noise 
                autocorelation, a=0.32 (as in Steiger 2014, dating back to Mann 2007)
                sn=sqrt(var(N)), varN follows from dividing var(X) by SNR, Steiger uses calibration timeseries for var(X)
    Output:
        pp_Y: perturbed timeseries (locations,time)
        pp_R: variance of error timeseries
        #I'm not 100% sure if I need to save the error-variance or the
        actual error value. I guess the variance is more correct.
    """
    shape=np.shape(prior_vals)
    ax_time=prior_vals.get_axis_num('time')
    ax_site=prior_vals.get_axis_num('site')
	
    if noisetype=='w':
        #compute errors fiven SNR
        rng=np.random.default_rng(seed)
        errs=rng.standard_normal(size=(shape[ax_time],shape[ax_site]))
        prior_std=prior_vals.std('time')
        std_scaled=prior_std/SNR
        pp_Y=prior_vals+np.einsum('ij,j->ij',errs,std_scaled)
        pp_R=((std_scaled)**2)
        #repeat in time to get timeseries of error in same shape
        pp_R=np.array([pp_R.data for _ in range(shape[ax_time])])
        #put back to dataarray
        pp_R=xr.DataArray(pp_R,dims=['time','site'])
        pp_R['time']=pp_Y.time
        
    #check that formula is correct
    #print('SNR',prior_std/(pp_Y-prior_vals).std(axis=-1))
    
    elif noisetype=='r':
        sn=np.sqrt(prior_vals.std('time')/SNR)
        rng=np.random.default_rng(seed)
        errs=rng.standard_normal(size=(shape[ax_time],shape[ax_site]))
        summand=np.einsum('ij,j->ij',errs,sn)*np.sqrt(1-a**2)
        summand=xr.DataArray(summand,dims=['time','site'])
        #timeseries length
        l=shape[ax_time]
        #copy prior_vals and replace values
        #remove first value (noise time series starts from second point)
        pp_Y=prior_vals.copy(deep=True)
        noise=pp_Y.copy(deep=True)
        noise.loc[dict(time=prior_vals.time.values[0])]=0
        for i,t in enumerate(prior_vals.time.values[1:]):
            noise.loc[dict(time=t)]=a*noise.isel(time=(i-1))+summand.isel(time=i)
            #you can not assign values with isel/sel method:
            #pp_Y.isel(time=i)=a*pp_Y.isel(time=(i-1))+summand.isel(time=i)
        pp_Y=(pp_Y+noise).isel(time=slice(2,None))
        pp_R=(noise**2).isel(time=slice(2,None))
        
    return pp_Y,pp_R
    
 ###### ICECORE SLIGHTLY PSM ADAPTED FROM S.DEE/F.ZHU. 
 # I use a  standard pressure value. The main thing about the psm is that it lowers the standard deviation/spread of the distribution (kind of a low pass filter)
 
 #Taken from F.Zhu:https://github.com/fzhu2e/LMRt/blob/master/LMRt/psm.py who itself adapted it from PRYSM

def diffusivity(rho, T=250, P=0.9, rho_d=822, b=1.3):
    '''
    DOCSTRING: Function 'diffusivity'
    Description: Calculates diffusivity (in m^2/s) as a function of density.
    Inputs:
    P: Ambient Pressure in Atm
    T: Temperature in K
    rho: density profile (kg/m^3)
    rho_d: 822 kg/m^2 [default], density at which ice becomes impermeable to diffusion
    Defaults are available for all but rho, so only one argument need be entered.
    Note values for diffusivity in air:
    D16 = 2.1e-5*(T/273.15)^1.94*1/P
    D18 = D16/1.0285
    D2 = D16/1.0251
    D17 = D16/((D16/D18)^0.518)
    Reference: Johnsen et al. (2000): Diffusion of Stable isotopes in polar firn and ice:
    the isotope effect in firn diffusion
    '''

    # Set Constants

    R = 8.314478                                                # Gas constant
    m = 18.02e-3                                                # molar weight of water (in kg)
    alpha18 = np.exp(11.839/T-28.224e-3)                    # ice-vapor fractionation for oxygen 18
    p = np.exp(9.5504+3.53*np.log(T)-5723.265/T-0.0073*T)     # saturation vapor pressure
    Po = 1.                                                 # reference pressure, atmospheres
    rho_i = 920.  # kg/m^3, density of solid ice

    # Set diffusivity in air (units of m^2/s)

    Da = 2.1e-5*np.power((T/273.15), 1.94)*(Po/P)
    Dai = Da/1.0285

    # Calculate Tortuosity

    invtau = np.zeros(len(rho))
    #  for i in range(len(rho)):
    #      if rho[i] <= rho_i/np.sqrt(b):
    #          # invtau[i]=1.-1.3*np.power((rho[i]/rho_d),2)
    #          invtau[i] = 1.-1.3*np.power((rho[i]/rho_i), 2)
    #      else:
    #          invtau[i] = 0.

    selector =rho <= rho_i/np.sqrt(b)
    invtau[selector] = 1.-1.3*(rho[selector]/rho_i)**2

    D = m*p*invtau*Dai*(1/rho-1/rho_d)/(R*T*alpha18)

    return D

def densification(Tavg, bdot, rhos, z):
    """
    print('Tavg')
    print(Tavg)
    print('bdot')
    print(bdot)
    print('rhos')
    print(rhos)
    print('z')
    print(z)
    """
    ''' Calculates steady state snow/firn depth density profiles using Herron-Langway type models.
    Args:
        Tavg: 10m temperature in celcius ## CELCIUS!  # fzhu: should be in K now
        bdot: accumulation rate in mwe/yr or (kg/m2/yr)
        rhos: surface density in kg/m3
        z: depth in true_metres
        model can be: {'HLJohnsen' 'HerronLangway' 'LiZwally' 'Helsen' 'NabarroHerring'}
        default is herronlangway. (The other models are tuned for non-stationary modelling (Read Arthern et al.2010 before applying in steady state).
    Returns:
        rho: density (kg/m3) for all z-values.
        zieq: ice equivalent depth for all z-values.
        t: age for all z-values (only taking densification into account.)
        Example usage:
        z=0:300
        [rho,zieq,t]=densitymodel(-31.5,177,340,z,'HerronLangway')
        plot(z,rho)
    References:
        Herron-Langway type models. (Arthern et al. 2010 formulation).
        Aslak Grinsted, University of Copenhagen 2010
        Adapted by Sylvia Dee, Brown University, 2017
        Optimized by Feng Zhu, University of Southern California, 2017
    '''
    rhoi = 920.
    rhoc = 550.
    rhow = 1000.
    rhos = 340.
    R = 8.314

    # Tavg=248.
    # bdot=0.1
    # Herron-Langway with Johnsen et al 2000 corrections.
    # Small corrections to HL model which are not in Arthern et al. 2010

    c0 = 0.85*11*(bdot/rhow)*np.exp(-10160./(R*Tavg))
    c1 = 1.15*575*np.sqrt(bdot/rhow)*np.exp(-21400./(R*Tavg))

    k0 = c0/bdot  # ~g4
    k1 = c1/bdot

    # critical depth at which rho=rhoc
    zc = (np.log(rhoc/(rhoi-rhoc))-np.log(rhos/(rhoi-rhos)))/(k0*rhoi)  # g6

    ix = z <= zc  # find the z's above and below zc
    upix = np.where(ix)  # indices above zc
    dnix = np.where(~ix)  # indices below zc

    q = np.zeros((z.shape))  # pre-allocate some space for q, rho
    rho = np.zeros((z.shape))

    # test to ensure that this will not blow up numerically if you have a very very long core.
    # manually set all super deep layers to solid ice (rhoi=920)
    NUM = k1*rhoi*(z-zc)+np.log(rhoc/(rhoi-rhoc))

    numerical = np.where(NUM <= 100.0)
    blowup = np.where(NUM > 100.0)
    #print(k1,rhoi,zc,rhoc)
    
    warnings.simplefilter("ignore",category=RuntimeWarning) 
    
    q[dnix] = np.exp(k1*rhoi*(z[dnix]-zc)+np.log(rhoc/(rhoi-rhoc)))  # g7
    q[upix] = np.exp(k0*rhoi*z[upix]+np.log(rhos/(rhoi-rhos)))  # g7

    rho[numerical] = q[numerical]*rhoi/(1+q[numerical])  # [g8] modified by fzhu to fix inconsistency of array size
    rho[blowup] = rhoi

    # only calculate this if you want zieq
    tc = (np.log(rhoi-rhos)-np.log(rhoi-rhoc))/c0  # age at rho=rhoc [g17]
    t = np.zeros((z.shape))  # pre allocate a vector for age as a function of z
    t[upix] = (np.log(rhoi-rhos)-np.log(rhoi-rho[upix]))/c0  # [g16] above zc
    t[dnix] = (np.log(rhoi-rhoc)-np.log(rhoi+0.0001-rho[dnix]))/c1 + tc  # [g16] below zc
    tdiff = np.diff(t)

    # make sure time keeps increasing even after we reach the critical depth.
    if np.any(tdiff == 0.00):
        inflection = np.where(tdiff == 0.0)
        lineardepth_change = t[inflection][0]

        for i in range(len(t)):
            if t[i] > lineardepth_change:
                t[i] = t[i-1] + 1e-5

    zieq = t*bdot/rhoi  # [g15]

    return rho, zieq, t


def ice_archive(d18Oice, pr_ann, tas_ann, psl_ann, nproc=8):
    ''' Accounts for diffusion and compaction in the firn.
    Args:
        d18Oice (1d array: year in int): annualizd d18O of ice [permil]
        pr_ann (1d array: year in int): precipitation rate [kg m-2 s-1] ## should be in mm/month now
        tas_ann (1d array: year in int): annualizd atomspheric temerature [K]
        psl_ann (1d array: year in int): annualizd sea level pressure [Pa]
        nproc (int): the number of processes for multiprocessing
    Returns:
        ice_diffused (1d array: year in int): archived ice d18O [permil]
    '''
    # ======================================================================
    # A.0: Initialization
    # ======================================================================

    yr2sec_factor = 3600*24*365.25
    # accumulation rate [m/yr], but input is in mm/month
    accum=pr_ann*12/1000

    # depth horizons (accumulation per year corresponding to depth moving down-core)
    bdown = accum[::-1].values
    bmean = np.mean(bdown)
    depth = np.sum(bdown)
    depth_horizons = np.cumsum(bdown)
    dz = np.min(depth_horizons)/10.  # step in depth [m]
    
    #print(dz)
    #print(depth)
    
    Tmean = np.mean(tas_ann).values  # unit in [K]
    Pmean = np.mean(psl_ann).values*9.8692e-6  # unit in [Atm]

    # contants
    rho_s = 300.  # kg/m^3, surface density
    rho_d = 822.  # kg/m^2, density at which ice becomes impermeable to diffusion
    rho_i = 920.  # kg/m^3, density of solid ice

    # ======================================================================
    # A.1: Compaction Model
    # ======================================================================
    z = np.arange(0, depth, dz) + dz  # linear depth scale
    #print('start',z[0])
    #print('end',z[-1])
    #print('dz',dz)
    
    # set density profile by calling densification function
    #print(Tmean, bmean, rho_s, z)
    rho, zieq, t = densification(Tmean, bmean, rho_s, z)

    rho = rho[:len(z)]  # cutoff the end
    time_d = np.cumsum(dz/bmean*rho/rho_i)
    ts = time_d*yr2sec_factor  # convert time in years to ts in seconds

    # integrate diffusivity along the density gradient to obtain diffusion length
    D = diffusivity(rho, Tmean, Pmean, rho_d, bmean)

    D = D[:-1]
    rho = rho[:-1]
    diffs = np.diff(z)/np.diff(time_d)
    diffs = diffs[:-1]

    # Integration using the trapezoidal method

    # IMPORTANT: once the ice reaches crtiical density (solid ice), there will no longer
    # be any diffusion. There is also numerical instability at that point. Set Sigma=1E-13 for all
    # points below that threshold.

    # Set to 915 to be safe.
    solidice = np.where(rho >= rho_d-5.0)
    diffusion = np.where(rho < rho_d-5.0)

    dt = np.diff(ts)
    sigma_sqrd_dummy = 2*np.power(rho, 2)*dt*D
    sigma_sqrd = integrate.cumtrapz(sigma_sqrd_dummy)
    diffusion_array = diffusion[0]
    diffusion_array = diffusion_array[diffusion_array < len(sigma_sqrd)]  # fzhu: to avoid the boundary index error
    diffusion = np.array(diffusion_array)

    #  rho=rho[0:-1] # modified by fzhu to fix inconsistency of array size
    #  sigma=np.zeros((len(rho)+1)) # modified by fzhu to fix inconsistency of array size
    sigma = np.zeros((len(rho)))
    sigma[diffusion] = np.sqrt(1/np.power(rho[diffusion],2)*sigma_sqrd[diffusion]) # modified by fzhu to fix inconsistency of array size
    #sigma[solidice]=np.nanmax(sigma) #max diffusion length in base of core // set in a better way. max(sigma)
    sigma[solidice] = sigma[diffusion][-1]
    sigma = sigma[:-1]

    # ======================================================================
    # A.2. Diffusion Profile
    # ======================================================================
    # Load water isotope series
    del18 = np.flipud(d18Oice)  # NOTE YOU MIGHT NOT NEED FLIP UD here. Our data goes forward in time.
    #print(len(z),len(depth_horizons),len(del18))
    # interpolate over depths to get an array of dz values corresponding to isotope values for convolution/diffusion
    iso_interp = np.interp(z, depth_horizons, del18)

    # Return a warning if the kernel length is approaching 1/2 that of the timeseries.
    # This will result in spurious numerical effects.
    
    #this warning is misleading, nothing wrong with the time series, just take a good kernel from the beginning
    """
    zp = np.arange(-100, 100, dz)
    if (len(zp) >= 0.5*len(z)):
        print('z',len(z))
        print('zp',len(zp))
        print("Warning: convolution kernel length (zp) is approaching that of half the length of timeseries. Kernel being clipped.")
        bound = 0.20*len(z)*dz
        zp = np.arange(-bound, bound, dz)
    """
    bound = 0.20*len(z)*dz
    zp = np.arange(-bound, bound, dz)
    
    #  print('start for loop ...')
    #  start_time = time.time()

    rm = np.nanmean(iso_interp)
    cdel = iso_interp-rm

    diffused_final = np.zeros(len(iso_interp))
    if nproc == 1:
        for i in range(len(sigma)):
            sig = sigma[i]
            part1 = 1./(sig*np.sqrt(2.*np.pi))
            part2 = np.exp(-zp**2/(2*sig**2))
            G = part1*part2
            #  diffused = np.convolve(G, cdel, mode='same')*dz  # fzhu: this is way too slow
            diffused = signal.fftconvolve(cdel, G, mode='same')*dz  # put cdel in the front to keep the same length as before
            diffused += rm  # remove mean and then put back
            diffused_final[i] = diffused[i]

    else:
        #  print('Multiprocessing: nproc = {}'.format(nproc))

        def conv(sig, i):
            part1 = 1./(sig*np.sqrt(2.*np.pi))
            part2 = np.exp(-zp**2/(2*sig**2))
            G = part1*part2
            diffused = signal.fftconvolve(cdel, G, mode='same')*dz
            diffused += rm  # remove mean and then put back

            return diffused[i]
        
        res=conv(sigma,range(len(sigma)))
        #res = Pool(nproc).map(conv, sigma, range(len(sigma)))
        diffused_final[:len(res)] = np.array(res)

    #  print('for loop: {:0.2f} s'.format(time.time()-start_time))

    # take off the first few and last few points used in convolution
    diffused_timeseries = diffused_final[0:-3]

    # Now we need to pack our data back into single year data units based on the depths and year interpolated data
    final_iso = np.interp(depth_horizons, z[0:-3], diffused_timeseries)
    ice_diffused = final_iso

    return ice_diffused
 
def infilt_weighting_old(d18,pr,evap):
    """
    CODE UNUSED
    
    Takes yearly mean of d18O values taking into account infiltration (pr-evap)
    Input needs to be monthly xarrays.
    Units: True if possible convers
    
    """
    #Check equal dimensions and units between pr and evap, same time_series
    assert pr.dims==evap.dims==d18.dims, "Input xarrays have wrong dimensions"
    assert (pr.time==evap.time).all() and (evap.time==d18.time).all(), "Input xarrays have unequal timeseries"

    #(deep) copy input arrays to avoid memory problem
    d18=d18.copy()
    pr=pr.copy()
    evap=evap.copy()
    
    #eventually adjust evap units
    ev_u=evap.attrs['units']
    pr_u=pr.attrs['units']
    
    if pr_u=='m/s':
        #assuming 30 days per month convert to mm/month
        pr=pr*1000*60*60*24*30
        pr_u='mm/month'
    #print(pr_u)
    #print(ev_u)

    if (ev_u != pr_u):
        #unequal dimensions
        if ev_u=='mm/month' and pr_u=='kg/m^2s':
            pr=kg_to_mm(pr)
            #convert pr to mm/month
        elif ev_u=='kg/m^2s' and pr_u=='mm/month':
            #convert evap to mm/month
            evap=kg_to_mm(evap)
        else:
            print('evaporation and precipitation units unknown, conversion not possible')
    
    #make evap values positive for later subtraction
    evap=np.abs(evap)
    
    #set nan values to zero in all arrays (kind of a mask)
    for ar in [d18,pr,evap]:
        cond = ar.isnull()
        ones = xr.where(cond, 0.0, 1.0)
        ar = ar * ones

    #calculate infiltration=precipitation-evaporation
    inf=pr-evap
    #print(pr.time)
    #print(evap.time)
    #print(inf.time)
    #weigh infiltration values by month length (automatically also weights d18O)
    month_length = inf.time.dt.days_in_month

    #mask where infiltration values are zero
    cond = inf.isnull()
    ones = xr.where(cond, 0.0,1.0)
    #create mask with relevant month lengths (has shape time*lat*lon)
    month_length = month_length * ones

    #weighted d18 year mean
    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum('time')

    # Calculate the numerator, resample instead of group by to keep time dimension correctly
    inf_weighted = (inf * wgts*d18).resample(time='AS').sum(dim='time')

    # Calculate the denominator
    denom=(inf*wgts).resample(time='AS').sum(dim='time')

    infw=inf_weighted/denom
    
    infw=infw.rename('d18_inf')
    return infw
    
def psm_speleo(d18O,T,tau0=2.5,filt=True, frac=True):
    """
    UNUSED 
    
    Code following PRYSM Code, Sylvia Dee
    Convolutes  yearly infiltration weighted d18O timeseries with karst
    
    Input (Takes xarrays)
        d18O: infiltration weighted yearly mean d18O values per gridbox (e.g use function infiltration weight)
        T: yearly mean temperature for each gridpoint (in Kelvin!)
        tau: set to 2.5 (best value according to Janica)
        filt: if True use karst filter, else: no filter
        frac: if True use fractionation, else no fractionation
    
    Boundary effects in convolution:
    - Usual np.convolve uses some kind of zero padding at the border -> weird (could use mode valid, but would use values)
    - Alternative scipy.ndimage.convolve1d mode=nearest for instance, where first/last value repeated
            -> but the origin is shifted (Je ne comprends pas)
    - How made: padding beginning and end by hand, convolve, cut array
    - Easy fix: Only convolve deviations from the mean and add the mean (seems to be correct?)
    
    
    """
    #to be sure we don't affect the input arrays
    d18O=d18O.copy()
    T=T.copy()


    if filt==True:
        #set timeseries
        tau=np.arange(len(d18O.time))

        #Green's function
        #well-mixed model
        g = (1./tau0) * np.exp(-tau/tau0)

        #normalize tau (as we have yearly spaced values, we just sum up all values)
        g=g/np.sum(g)

        #convolve d18O with g
        #subtract mean
        mean=d18O.mean('time')
        
        #get time axis number
        ax=d18O.get_axis_num('time')
        #convolve padding first/last value (no-problem as g decreases very quickly anyqay)
        conv=np.apply_along_axis(lambda m: np.convolve(m, g), axis=ax, arr=(d18O-mean).values)[:len(d18O.time)]
        #exchange values in initial array
        d18O.values=conv
        d18O=d18O+mean
    
    if frac==True:
        #Fractionation (Friedman and O'Neill 1977
        d18Oc = (d18O + 1000)/1.03086*np.exp(2780/T**2 - 0.00289)-1000.0
    else:
        d18Oc=d18O

    d18Oc=d18Oc.rename('Karst filtered and fractionated d18O')
    
    return d18Oc
