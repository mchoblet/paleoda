import numpy as np
import xarray as xr
import xskillscore as xss
import os
from haversine import haversine_vector, Unit
from joblib import Parallel, delayed
from scipy.stats import rankdata


#when possible evaluation function from the xskillscore package are used. Functions here provide a wrapper that

def Corr(true,recon): return xr.corr(true,recon,dim='time')

def p_value(true,recon):
    """
    p-value of correlation
    Takes Dataarrays
    """
    
    return xss.pearson_r_p_value(true,recon,dim='time',skipna=True)

def p_bootstrap(true,recon,size=100):
    #p values by bootstrapping, where one time series is shifted randomly against the truth (keeps autocorrelations
    #size: 100 how many times the experiment is repeated (that's also the number+1 of the ranks)
    #for the 95% confidence intervall choose the points where the rank is larger than 95.

    nums=np.random.randint(1,len(true.time),size=100)
    standard=xr.corr(true,recon,dim='time')
    c=[]
    for i,n in enumerate((nums)):
        corrs=xr.corr(recon.roll(time=n),true,dim='time')
        c.append(corrs.values)
    c.append(standard.values)

    from scipy.stats import rankdata
    #the last rank is the relevant one
    ranks=rankdata(c,axis=0)[-1]
    #print(np.shape(ranks))
    ranks_da=xr.DataArray(ranks,dims=('lat','lon'))
    return ranks_da

def eff_p_value(true,recon):
    """
    Effective p-value
    Takes Dataarrays
    """
    return xss.pearson_r_p_value(true,recon,dim='time',skipna=True)

def CE(true,recon):
    """
    works on xarrays
    Input:
        true: true(reference) time series/fields
        recond: reconstructed time series/fields
    
    Output:
        CE-field
    """
    
    assert true.shape==recon.shape, 'input timeseries do not have same shape'
    
    denom=np.square(true-recon).sum('time')
    nom=np.square(true-true.mean('time')).sum('time')
    CE=1-denom/nom

    return CE

def RMSPE(true,recon):
    """
    root mean square percentage error with respect to the true time-series
    Use preferably when you compare for different models which use different units
    mean Root mean square error 
    Input are x-arrays
    """
    assert true.shape==recon.shape, 'input timeseries do not have same shape'
    dif=true-recon
    eps=1e-10 #to avoid division by zero problem
    s=np.square(dif/(true+eps)).sum('time')
    rmspe = np.sqrt(s/(dif.time.shape[0]))
    return rmspe*100

def RMSE(true,recon):
    """
    root mean square error with respect to the true time-series
    mean Root mean square error 
    Input are x-arrays
    """
    assert true.shape==recon.shape, 'input timeseries do not have same shape'
    """
    dif=true-recon
    s=np.square(dif).sum('time')
    rmse = np.sqrt(s/(dif.time.shape[0]))
    """
    
    return xss.rmse(true,recon,dim='time',skipna=True)

def MAE(true,recon):
    """
    Mean absolute error
    """
    assert true.shape==recon.shape, 'input timeseries do not have same shape'
    
    #dif=true-recon
    #mae=abs(dif).sum('time')/(dif.time.shape[0])
    ma=xss.mae(true,recon,dim='time',skipna=True)
    return ma 

def MAPE(true,recon):
    #mean absolute percentage error
    assert true.shape==recon.shape
    ma=xss.mape(true,recon,dim='time',skipna=True)
    return ma

def RE(true,recon,prior):
    """
    Reduction of error (as in Bhend 2012) compared to uninformed prior mean
    prior-mean does not have the time dimension, xarray automatically does the broadcasting correctl
    """
    den=np.square(true-recon).sum('time')
    nom=np.square(prior.mean('time')-true).sum('time')
    return 1-den/nom

####Rank histograms:
#uses xss.rank_histogram(Y,HXf,dim='site',member_dim='time') directly in the wraper,
#defined here only for completeness. There is a non-xss solution at the end of this file.

def rank_histogram(Y,HXf): return xss.rank_histogram(Y,HXf,dim='site',member_dim='time')

def CD(fullfield,pp_y,pp_y_lat,pp_y_lon,nbins=100):
    """
    Update 16.09: Computing correlation distance
    
    Alternative I might implement: Covariance-distance calculation
        Scale each column (?) by the covariance of proxy estimates pp_y in order
    t   o mimic the kalman gain (except for the error term part). 
    
    Input:
        - Full field (prior or reconstruction)
        IMPORTANT: must contain a coordinate z=('lat','lon). (e.g. stacked during the wrapper)
        - pp_y: proxy estimate values ('time','site') (HXf)
        - pp_y_lat: latitudes of observations
        - pp_y_lon: longitudes of observations
        - nbins: number of bins for the binning calculations
        
    """
    
    #1. compute distances, initialize distance bins_
    #locations of observations
    loc=np.array([[lat,lon] for lat,lon in zip(pp_y_lat,pp_y_lon)])
    #take locations from prior (stacked before kalman filter with z=('lat','lon'))
    coords=[list(z) for z in fullfield.z.values]
    
    #cores=os.cpu_count()
    #Prooved to be the fastest, vectorizing this is actually slower (also mentioned here https://github.com/mapado/haversine)
    #dist=Parallel(n_jobs=cores)(delayed (haversine_vector)(l,coords, Unit.KILOMETERS,comb=True) for l in loc)
    #safer:
    dist=haversine_vector(loc,coords, Unit.KILOMETERS,comb=True)
    
    dist=np.array(dist).T.reshape(-1)
    
    
    #1.5 flatten arrays to 1-D vectors
    dist=np.squeeze(dist).reshape(-1)
    #maximum distance ~20000km, 100km spacing gives us 200 points for x-(distance)-axis
    bins=np.arange(0,20000,nbins)
    bin_a=np.digitize(dist, bins)-1

    
    #2. compute correlations (~10 seconds, chunked with dask else memory killed)
    data=fullfield.chunk({'z':250})
    
    #import pdb
    #pdb.set_trace()
    
    corr=xr.corr(data,pp_y,dim='time')
    #corr=xr.cov(data,pp_y,dim='time')
    vals=corr.values.reshape(-1)
    
    #var=np.var(pp_y,axis=0)
    #corr=corr/var
    
    #print('vals-shape',np.shape(vals))
    #3. assign correlations to bins (xr.groupby-solution-slower)
    
    #eventually some bins without a distance
    unique_bins=np.unique(bin_a)

    cor_mean=np.empty(len(unique_bins))
    cor_std=np.empty(len(unique_bins))

    for i, l in enumerate(unique_bins):
        cor_mean[i]=np.mean(vals[np.argwhere(bin_a==l)])
        cor_std[i]=np.std(vals[np.argwhere(bin_a==l)])
    #bins[unique_bins] are the distances used for cor_mean and cor_std
    return cor_mean,cor_std,bins[unique_bins]




##CURRENTLY UNUSED

def rankhisto(obs,ens):
    """
    Input: obs and ens are xarrays
    Obs has the dimension ensemblemembers
    
    To-Do: Vectorize (when I'll have reasonable input data)
    
    rankdata min method:
    ‘min’: The minimum of the ranks that would have been assigned to all the tied values is assigned to each value.
    (This is also referred to as “competition” ranking.)
    """
    
    a=[]
    #number of ensemble members for final histogram
    bins=ens.members.shape[0]
    
    for l in obs.loc:
        e=ens.sel(loc=l,method='nearest')
        for t in obs.time:
            ### eventually observation time is more sparse than ens.time, so do not compare if there is no 
            ### observed value actually available
            vals=e.sel(time=t,method='nearest').values
            vals=vals.append(obs.sel(time='t',loc=l))
            #use scipy rankdata for appended array, relevant rank is the last value
            rank=rankdata(vals,method='min')[-1]
            a.append(rank)
            
    hist=np.histogram(a,bins)
    return hist
