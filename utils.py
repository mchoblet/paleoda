#Some useful functions that I do not want to put elsewhere

import numpy as np
import xarray as xr
from bisect import bisect_left
import bisect
import dataloader
import psm_pseudoproxy
import evaluation
import tqdm
import warnings
from numba import njit,prange


def config_check(cfg):
    """
    convert configuration settings that should be a list into a list:
    metrics, psm,obsdata,pp_var,
    checks timeresolutions, length of obsdata/psm
    
    #Does not check everything!
    """
    if not isinstance(cfg['metrics'],list):
        #string conversion guarantees that None is skipped
        cfg['metrics']=[str(cfg['metrics'])]
    if not isinstance(cfg['psm'],list):
        cfg['psm']=[cfg['psm']]
    if not isinstance(cfg['obsdata'],list):
        cfg['obsdata']=[cfg['obsdata']]
    if not isinstance(cfg['proxy_error'],list):
        cfg['proxy_error']=[cfg['proxy_error']]
    if not isinstance(cfg['time_scales'],list):
        cfg['time_scales']=[cfg['time_scales']]
        
    #check length of proxy databases and defined psm.
    if len(cfg['obsdata'])!=len(cfg['psm']):
        #when only one None given for psm extend it to a list of correct length
        if len(cfg['psm'])==1:
            if cfg['psm'][0]==None:
                cfg['psm']=[None for i in range(len(cfg['obsdata']))]
        else:
            raise TypeError('obsdata- and psm- configuration are not of same length')

    ##check proxy error config
    if len(cfg['proxy_error'])!=len(cfg['obsdata']):
        if len(cfg['proxy_error'])==1:
            #repeat first value
            val=cfg['proxy_error'][0]
            cfg['proxy_error']=[val for i in range(len(cfg['obsdata']))]
        else:
            raise TypeError('obsdata- and proxy_error- configuration are not of same length')

    #check timescale config 
    if len(cfg['time_scales'])!=len(cfg['obsdata']):
    #when only one None given for psm extend it to a list of correct length
        if len(cfg['time_scales'])==1:
            #repeat first value
            val=cfg['time_scales'][0]
            cfg['time_scales']=[val for i in range(len(cfg['obsdata']))]
        else:
            raise TypeError('obsdata- and time_scale- configuration are not of same length')   
    
    #in case we use pseudoproxies do addtional adjustments

    if cfg['ppe']['use']==True:
        if not isinstance(cfg['ppe']['metrics_ppe'],list):
            cfg['ppe']['metrics_ppe']=[cfg['ppe']['metrics_ppe']]
    
    ###multi-timescale timescale list: sorting and checking

    ts_list=cfg['timescales']
    #convert to list
    if not isinstance(ts_list,list) and not isinstance(ts_list,np.ndarray):
        ts_list=[ts_list]
        print(ts_list)

    #convert to integers
    ts_list=list(map(int,ts_list))

    #sort
    ts_list=np.sort(ts_list)
    cfg['timescales']=np.array(ts_list)
    
    #check if all resolutions divisors of block length
    bl=ts_list[-1] #last element is block length
    for t in ts_list[:-1]:
        assert bl%t==0, 'Time resolution {t} is not a divisor of {bl} '.format(t=t,bl=bl)
   
    return cfg

def prior_preparation(cfg):
    """
    load_priors and take annual mean
    save all variables in one Dataset and separately save attribues
    also save the raw (monthly) prior, in case it is needed for the PSM
    """
    prior=[]
    prior_raw=[]
    for v,p in cfg.vp.items():
        #load only variables which have path not set to None
        if p!=None:
            print('Load',v,' from ',p)
            print('computing yearly average')            
            data=dataloader.load_prior(p,v) 
            #compute annual mean ac
            data_m=dataloader.annual_mean(data,avg=cfg.avg,check_nan=cfg.check_nan)
            
            prior.append(data_m)
            prior_raw.append(data)
    #create Dataset (This will always be a Dataset not Dataarray, also for only one variable)
    prior=xr.merge(prior)
    prior_raw=xr.merge(prior_raw)
    #copy attributes
    attributes=[]
    for v in prior.keys():
        a=prior[v].attrs
        attributes.append(a)
    return prior, attributes, prior_raw

def proxy_load(c):
    """
    #load proxies, their locations and the time-axis
    #this time-axis is the one used for looking at the proxy-timeresolution, the resampling process.
    #create a time_range given c.proxy_time, starts at beginning of year
    """
    
    print('>>>>>LOADING PROXY DATA')
    #Also replaces the error by c.proxy_error (fixed one for all proxies)
    start=c.proxy_time[0]
    end=c.proxy_time[1]
    time=xr.cftime_range(start=start,end=end,freq='YS',calendar='365_day')     
    
    pp_y_all=[]
    pp_r_all=[]

    #loop over each database
    for ip,p in enumerate(c.obsdata):
        #load proxy database, slice time and extract lat and lon
        proxy_db=xr.open_dataset(p,use_cftime=True).squeeze(drop=True)
        
        proxy_db=proxy_db.sel(time=slice(time[0],time[-1]))
        
        lat=proxy_db.lat
        lon=proxy_db.lon
        
        #reindex times (adds nans correctly at beginning/end, slice doesn't extend range)
        proxy_db=proxy_db.reindex({'time':time},method='nearest')
        
        #load proxy values, can also serve as a time mask when useing PPE
        pp_y=proxy_db[c.obs_var[ip][0]]
        if c.psm[ip]!='linear':    
            pp_r=proxy_db[c.obs_var[ip][1]]
        #eventually take the linear regression error when working with a linear PSM
        else:
            pp_r=proxy_db[c.linear['error']]
        
        #drop sites that do not contain values after the time reindexing
        #keeping them would lead to problems in multi timescale
        
        for idx,s in enumerate(pp_y.site):
        
        
            #drop locations without records, but only when working with realproxies
            
            if c.ppe['use']==True:
                avail_times=pp_y.sel(site=s).time.values
            else:
                avail_times=pp_y.sel(site=s).dropna('time').time.values
                            
            if len(avail_times)==0:
                pp_y=pp_y.drop_sel(site=s)
                pp_r=pp_r.drop_sel(site=s)
                lon=lon.drop_sel(site=s)
                lat=lat.drop_sel(site=s)
                proxy_db=proxy_db.drop_sel(site=s)  
                
            #ERROR value replacing
            #filling proxy dataarray like this is the wy to go, because there is no 2d
            #indexing for dataarrays afaik
            
            elif c.proxy_error is not None:
                pp_r.loc[dict(site=s,time=avail_times)]=np.ones(len(avail_times))*c.proxy_error[ip]
        
        #replace site number by string
        sites=[str(ip)+'.'+str(s.values.tolist()) for s in proxy_db['site']]
        
        
        pp_y['site']=sites
        pp_r['site']=sites
        #pp_y_lon['site']=sites
        #pp_y_lat['site']=sites
        
        pp_y_all.append(pp_y)
        pp_r_all.append(pp_r)
        #pp_y_lon.append(lon.values)
        #pp_y_lat.append(lat.values)
    
        #not sure if necessary, but jupyter notebooks started to slow down a lot.
        del proxy_db
    
    #return pp_y_all,pp_r_all,pp_y_lon,pp_y_lat        
    return pp_y_all,pp_r_all

def proxy_timeres(c,pp_y_all):
    """
    For each proxy compute the time resolution (1 value). Resolution mode given by cfg['time_scales'] (e.g. mean, median, min...).
    The resolution is round to the next largest available resolution
    
    In principle one could also attribute one resolution to each timestep, but this doesn't fit to our resampling procedure later, so this part is commented out.
    Will maybe reuse that later. This is why the code still contains part which would produce a 2-D output (time,site) instead
    of just (site)-dimensional as it is now
    
    ---------
    Input:
        c: config (as namespace)
        pp_y_all: List of DataArrays for each proxy_database
    
    Return:
        time_res_list: List of DataArray, one DataArray for each proxy database with 1 timeres value (site coordinate is kept)
    
    """
    
    #timescales have already been sorted in the config_check step
    timescales=c.timescales
    timeres_list=[]
    for ip,db in enumerate(pp_y_all):
        res=db.copy(deep=True)
        
        resols=xr.DataArray(np.zeros(len(db.site)),coords=dict(site=db.site))
        mode=c.time_scales[ip]
        #loop over sites
        for i,s in enumerate(res.site):
            times=db.sel(site=s).dropna('time').time
            years=times.time.dt.year
            
            if len(years)==1: 

                print('only one record for site ',s.values,'. Giving it timescale 1')
                resols[i]=1
                #res.loc[dict(site=s,time=times)]=1
            else:
                #compute distance to the right/left and double the one value at the respective end which is missing
                dist_right=np.abs(np.array(years)[1:]-np.array(years)[:-1])
                #dist_right=np.concatenate([dist_right,np.array([dist_right[-1]])])

                #dist_left=np.abs(np.array(years)[:-1]-np.array(years)[1:])
                #dist_left=np.concatenate([np.array([dist_left[0]]),dist_left])

                #for constant resolutions repeat resolution according to number of years
                if  mode== 'min':
                    res=dist_right.min()
                    #resols=dist_right[:-1].min()*np.ones(len(years))
                elif mode=='mean':
                    res=dist_right.mean()
                    #resols=dist_right[:-1].mean()*np.ones(len(years))
                elif mode=='median':
                    res=np.median(dist_right)
                
                #elif mode=='most':
                #    values,counts=np.unique(dist_right[:-1],return_counts=True)
                #    ind=np.argmax(counts)
                #    resols=values[ind]*np.ones(len(years))  
                
                #number corresponds to prescribed mode
                elif isinstance(mode,float) | isinstance(mode,int) :
                    res=mode
                    #resols=mode*np.ones(len(years))

                #elif mode=='rl_max':
                #    resols=np.array([dist_right,dist_left]).max(axis=0)
                #elif mode=='rl_min':
                #    resols=np.array([dist_right,dist_left]).min(axis=0)
                #elif mode=='rl_mean':
                #    resols=np.array([dist_right,dist_left]).mean(axis=0)

                else:
                    import sys
                    sys.exit("Time resolution mode unknown. Exit.")
                
                #Round:
                #Check that time_res is not larger than largest: -> if so, assign to largest
                if res > timescales[-1]:
                    resols[i]=timescales[-1]
                else:
                    resols[i]=timescales[bisect_left(timescales,res)]
                #eventually round estimated time_scales to ones predefined by algorithm (c.multi_timescale['timescales'])
                #if c.round_time_scales:
                #    eps=1e-10 #epsilon needed for ceil rounding (3 goes to 5 instead of 1)
                #    resols=np.array([min(scales, key=lambda x:abs(x-r+eps)) for r in resols])
                        
        timeres_list.append(resols)
        
    return timeres_list

def resample_proxies(c,timeres_list,times_list,pp_y_all):
    """
    Resampling procedure for each proxy:
        - Upsample proxy timeseries to yearly resolution. Use 'nearest'/'linear' interpolation for the
          nans in between
        - Lowpass filter this timeseries (no filtering if the targetresolution is small than 4 years <- side effects?)
        - Resample to target time_series
        - mask holes in original time series according to the cfg['mask'] factor
        
    The proxies are brought together into a list of DataArrays.
    The sites are given a prefix to distinguish from which proxy-db they are
    """
    
    mask_ = c.mask #masking tolerance factor (mask_ * time_res is max. gap size)
    mode=c.resample_mode
    timescales=np.array(c.timescales) #make sure it's really a numpy array
    
    #create list of lists for each proxy_db and each timescale
    lisst=[]
    
    #store data in dictionary for all proxies in this database
    dictionary={}
    for scale in timescales:
        dictionary[str(scale)]=dict(ts=[],sites=[])
    
    #loop over proxy dbs
    print('resampling of proxies in each database')
    for i, db in enumerate(pp_y_all):
    
        timeres_vals=timeres_list[i].values
        for ii,s in enumerate(tqdm.tqdm(db.site)):
        
            #proxy data
            data=db.sel(site=s)
            #timeresolution for this one proxy
            res=int(timeres_vals[ii])
            #look up targettimeseries for this resolution (It's fast)
            
            idx=int(np.argwhere(timescales==res).flatten())
            target_time=times_list[idx]
            
            #resample. If res <4 don't use the lowpass filter.
            if res<4:
                    filt=False
            else:
                    filt=True
            
            resampled=make_equidistant_target(data,target_time,target_res=res,method_interpol=mode,filt=filt,min_ts=1,)
            #mask the gaps
            resampled=mask_the_gap_alt(resampled,data, time_res=res,tol=mask_)
                
            #add to dictionary
            dictionary[str(res)]['ts'].append(resampled.values)
            
            #create site with prefix indicating the database. 
            #site=float(str(i)+'.'+str(s.values.tolist()))
            #Keep site as string, else 0.1 will be=0.10
            dictionary[str(res)]['sites'].append(s.values.tolist())
            
            if c.reuse==True:
                #loop over all the other timescales to the right
                for t_ii,t_i in enumerate(timescales[idx+1:]):
                    res=int(t_i)
                    #targettimeseries for this resolution
                    idx=int(np.argwhere(timescales==res).flatten())
                    target_time=times_list[idx]
            
                    #resample. If res <4 don't use the lowpass filter.
                    if res<4:
                            filt=False
                    else:
                            filt=True    
                    resampled=make_equidistant_target(data,target_time,target_res=res,method_interpol=mode,filt=filt,min_ts=1,)
                    #mask the gaps
                    resampled=mask_the_gap_alt(resampled,data, time_res=res,tol=mask_)

                    #add to dictionary
                    dictionary[str(res)]['ts'].append(resampled.values)
                    dictionary[str(res)]['sites'].append(s.values.tolist())
                    
        lisst.append(dictionary)
    
    #loop over dictionary and bring together
    final_list=[]
    for i,dic in dictionary.items():
        vals=np.stack(dic['ts'])
        sites=dic['sites']
        idx=int(np.argwhere(timescales==int(i)).flatten())
        target_time=times_list[idx]
        data_array=xr.DataArray(vals,coords=dict(site=sites,time=target_time))
        #We add an attribute to each time-series to have the number of proxies per database directly accesible
        #convert sites floats and the to integers, count occurence
        
        integers=(list(map(int,list(map(float,sites)))))
        data_array.attrs['DB_members']=np.unique(integers,return_counts=True)[1]
        final_list.append(data_array.transpose('time','site'))
       
    return final_list


def make_equidistant_target(data,target_time,target_res,method_interpol='nearest',filt=True,min_ts=1):
    """
    Takes a proxy timeseries "data" (fully resolved,with nans in between if no value available) and resamples it equidistantly to the (equidistant) target timeseries
    "target_time"  (DataArray of cftime-objects, we need .dt.time.year accessor), which has the resolution "target_res" (consistency with target_time is not checked).
    We usualy set the target_res to the median resolution.
    
    The resampling procedure is adapted from the Paleospec R-package: https://github.com/EarthSystemDiagnostics/paleospec/blob/master/R/MakeEquidistant.R.
    The time resolution is based on yearly data. Other time resolution (monthly) would require adapting the filtering part.
    
    Code consists of the following steps.
        0. Duplicate first non-nan-data point if this required by target_time spacing
        1. Resample and interpolate time series to 'min_ts'-resolution (yearly makes sense in our case). Nearest neighbor interpolation!
        2. Low_pass filter resampled time_series (High order Butterworth filter used in original R-package, I use filtfilt to avoid time lag)
        3. Resample to target resolution
    
    Comments:
        1. Be aware that some proxy records have huge jumps without data in between. The resampled values there are not meaningful and need to be masked separately.
        2. Use xarray > v2022.06.0 to make use of fast resampling operation (but slowness of old version not a problem for our time-lengths)
    
    Example:
        Given some time-series with measurements at time [4,9,14,19,24], which we treat as mean-values for the time range centered on these times.
        We want to resample it equidistanced for the times [0,5,10,15,20,25]. These target labels are actually the left edge of a time block 
        (in the DA we effectively reconstruct the mean of the years [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24]])
        Therefore when using the xarray resample method in the final step (down-sampling) it is important to set closed='left' (and eliminate the last element), so that it is logically consistent.
    """
    
    #drop nan entries in data, extract data which are not nan
    #without dropping nans interpolation wont work
    data=data.dropna('time')
    vals=data.values

    #time_values and years contained in original time_series
    time=data.time.values
    time_years=data.time.dt.year.values

    #For the first year included in proxy timeseries find the nearest year in target_time, which is smaller than the first year.
    #Repeat first data value and append this value and its time to the values. Do not do this if the first year is part of the target_time.

    first_year=time_years[0]
    target_years=target_time.time.dt.year.values

    #find by modulo calcuation and search sorted (could also create new Datetimeobject)
    #take into consideration the start time that might be shifted
    start=first_year-first_year % target_res + target_years[0]%target_res

    if start!=first_year:
        idx = np.searchsorted(target_years, first_year, side="left")
        time_add=target_time[idx-1].values

        #insert time and duplicate first value
        time=np.insert(time,0,time_add)
        vals=np.insert(vals,0,vals[0])

    vals_new=xr.DataArray(data=vals,coords=dict(time=time))

    #1. resampling (upsampling) and interpolating (upsampling)
    min_ts=str(min_ts)+'YS'
    #import pdb
    #pdb.set_trace()
    try:
        upsampled=vals_new.resample(time=min_ts).interpolate(method_interpol)
    except:
        if len(vals_new.time)==1:
            #case of only one value, then no interpolation.
            #already have correct time, checked in start!=first_year
            upsampled=vals_new
        else:
            import pdb
            pdb.set_trace()

    ##Fill nans (already done in previous step)
    #upsampled=upsampled.interpolate_na('time',method='linear')

    #2. LOW PASS FILTER for resampled time series (avoid aliasing)
    from scipy.signal import butter, lfilter, filtfilt

    def butter_lowpass(cutoff, fs, order=6, kf=1.2):
        # kf:  scaling factor for the lowpass frequency; 1 = Nyquist, 1.2 =
        #' 1.2xNyquist is a tradeoff between reducing variance loss and keeping
        #' aliasing small
        #fs is basic timestep (min_ts)
        #nyquist frequency
        nyq = 0.5 * fs 
        normal_cutoff = cutoff / nyq * kf
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=6):
        #filtfilt does not introduce a time-lag in comparison to butterworth 
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        #y = lfilter(b, a, data)
        return y

    cutoff=1/target_res
    fs=1 #yearly base frequency
    #pdb.set_trace()
    if filt==True:
        try:
            up_filt=butter_lowpass_filter(upsampled,cutoff,fs,order=4)
        except:
            #for short reconstruction time range (e.g.1900-1999), the resampling can not work out, then just take the value as it is
            up_filt=upsampled
    else:
        up_filt=upsampled

    ###3. RESAMPLE TO TARGET RESOLUTION
    
    #string for resampled option 'YS': year start is very important
    target_res_st=str(target_res)+'YS'

    #convert up_filt to Dataarray in order to use resample method
    up_filt=xr.DataArray(up_filt, coords=dict(time=upsampled['time']))
    #pdb.set_trace()
    resampled=up_filt.resample(time=target_res_st,closed='left').mean('time')
    
    #reindex time to get back to the global full timescales (non existing values filled with nan)
    
    final=resampled.reindex(time=target_time)

    return final


def mask_the_gap_alt(resampled_ts, original_ts, time_res,tol):
    """
    Function for masking gaps after the resampling.
    It looks for gaps in the original timeseries and masks them in the resampled timeseries.
    
    Input: 
        resampled: equidistant time-series (from proxy_db beginning to end, containing nans at beginning/start)
        time_res: Resolution of resampled time-series
        original_ts: original time_series of proxy from proxy_db-table (containing nans in between measurement if measurement not yearly)
        tol(erance): size of gap with respect to time_res (tol*time_res), here it is a factor that is multiplied with each timeresolution
        
    """
    #copy
    resampled_ts=resampled_ts.copy()
    
    #maximum allowed gap
    max_gap=tol*time_res

    #screen original timeseries for jumps
    original_ts_years=original_ts.dropna('time').time.dt.year
    gaps=np.abs(np.array(original_ts_years)[1:]-np.array(original_ts_years)[:-1])

    #find index where gap > max_gap (left end)
    args=np.argwhere(gaps>max_gap).flatten()

    #select according years
    #starts=original_ts.dropna('time').time[args].dt.year
    #ends=original_ts.dropna('time').time[args+1].dt.year

    starts=original_ts_years[args]
    ends=original_ts_years[args+1]
    
    target_time_ts=resampled_ts['time']
    target_years_ts=resampled_ts['time'].dt.year

    #in target years, find the ones that are larger than start and smaller than end
    #bisectional search is the most efficient way, list comprehension would be orders of magnitude slower
    #we use bisect-righ for findin the elements (right/left indicates if index to right/left is chosen for equality)
    #For the end we keep the first to the left of the end (because it's influenced by the measurement to the right) and thus
    #select the penultimate one
    
    for ii,t in enumerate(starts):

        #find indices with bisect right
        start_idx=bisect.bisect_right(target_years_ts,starts[ii]) 

        #end index,-2 because slice also selects last element
        end_idx=bisect.bisect_right(target_years_ts,ends[ii])-2

        resampled_ts.loc[dict(time=slice(target_time_ts[start_idx],target_time_ts['time'][end_idx]))]=np.nan
        
    return resampled_ts


def noresample_proxies(c,timeres_list,times_list,pp_y_all):
    """
    Function for using multi-timescale DA without the (full) proxy resampling scheme.
    It just does the necessary conversion to the times from times_list
        
    The proxies are brought together into a list of DataArrays.
    The sites are given a prefix ('0.', '1.'...) to distinguish from which proxy-db they are
    """
    timescales=np.array(c.timescales)
    
    #create list of lists for each proxy_db
    lisst=[]
    
    #store data in dictionary for all proxies in this database
    dictionary={}
    for scale in timescales:
        dictionary[str(scale)]=dict(ts=[],sites=[])
    
    #loop over proxy dbs
    for i, db in enumerate(pp_y_all):
        timeres_vals=timeres_list[i].values
        
        for ii,s in enumerate(db.site):
            #proxy data
            data=db.sel(site=s)
            #timeresolution for this one proxy
            res=int(timeres_vals[ii])
            res_str=str(res)+'YS'
            #look up targettimeseries for this resolution (It's fast)
            idx=int(np.argwhere(timescales==res).flatten())
            target_time=times_list[idx]
            resampled=data.resample(time=res_str,closed='left').mean()
                
            #add to dictionary
            dictionary[str(res)]['ts'].append(resampled.values)
            
            #create site with prefix indicating the database. 
            #site=float(str(i)+'.'+str(s.values.tolist()))
            #Keep site as string, else 0.1 will be=0.10
            dictionary[str(res)]['sites'].append(s.values.tolist())
            
            if c.reuse==True:
                #loop over all the other timescales to the right
                for t_ii,t_i in enumerate(timescales[idx+1:]):
                    res=int(t_i)
                    res_str=str(res)+'YS'
                    #target timeseries for this resolution
                    idx=int(np.argwhere(timescales==res).flatten())
                    resampled=data.resample(time=res_str,closed='left').mean()

                    #add to dictionary
                    dictionary[str(res)]['ts'].append(resampled.values)
                    dictionary[str(res)]['sites'].append(s.values.tolist())
                    
        lisst.append(dictionary)
    
    #loop over dictionary and bring together
    final_list=[]
    for i,dic in dictionary.items():
        vals=np.stack(dic['ts'])
        sites=dic['sites']
        idx=int(np.argwhere(timescales==int(i)).flatten())
        target_time=times_list[idx]
        data_array=xr.DataArray(vals,coords=dict(site=sites,time=target_time))
        #We add an attribute to each time-series to have the number of proxies per database directly accesible
        #convert sites floats and the to integers, count occurence
        
        integers=(list(map(int,list(map(float,sites)))))
        data_array.attrs['DB_members']=np.unique(integers,return_counts=True)[1]
        final_list.append(data_array.transpose('time','site'))
       
    return final_list

def psm_apply(c,prior,prior_raw, pp_y_all):
    """
    Takes prior and config.
    Psm weighted yearly average requires monthly data (prior_raw)
    Converts prior values into HXfull (full time-series at proxy locations, corresponding to prior)
    PP_y_all needed for proxy site - name and metadata
    
    Options:
        -interpolation to nearest grid point (None vs distance weighted)
        - No-psm: Nearest/distance weighted variable of type XY 
        - Speleo:
            - weighting: inf, prec, None
            - height correction
            - fractionation: True/False
            - alphas (Tremaine for calcite, Grossman for Aragonite)
            - filter: False/True/float (transit time in cave) #standard convolution time is 2.5 years
        - Icecore:
            - weighting: precipitation, None
            - height: True/False (given by orography file)
        
    These options are given in c.psm as a list
    """
    #List where we append the model proxy estimates to for each database
    HXfull_all=[]
    
    #loop over psms (which corresponds do looping over the proxy dbs, because there is one psm for each proxy db)
    for i,psm in enumerate(c.psm):
        #extract values
        #lats=pp_y_lat[i]
        #lons=pp_y_lon[i]
        
        lats=pp_y_all[i].lat.values
        lons=pp_y_all[i].lon.values

        proxies=pp_y_all[i]
        if psm==None:
            #extract needed variable from prior
            var=c.var_psm[i]
            prior_var=prior[var]
            #extrapolate according to positions of proxies
            HXfull=psm_pseudoproxy.obs_from_model(prior_var,lats,lons,interpol=c.interpol)
        
        elif psm=='linear':
            HXfull=psm_pseudoproxy.linear_psm(c, prior, pp_y_all[i])
        
        elif psm=='speleo':
            #weighting
            print('USING SPELEO PSM')
            
            if c.speleo['weighting']=='inf':
                print('>>>>>>>>>>>>>GETTING MONTHLY d18O Data')
                d18=psm_pseudoproxy.infilt_weighting(prior_raw['d18O'],prior_raw['prec'],prior_raw['evap'],slice_unfinished=True,check_nan=c.check_nan)
            elif c.speleo['weighting']=='prec':
                print('>>>>>>>>>>>>>GETTING MONTHLY d18O Data')
                d18=psm_pseudoproxy.prec_weighting(prior_raw['d18O'],prior_raw['prec'],slice_unfinished=True,check_nan=c.check_nan)
            else:
                d18=prior['d18O']
            
            d18=psm_pseudoproxy.obs_from_model(d18,lat=lats,lon=lons,interpol=c.interpol)
            #replace site names, else missing
            d18['site']=proxies['site']
            
            tsurf=prior['tsurf']
            tsurf=psm_pseudoproxy.obs_from_model(tsurf,lat=lats,lon=lons,interpol=c.interpol)
            tsurf['site']=proxies['site']
            
            #height correction 
            if c.speleo['height']==True:
                print('>>>>>>>>>>>>>APPLYING HEIGHT CORRECTION')
                oro=xr.open_dataset(c.oro)['oro']
                #my obs_from model function is a sel that works with single lats and lons
                #obs from model interpolation seems to be important for some speleos in order to not completly
                #be off (especially Echam and CESM Model)
                oro=psm_pseudoproxy.obs_from_model(oro,lat=lats,lon=lons,interpol='dw')
                
                elev=proxies['elev']
                z=(elev-oro)
            
                #Tsurf: -0.65 https://en.wikipedia.org/wiki/Lapse_rate
                #d18O take global value: global average -0.28: https://www.ajsonline.org/content/ajs/301/1/1.full.pdf
                
                #most height sensitive speleos are in the himalay, where the lapse rate is smaller
                d18= d18 + -0.15/100*z
                tsurf = tsurf + -0.65/100*z

            #fractionation (separated treatment of aragonite and calcite)   
            if c.speleo['fractionation']==True:
                print('>>>>>>>>>>>>>APPLYING FRACTIONATION')
                #distinguish between aragonite/non-aragonite sites, assume non-aragonite is calcite
                
                arag_sites=proxies.where(proxies['mineralogy']=='aragonite',drop=True).site
                calc_sites=proxies.where(proxies['mineralogy']!='aragonite',drop=True).site
                
                #applying the mean tsurf, not tsurf itself, as else the covariance pattern is seriously reduced!
                
                if c.speleo['fractionation_temp']=='mean':
                    print('use mean temperature')
                    d18_calc=psm_pseudoproxy.frac(d18.sel(site=calc_sites),tsurf.sel(site=calc_sites).mean('time'),psm_pseudoproxy.pdb_coplen,psm_pseudoproxy.alpha_calc_trem)
                    d18_arag=psm_pseudoproxy.frac(d18.sel(site=arag_sites),tsurf.sel(site=arag_sites).mean('time'),psm_pseudoproxy.pdb_coplen,psm_pseudoproxy.alpha_arag_grossman)
                
                elif c.speleo['fractionation_temp']=='regular':
                    print('use time-varying temperature')
                    d18_calc=psm_pseudoproxy.frac(d18.sel(site=calc_sites),tsurf.sel(site=calc_sites),psm_pseudoproxy.pdb_coplen,psm_pseudoproxy.alpha_calc_trem)
                    d18_arag=psm_pseudoproxy.frac(d18.sel(site=arag_sites),tsurf.sel(site=arag_sites),psm_pseudoproxy.pdb_coplen,psm_pseudoproxy.alpha_arag_grossman)
                else:
                    print('unknown fractionation_temp-mode')
                    break    
                
                d18.loc[dict(site=calc_sites)]=d18_calc
                d18.loc[dict(site=arag_sites)]=d18_arag
            
            #karst filter
            if c.speleo['filter']==True:
                print('>>>>>>>>>>>>>APPLYING KARST FILTER')
                #following the PRYSM PSM by S. Dee (2015). Transit time 2.5 as in Bühler 2021
                #for individual transit time it would be easiest to adapt the proxy_db DataSet with some additional coordinate as metadata
                
                tau0=c.speleo['t_time']
                #set timeseries
                tau=np.arange(len(d18.time))

                #Green's function
                #well-mixed model
                g = (1./tau0) * np.exp(-tau/tau0)

                #normalize tau (as we have yearly spaced values, we just sum up all values)
                g=g/np.sum(g)
                
                for s in d18.site:
                    #convolve d18O with g
                    #subtract mean
                    vals=d18.sel(site=s)
                    mean=vals.mean('time').values

                    #get time axis number
                    ax=vals.get_axis_num('time')
                    #convolve padding first/last value (no-problem as g decreases very quickly anyqay)
                    conv=np.apply_along_axis(lambda m: np.convolve(m, g), axis=ax, arr=(vals-mean).values)[:len(vals)]
                    #exchange values in initial array

                    d18.loc[dict(site=s)]=conv+mean

            HXfull=d18
            
        elif psm=='icecore':
            print('USING ICECORE PSM')
            #weighting
            if c.icecore['weighting']=='prec':
                d18=psm_pseudoproxy.prec_weighting(prior_raw['d18O'],prior_raw['prec'],slice_unfinished=True)
            else:
                d18=prior['d18O']
            
            d18=psm_pseudoproxy.obs_from_model(d18,lat=lats,lon=lons,interpol=c.interpol)
            
            #height correction
            if c.icecore['height']==True:
                print('>>>>>>>>>>>>>APPLYING HEIGHT CORRECTION')
                oro=xr.open_dataset(c.oro)['oro']
                oro=psm_pseudoproxy.obs_from_model(oro,lat=lats,lon=lons,interpol=c.interpol)
                #oro.sel(lat=lats,lon=lons,method='nearest')
                elev=proxies['elev']
                
                z=(elev-oro)
            
                #Tsurf: -0.65 https://en.wikipedia.org/wiki/Lapse_rate
                #d18O take global value: global average -0.28: https://www.ajsonline.org/content/ajs/301/1/1.full.pdf
                d18= d18 + proxies['lapse_rate']/100*z
            
            #Diffusion and compactation
            if c.icecore['filter']==True:
                print('>>>>>>>>>>>>>APPLYING PRYSM ICECORE FILTER')
                
                prec_site=psm_pseudoproxy.obs_from_model(prior['prec'],lat=lats,lon=lons,interpol=c.interpol)
                prec_site['site']=d18.site
                tsurf_site=psm_pseudoproxy.obs_from_model(prior['tsurf'],lat=lats,lon=lons,interpol=c.interpol)                
                tsurf_site['site']=d18.site
                
                for s in tqdm.tqdm(d18.site):
                    #only nproc=1 workso
                    d18.loc[dict(site=s)]=psm_pseudoproxy.ice_archive(d18.sel(site=s),prec_site.sel(site=s),tsurf_site.sel(site=s),xr.DataArray(np.array([101.325]*len(prior.time))),nproc=1)

            HXfull=d18
            
            
   
            
        else:
            raise Exception("Given psm type unknown. check 'psm' in config dictionary.")
        
        #add site coordinate
        HXfull['site']=proxies['site']
        HXfull_all.append(HXfull)
    
    return HXfull_all
 
def resample_wrapper(c,pp_y_all,pp_r_all):    
    #Suppres warnings. Bad practice, but warnings in resampling part are annoying (some pandas stuff)
    warnings.simplefilter("ignore",category=DeprecationWarning) 
    warnings.simplefilter("ignore",category=FutureWarning) 

    #time arrays for each resolution
    length=int(c.proxy_time[1])-int(c.proxy_time[0])
    times_list=[xr.DataArray(xr.cftime_range(start=c.proxy_time[0],periods=(length//i+1),freq=str(i)+'YS',calendar='365_day'),dims='time') for i in c.timescales]

    #adapt times_list (cut end in case it doesn't fit perfectly with largest block size)
    #I needed an (eventually) different times_list for resampling the proxies
    #this could definitely be nicer
    new_times_list=[]
    time_sc=c.timescales
    for i,t in enumerate(times_list):
        ts=time_sc[i]
        end=str(((int(c.time[1])-int(c.time[0]))//ts)*ts+int(c.time[0]))
        if end>c.proxy_time[1]:
            end=str(((int(c.proxy_time[1])-int(c.time[0]))//ts)*ts+int(c.time[0]))
        new_times_list.append(t.sel(time=slice(c.time[0],end)))
    times_list=new_times_list

    #drop where there are no values in the final time range (only without ppe)
    first_time=times_list[0][0]
    last_time=times_list[0][-1]

    if c.ppe['use']==False:
        pp_y_all_new=[]
        pp_r_all_new=[]
        for idx,pp in enumerate(pp_y_all):
            pp_l=pp.sel(time=slice(first_time,last_time))
            pp2=pp.copy(deep=True)
            pp_r2=pp_r_all[idx].copy(deep=True)
            for s in (pp.site):
                avail_times=pp_l.sel(site=s).dropna('time').time.values            
                if len(avail_times)==0:
                    pp2=pp2.drop_sel(site=s.values)
                    pp_r2=pp_r2.drop_sel(site=s.values)
            pp_y_all_new.append(pp2)
            pp_r_all_new.append(pp_r2)
        pp_y_all=pp_y_all_new
        pp_r_all=pp_r_all_new

        #one list for each proxy_db containing the mean/median/min timeresolution rounded to given timescale, controled by config timescales
        print('COMPUTE TIMERESOLUTION OF PROXIES')
        timeres_list=proxy_timeres(c,pp_y_all) 

    #cut the proxies in time according to kalman_filter settings. it's important to cut only at that step here
    #else we don't get the real timeresolution of the proxy
    pp_y_all=[pp.sel(time=slice(first_time,last_time)) for pp in pp_y_all]
    pp_r_all=[pp.sel(time=slice(first_time,last_time)) for pp in pp_r_all]


    #list of Dataarrays for each timescale
    if c.ppe['use']==False:
        if c.resample:
            print('RESAMPLING THE PROXIES')
            lisst=resample_proxies(c,timeres_list,times_list,pp_y_all)

        #case where we don't want the fancy resampling technique, just assigning the values as they are (eventually means if more than one value for each subblock)
        elif len(c.timescales) > 1:
            lisst=noresample_proxies(c,timeres_list,times_list,pp_y_all)

        #just assign each proxy to annual and keep the timeseries as they are. Proxies are going to be used as they are available in the original table
        #Separate concetanation because xr-merge doesn't work due to the (necessary) metadata in the prox_dbs
        else:
            sites=[]; vals=[]; lats=[]; lons=[]
            for db in pp_y_all:
                sites.append(db.site.values); vals.append(db.values); lats.append(db.lat.values); lons.append(db.lon.values)

            time=pp_y_all[0].time #time has been set to the same in all databases
            #get site axis in data, not to be confounded with time axis
            ax=db.get_axis_num('site')
            sites=np.concatenate(sites,axis=-1); vals=np.concatenate(vals,axis=ax); lats=np.concatenate(lats,axis=-1); lons=np.concatenate(lons,axis=-1)
            da=xr.DataArray(vals,coords=dict(time=time,site=sites))
            da['lat']=('site',lats); da['lons']=('site',lons)
            #count Database members and add as an attribute
            integers=(list(map(int,list(map(float,sites)))))
            da.attrs['DB_members']=np.unique(integers,return_counts=True)[1]
            lisst=[da]

        #create error_list  
        #the lisst is a List with one DataArray for each time_resolution
        #it does not contain the resampled errors, because the resampling scheme doesn't fit to it. replace by the fixed error or by the median error if c.proxy_error=None
        lisst_r=[]
        for i,ii in enumerate(lisst):
            pp_r=ii.copy(deep=True)
            #loop through sites
            for j,s in enumerate(ii.site):
                avail_times=pp_r.sel(site=s.values).dropna('time').time.values
                if c.proxy_error[int(float(s))] is not None:
                    #int(float(s)) because we have already put the proxies together in lisst
                    pp_r.loc[dict(site=s,time=avail_times)]=c.proxy_error[int(float(s))]
                else:
                    med=np.nanmedian(pp_r_all[int(float(s))].sel(site=s))
                    pp_r.loc[dict(site=s,time=avail_times)]=np.ones(len(avail_times))*med
            lisst_r.append(pp_r)
    
    else:
        lisst=None
        lisst_r=None
    
    return pp_y_all,pp_r_all,times_list,lisst,lisst_r
 
   
def pseudoproxy_generator(cfg,HXfull_all,pp_y_all,times_list):
    """
    Wrapper for generating pseudoproxies at the locations given by the proxy-dbs.
    Generates proxies from the modeled proxy estimates (HXfull_all) with a specific SNR value.
    Can also produce pseudoproxies from another model (there HXfull is computed first given the PSM-config part):
        - in that case we might have to adapt times_list, as the model have slightly different, start/end points -> implemented
    
    Multi_timescale proxies:
    The Pseudoproxies are computed first on an annual basis, and then resampled to the target timeseries.
    (Same dictionary structure as in the resample proxy timeseries part)
    
    comments:
    cfg instead of c, because I need to adapt it for external proxies
    pp_y_all only needed for proxy metadata (speleo/icecore psm)
    return pp_y_all, pp_r_all (for each database)
    """
    from types import SimpleNamespace
    c=SimpleNamespace(**cfg)
    
    #create times-dictionary
    timescales=np.unique(np.hstack(c.ppe['multi_timescale']))
    timescales=np.sort(np.array(timescales))

    #exchange times_list
    length=int(c.proxy_time[1])-int(c.proxy_time[0])
    times_list=[xr.DataArray(xr.cftime_range(start=c.proxy_time[0],periods=(length//i++1),freq=str(i)+'YS',calendar='365_day'),dims='time') for i in timescales]

    dictionary={}
    for scale in timescales:
        dictionary[str(scale)]=dict(ts=[],sites=[],r=[])

    #internal proxy
    if c.ppe['source']=='internal':
        print('GENERATING PSEUDOPROXIES FROM SAME MODEL')
        #loop over Databases
        for i,HXfull in enumerate(HXfull_all):
            print('>>>>>>>>>>>>GENERATING pseudoproxies with SNR:', c.ppe['SNR'][i])
           
            #Resample HXfull to the right
            for time_s in c.ppe['multi_timescale'][i]:
                #option 'S' for end of timeseries
                step=str(time_s)+'YS'
                #resample HXfull
                HXfull_res=HXfull.resample(time=step,closed='left').mean('time')
                
                pp_y,pp_r=psm_pseudoproxy.pseudoproxies(HXfull_res,SNR=c.ppe['SNR'][i],noisetype=c.ppe['noise_type'],seed=c.seed)
            
                #cut pseudoproxies in time
                pp_y=pp_y.sel(time=slice(c.proxy_time[0],c.proxy_time[1]))
                pp_r=pp_r.sel(time=slice(c.proxy_time[0],c.proxy_time[1]))
                
                #append!
                dictionary[str(time_s)]['ts'].append(pp_y.values)#p.resample(time=step,closed='left').mean('time')
                dictionary[str(time_s)]['sites'].append(pp_y['site'].values)
                #also error
                dictionary[str(time_s)]['r'].append(pp_r.values)#seudo_r_all[i].resample(time=step,closed='left').mean('time')
    
        prior_ext=None
        HXfull_all_ext=None
    
    #external proxy
    elif c.ppe['source']=='external':
        print('GENERATING PSEUDOPROXIES FROM MODEL OTHER THAN PRIOR')
        import copy
        #adapt config to second model. also orography (can be none if no speleo-psm)
        cfg_2=copy.deepcopy(cfg)
        cfg_2['vp']=cfg_2['ppe']['other_model']['vp']

        try:
            cfg_2['oro']=cfg_2['ppe']['oro'] #it does not make sense to use orography option, as HXfull is computed with
        except:
            pass
        
        #eventually change the psm-config from above for the actual pseudoproxies (above this was for the HX) part
        for psm_new in list(cfg_2['ppe']['psm'].keys()):
            if cfg_2['ppe']['psm'][psm_new] is not None:
                cfg_2[psm_new] = cfg_2['ppe']['psm'][psm_new]
                print('Changed Pseudoproxy-PSM-Config for ', psm_new)

        c2=SimpleNamespace(**cfg_2)
        
        prior_ext, attributes_ext, prior_monthly_ext=prior_preparation(c2)
        
        
        HXfull_all_ext=psm_apply(c2,prior_ext,prior_monthly_ext, pp_y_all)
        
        for i,HXfull in enumerate(HXfull_all_ext):
            
            #Resample HXfull to the different timescales and then add the noise
            for time_s in c.ppe['multi_timescale'][i]:
                #option 'S' for end of timeseries
                step=str(time_s)+'YS'
                
                #resample HXfull
                HXfull_res=HXfull.resample(time=step,closed='left').mean('time')
                
                pp_y,pp_r=psm_pseudoproxy.pseudoproxies(HXfull_res,SNR=c.ppe['SNR'][i],noisetype=c.ppe['noise_type'],seed=c.seed)
            
                #cut pseudoproxies in time
                pp_y=pp_y.sel(time=slice(c.proxy_time[0],c.proxy_time[1]))
                pp_r=pp_r.sel(time=slice(c.proxy_time[0],c.proxy_time[1]))
                
                #append!
                dictionary[str(time_s)]['ts'].append(pp_y.values)#p.resample(time=step,closed='left').mean('time')
                dictionary[str(time_s)]['sites'].append(pp_y['site'].values)
                #also error
                dictionary[str(time_s)]['r'].append(pp_r.values)#seudo_r_all[i].resample(time=step,closed='left').mean('time')        
        
        
        """
        #ADAPTATION OF TIMES_LIST # NEEDS FIXING. THE IDEA IS NICE, BUT I HAVE TO THINK ABOUT HOW TO EXACTLY HANDLE IT.
        #PROBABLY JUST BEST TIME TO SELECT
        
        #make sure that the time or proxies fits to the model cfg['proxy_time']
        # and the timescales. this can a bit messed up because the models are not exactly of same
        #length.
        
        import pdb
        pdb.set_trace()
        
        #this: refers to the last timescale, hence the largest.
        
        #start and ending of model on the longest time-scale        
        start_mod=pp_y.time[0].dt.year.values.tolist()
        end_mod=pp_y.time[-1].dt.year.values.tolist()
        
        #To-Do: Is this really necessary?
        #compare to old starting point
        if (HXfull_all_ext[0].time[0].dt.year.values.tolist()!= start_mod): #or (HXfull_all[0].time[-1].dt.year.values.tolist()!= end_mod):
            #print('TIMES_LIST ADAPTED')
            length=int(end_mod)-int(start_mod)

            if start_mod >=1000:
                start_year_str=str(start_mod)
            else:
                start_year_str='0'+str(start_mod)

            if end_mod >=1000:
                end_year_str=str(end_mod)
            else:
                end_year_str='0'+str(end_mod)            
            
            times_list=[xr.DataArray(xr.cftime_range(start=start_year_str,end=end_year_str,freq=str(i)+'YS',calendar='365_day'),dims='time') for i in c.timescales]
        """
    else: raise Exception("Pseudoproxy source unknown, check c['ppe']['source'].")
           
    #loop over dictionary and bring together into list of lists
    final_list=[]
    error_list=[]
    for i,dic in dictionary.items():
        
        vals=np.concatenate(dic['ts'],axis=-1)
        sites=np.concatenate(dic['sites'],axis=-1)
        
        idx=int(np.argwhere(timescales==int(i)).flatten())
        target_time=times_list[idx]

        #target_time may not be fully covered by model -> times_list adapted in step before
        
        data_array=xr.DataArray(vals,coords=dict(time=target_time,site=sites))
        #We add an attribute to each time-series to have the number of proxies per database directly accesible
        #convert sites to integers, count occurence
        integers=(list(map(int,list(map(float,sites)))))
        data_array.attrs['DB_members']=np.unique(integers,return_counts=True)[1]
        final_list.append(data_array.transpose('time','site'))
        
        #add errors
        vals=np.concatenate(dic['r'],axis=-1)
        
        #import pdb
        #pdb.set_trace()
        
        data_array=xr.DataArray(vals,dims=('time','site'),coords=dict(time=target_time.values,site=sites))
        #We add an attribute to each time-series to have the number of proxies per database directly accesible
        data_array.attrs['DB_members']=np.unique(integers,return_counts=True)[1]
        error_list.append(data_array.transpose('time','site'))         
             
    return prior_ext, HXfull_all_ext,final_list,error_list,times_list

def anomaly_proxies(c,pp_y_all):
    """
    Anomalies only computed on the proxies (list form due to multiple timescales)
    Separation into two functions useful since wrapper also include multi-model-prior option.
    """
    def proxies_anomaly(proxy_list,start=None,end=None):
        pp_y_list=[]
        if start is not None:
            for pp in pp_y_all_a:
                m_p=pp.sel(time=slice(start,end)).mean('time',skipna=True)
                pp_y_list.append(pp-m_p)
        else:
            for pp in pp_y_all_a:
                m_p=pp.mean('time',skipna=True)
                pp_y_list.append(pp-m_p)
        return pp_y_list
    
    pp_y_all_a=pp_y_all.copy()    
    
    if isinstance(c.anomaly_time,list):
        start=c.anomaly_time[0]
        end=c.anomaly_time[1]
    else:
        start=None
        end=None
    
    if (c.anomaly=='HX') | (c.anomaly=='BG+HX') : 
        pp_y_all_a=proxies_anomaly(pp_y_all_a,start,end)
    
    return pp_y_all_a

def anomaly_noproxies(c,HXfull_all,prior):
    prior_a=prior.copy(deep=True)
    HXfull_all_a=HXfull_all.copy(deep=True)
    
    print('COMPUTE ANOMALIES IN MODE:',c.anomaly)
    #compute annomaly with respect to mean
    #only background/prior
    if isinstance(c.anomaly_time,list):
        start=c.anomaly_time[0]
        end=c.anomaly_time[1]

    if c.anomaly=='BG':
        if isinstance(c.anomaly_time,list):
            mean=prior_a.sel(time=slice(start,end)).mean('time',skipna=True)
        else:
            mean=prior_a.mean('time')
        prior_a=prior_a-mean
    #background / prior proxy estimates (and also the proxies!)
    elif (c.anomaly=='HX') | (c.anomaly=='BG+HX') : 
        if isinstance(c.anomaly_time,list):        
            m_H=HXfull_all_a.sel(time=slice(start,end)).mean('time',skipna=True)
            HXfull_all_a=HXfull_all_a-m_H
            
            if c.anomaly=='BG+HX':
                mean=prior_a.sel(time=slice(start,end)).mean('time',skipna=True)
                prior_a=prior_a-mean
                
        else:
            m_H=HXfull_all_a.mean('time',skipna=True)
            HXfull_all_a=HXfull_all_a-m_H
        
            if c.anomaly=='BG+HX':
                mean=prior_a.mean('time',skipna=True)
                prior_a=prior_a-mean
    
    #add back the db-members attribute that would get lost
    HXfull_all_a.attrs['DB_members']=HXfull_all.attrs['DB_members']
    
    return HXfull_all_a,prior_a

def extra_assimil(c,prior,prior_raw,HXfull_all_fin):
    """
    Update 06.09.22: Proxy estimates are always saved!
    Creates a list of all the stuff that is also supposed to be assimilated,
    proxy estimates, globalmean, lat_mean, regional_means.
    Important is that they all have the form (time,site/name...). The second axis is where all the data is going to be concatenated,
    so I try to give it a usefull name.
    We also create a list with the length of this lists to be able to separate the assimilated vector again (can be separated using np.cumsum(lengths) and np.split
    """

    extra_list=[]
    names=[] #saving latitudes, site names, ..
    lengths=[]
    names_short=[] #to keep track of what the values in length stand for
    
    #if c.extra_assi['proxy_estimate']==True:
    #proxy estimates always needed
    extra_list.append(HXfull_all_fin.transpose('time',...).values)
    names.append(HXfull_all_fin.site.values.tolist())
    for o,i in enumerate(HXfull_all_fin.attrs['DB_members']):
        lengths.append(i)
        string_i='DB_'+str(o)
        names_short.append(string_i)

    if c.extra_assi['globalmean'] is not None:
        for v in c.extra_assi['globalmean']: 
            data=globalmean(prior[v],str(v))
            data=data.expand_dims('site',axis=-1)
            extra_list.append(data.values)
            names.append([data.name])
            lengths.append(1)    
            names_short.append(('gm_'+str(v)))

    if c.extra_assi['lat_mean'] is not None:   
        for v in c.extra_assi['lat_mean']:

            data=lat_mean(prior[v],str(v)).transpose('time','lat')
            data=data.rename(dict(lat='site'))
            data['site']=['lm_'+str(v)+'_'+str(lon) for lon in data.site.values ]
            
            extra_list.append(data.values)
            names.append(data.site.values.tolist())
            lengths.append(len(data.site))
            names_short.append(('lm_'+str(v)))

    if c.extra_assi['regionalmean'] is not None:     
        for v in c.extra_assi['regionalmean']:
            lats=c.extra_assi['region'][0]
            lons=c.extra_assi['region'][1]
            name=str(v)+str(c.extra_assi['region'])
            
            data=regionalmean(prior[v],lats,lons,name)
            data=data.expand_dims('site',axis=-1)

            extra_list.append(data.values)
            names.append([data.name])
                
            names_short.append(('reg_'+str(v)))
            lengths.append(1)
    
    if c.extra_assi['prec_weighted']==True:
        d18_prec=psm_pseudoproxy.prec_weighting(prior_raw['d18O'],prior_raw['prec'])
        d18_prec=d18_prec.stack(site=('lat','lon'))

        d18_prec['site']=['d18_precweight_'+'_'+str(s) for s in d18_prec.site.values ]

        extra_list.append(d18_prec)
        names.append(d18_prec.site.values)
        lengths.append(len(d18_prec.site))
        names_short.append(('pr_w_d18'))
    
    if len(lengths)>0:
        #import pdb
        #pdb.set_trace()
        
        extra_list=np.concatenate(extra_list,axis=-1)
        names=np.concatenate(names)
    
    return extra_list,names,lengths,names_short
    

def prior_block_mme(prior,bs,idx_list):
    """
    27.09.22: This is the code I finally use for block creation
    
    Create a Prior-Block, which extends the randomized yearly prior to <bs> continous years
    along the third dimension. In the end I don't need numba.
    Input: 
        - Prior: List of ( Flattened version (time, #gridboxes/values) <- Length of second axis important.) for each model
        - bs: #blocksize
        - idx: random indices, Array of shape(num_models, reps, nens)
    Output:
        - block (bs,nens, #values)
        
    #UPDATE 21.09.22: adapted to multi-model prior. Also runs when just using one single model
    """
    nens=idx_list.shape[-1]
    num_models=idx_list.shape[0]
    stack_size=prior[0].shape[1]
    
    block=np.empty((bs,nens*num_models,stack_size))
    for n in range(num_models):
        idx=idx_list[n]
        mi=nens*n
        ma=nens*(n+1)
        for i in range(bs):
            block[i,mi:ma]=prior[n][idx+i,:]
    return block




@njit(parallel=True,fastmath=True)
def prior_block(nens,bs,idx_list,prior):
    """
    Create a Prior-Block, which extends the randomized yearly prior to <bs> continous years
    along the third dimension.
    Input: 
        - Prior: List of ( Flattened version (time, #gridboxes/values) <- Length of second axis important.) for each model
        - nens: ensemble memberers bs: #blocksize
        - idx: Array of shape(num_models, reps, nens)
    Output:
        - block (bs,nens, #values
        
    #UPDATE 21.09.22: adapted to multi-model prior. Also runs when just using one single model
    """
    num_models=idx_list.shape[0]
    stack_size=prior[0].shape[1]
    block=np.empty((bs,nens*num_models,stack_size))
    for n in prange(num_models):
        current_vals=prior[n]
        idx=idx_list[0]
        n_j=nens*n
        for i in prange(bs):
            #for j in prange(nens):
            #M=MC+i
            for k in prange(stack_size):
                for j in prange(nens):
                    block[i,n_j+j,k]=current_vals[idx[j]+i,k]
    return block

def prior_block_nonumba(prior,nens,stack_size,bs,idx):
    block=np.empty((bs,nens,stack_size))
    for i in range(bs):
        block[i]=prior[idx+i,:]
    return block

#AVERAGE AND MEAN CALCULATOR
@njit(parallel=True)
def anomean_with_numba(array_in,size):
    """
    Calculate anomaly and mean of array_in along axis0.
    
    Input:
        Array_in: (blocksize, values_vector[1]*nens) (Is not changed
        size: subblock size
    The second dimension of array_in is usually ~10^7, so array_in-array_in.mean(axis=0) is usually too slow,
    that's why I had to come up with this numba solution.
    """
    s=array_in.shape
    number=s[0]//size
    #Initialize array for mean and anomaly (latter has original size)
    mean=np.empty((number,s[1]))
    anom=np.empty_like(array_in)
    #looping over blocks if size!=s[0]
    if number>1:
        #loop over all points along axis1
        for i in prange(s[1]):
            for j in prange(number):
                vals=array_in[j*size:(j+1)*size,i]
                m=vals.mean()
                mean[j,i]=m
                anom[j*size:(j+1)*size,i]=vals-m
    else:
        for i in prange(s[1]):
            vals=array_in[:,i]
            m=vals.mean()
            mean[0,i]=m
            anom[:,i]=vals-m        
    return mean,anom

@njit(parallel=True)
def mean_plus_anoms(mean,anom):
    """
    Numba solution for adding the mean back to the annomalies
    Input:
        - Mean: (Nens, Values)
        - Annom (Bs, Nens, Values)
    Output:
        (bs,nens,values)
    On Gryffindor the time-advantage might not be so large as on Ravenclaw (there 2-5).
    
    Without this it would just be: mean + anom (broadcasting is automatico
    
    """
    #copy anom shape, no deepcopy as this is slow
    new_ensemble=np.empty_like(anom)
    bs,nens,nvals=anom.shape
    for i in prange(bs):
        for j in prange(nens):
            #mean_j=mean[j] #this is not necessary for numba
            for k in prange(nvals):
                new_ensemble[i,j,k]=mean[j,k]+anom[i,j,k]
    return new_ensemble

def globalmean(field,name=''):
    """
    Function that calculates the global mean of a climate field 
    (DataArray from one variable, not all variables)
    Does latitude weighting.

    Input:
        - Climate Field (time,lat,lon): GMT computed over lat,lon
        - variable name as string, (needed for naming)
    Output:
        - Global mean as DataArray with name "globalmean_+<name>"
        
    Other indices (I do not have ocean variables for that)
    - El -Nino index: Sea Surface Temperature(5°N - 5°S), 120° - 170° West (190-240° in 0-360) mode
    (https://en.wikipedia.org/wiki/Multivariate_ENSO_index) (I don't have SST)
    - AMOC index: usually defined as the stream function for the zonally integrated meridional volume transport in the ocean
    (at some specific latitude). Steiger 2016: (defined here as the maximum value of the overturn-
    ing stream function in the North Atlantic between 25 and
    70◦ N and between depths of 500 and 2000 m
    - Convert -180,180 to 0,360 -> lon % 360
    """ 
    lat=field.lat
    wgt=np.cos(np.deg2rad(lat))
    field_m=field.weighted(wgt).mean(('lat','lon'))
    field_m=field_m.rename(('gm_'+name))
    return field_m

def lat_mean(field,name):
    """
    Function that calculates the latitudinal mean of a climate field.
    """ 
    lat=field.lat
    field_m=field.mean('lon',skipna=True)
    field_m=field_m.rename(('lm_'+name))
    return field_m


def regionalmean(field,lats,lons,name):
    """
    Function that calculates the regional mean of a climate field. (Eg nino index if you have SST data).
    Does latitudinal weight averageing. Also computes correct average for cross zero meridan regions (e.g 350,10)

    Input:
        - Climate Field (time,lat,lon): GMT computed over lat,lon for given restructions
        - lat =[latSouth,latNorth], lon =[latWest,lonWest] Limits of the Region of interest.
        Given in -90 to 90 and 0 to 360 degrees.
        - variable name

    Maybe in the future: Non-rectangular regions (there I would need to explicitely specify all the gridpoints I guess)
    """
    lat=field.lat
    lon=field.lon
    
    #selection of longitudes
    if lons[0]<lons[1]:
        sel_lat = lat.where( (lat >= lats[0] ) & (lat <= lats[1]), drop=True)
        sel_lon = lon.where( (lon >= lons[0] ) & (lon <= lons[1]), drop=True)
    #cross zero-meridian region option
    else:
        sel_lat = lat.where( (lat >= lats[0] ) & (lat <= lats[1]), drop=True)
        sel_lon_1 = lon.where( (lon >= lons[0] ), drop=True)
        sel_lon_2 = lon.where((lon <= lons[1]), drop=True)
        sel_lon=xr.concat([sel_lon_1,sel_lon_2],dim='lon')
    
    wgt=np.cos(np.deg2rad(lat))
    field_m=field.sel(lat=sel_lat,lon=sel_lon).weighted(wgt).mean(('lat','lon'))
    field_m=field_m.rename(('regm_'+name))
    return field_m

def evaluation_saving(c, num_vars, names_short_vector, splitted_mean, splitted_std, times_list, coordinates, truth, prior, lisst, HXfull_all_fin, rank_dic, rank_dic_post, MC_idx_list, sites, prior_block,  attributes, pp_y_all,pp_r_all,split_vector,time_res,cfg):
    """
    Function where all the evaluations and saving procedures are done. Outsourced to this function to make the wrapper less cluttered.
    """
    
    #bring into list, because we loop over each variable
    ds_list=[]
    
    print('Save variables')
    
    #save the datavariables
    for j in range(num_vars):
        name=names_short_vector[j]

        save_mean=xr.DataArray(splitted_mean[j],coords=dict(time=times_list[0],site=coordinates)).unstack('site')
        save_std=xr.DataArray(splitted_std[j],coords=dict(time=times_list[0],site=coordinates)).unstack('site')

        string_m=name+'_mean'
        std_std=name+'_std'
        ds=xr.Dataset(data_vars={string_m: save_mean,std_std:save_std})

        #compute pseudoproxy metrics
        if c.ppe['use']==True and c.ppe['metrics_ppe'][0] is not None:
            print('Compute pseudoproxy metrics for ',name)
            #define a truth, and slice prior correctly in time
            truth_var=truth[name].sel(time=slice(times_list[0][0],times_list[0][-1]))

            for metric in c.ppe['metrics_ppe']:
                if metric in ['Corr','p_value','p_bootstrap','eff_p_value','CE','RMSE','RMSPE','MAE','MAPE']:
                    
                    #eval makes metric executable
                    metric_func='evaluation.'+metric
                    try:
                        result=eval(metric_func)(truth_var,save_mean)
                        metric_name=name+'_'+metric
                        ds[metric_name]=(('lat','lon'),result.values)
                    except:
                        #import pdb
                        #pdb.set_trace()
                        print('metric',metric, ' could not be computed')
                        
                elif metric=='RE':
                    #requires uninformed prior (truth)
                    uninformed_prior=prior[name] #will compute mean over this prior
                    result=evaluation.RE(truth,save_mean,uninformed_prior)
                    ds[(name+'_'+str(metric))]=(('lat','lon'),result.values)
        ds_list.append(ds)
    
    #non pseudoproxy_metrics
    for metric in c.metrics:        
        if metric=='CD':
            #COMPUTE CORRELATION OF MODELED PROXY ENSEMBLE (of proxy var) TO BACKGROUND CLIMATE FIELD (full)
            #this is more or less the correlation that is used for reconstructing (but not exactly, as for each timestep the proxy availabiltiy varies)
            #+ we have a different prior. Keep this in mind when interpreting this.
            #Computed for each timescale separately
            #we create a separate DataArray for that because things get large
            corr_ds=xr.Dataset()
            for name in c.reconstruct:

                Xf=prior[name].stack(z=('lat','lon'))

                #loop over proxy time_res
                for i_tres,tres in enumerate(time_res):

                    #OLD: extract available proxies at that resolution (lisst[i_tres].site)
                    #NEW: Compute covariance-distance for all proxy locationscd

                    #proxy_estimates=HXfull_all_fin.sel(site=lisst[i_tres].site)
                    proxy_estimates=HXfull_all_fin
                    proxy_lats=proxy_estimates.lat
                    proxy_lons=proxy_estimates.lon

                    #resample to that time_series
                    proxy_estimates,_= anomean_with_numba(proxy_estimates.values,tres)
                    #convert proxy_estimates to DataArray, as this is needed by CD-function
                    proxy_estimates=xr.DataArray(proxy_estimates,dims=('time','site'))

                    #get amount of proxies per db to split it
                    split_idx=np.cumsum(lisst[i_tres].attrs['DB_members'])[:-1]
                    splitted=np.split(proxy_estimates,split_idx,axis=-1)

                    splitted_lon=np.split(proxy_lons,split_idx,axis=-1)
                    splitted_lat=np.split(proxy_lats,split_idx,axis=-1)

                    step=str(tres)+'YS'
                    XF_resampled=Xf.resample(time=step,closed='left').mean()

                    for i_db, db_vals in enumerate(splitted):
                        cor_mean,cor_std,dist=evaluation.CD(XF_resampled,db_vals,splitted_lat[i_db],splitted_lon[i_db])
                        #CD stands for covariance distance
                        name_str=name+'_DB_'+str(i_db)+'_'+str(tres)
                        corr_ds[(name_str+'_m')]=(('dists'),cor_mean)
                        corr_ds[(name_str+'_s')]=(('dists'),cor_std)

            #fill distances only once, always the same
            corr_ds=corr_ds.assign_coords(dists=dist)

            #SAVE CORR_DS
            #add variable name to experiment path and if it is missing '.nc'
            path2=c.output_file+'_covdist'+'.nc'
            #increment path name by number if it already exits
            path2=dataloader.checkfile(os.path.join(base_path,path2))
            
            corr_ds.to_netcdf(path=path2)
            #CD FOR GLOBALMEANTEMP not implemented, but you can easily compute it afterwards (prior-calc, gmt-calc, MC_idx + selection, Saved_prior)

        #for rank histogram compute sum over all rank histograms (doesn't matter if sum or mean)
        elif metric=='rank_histogram':
            #loop over time_res
            for tres in time_res:
                ranks=np.sum(rank_dic[str(tres)],axis=0)
                string='rank_histogram_'+str(tres)
                ds[string]=(('ranks'),ranks)

        elif metric=='rank_histogram_posterior':
            #loop over time_res
            for tres in time_res:
                ranks=np.sum(rank_dic_post[str(tres)],axis=0)
                string='rank_histogram_post'+str(tres)
                ds[string]=(('ranks'),ranks)

    ds=xr.merge(ds_list)        
    #ds=ds.assign_coords(MC_idx=(('reps','nens'),MC_idx))    
    ds=ds.assign_coords(MC_idx=(('model','reps','nens'),MC_idx_list))    

    #Proxy estimates are always saved.
    ds['site']=sites

    proxies=np.concatenate(splitted_mean[num_vars:(num_vars+len(c.obsdata))],axis=-1)
    proxies_std=np.concatenate(splitted_mean[num_vars:(num_vars+len(c.obsdata))],axis=-1)

    ds['HXf_m']=(('time','site'),proxies)
    ds['HXf_std']=(('time','site'),proxies_std)

    #also save the prior estimate
    if num_vars>0:
        proxy_start=split_vector[len(c.reconstruct)-1]
        try:
            proxy_end=split_vector[len(c.reconstruct)+len(c.obsdata)]-1
        except:
            proxy_end=None
    else:
        proxy_start=0
        try:
            proxy_end=split_vector[len(c.obsdata)]-1
        except:
            proxy_end=None

    proxy_block=prior_block[:,:,proxy_start:proxy_end]
    proxy_prior_m=proxy_block.mean(axis=1)
    proxy_prior_std=proxy_block.std(axis=1)

    ds['HXf_prior_m']=(('bs','site'),proxy_prior_m)
    ds['HXf_prior_std']=(('bs','site'),proxy_prior_std)


    #counter for looping through the assimilated vector
    counter=num_vars+len(c.obsdata)  

    #global means
    c_global=c.extra_assi['globalmean'] 
    if c_global is not None and len(c_global)>0:
        for i in range(len(c_global)):
            name=names_short_vector[counter+i]
            ds[name]=(('time'),splitted_mean[counter+i].squeeze())
            ds[(name+'_std')]=(('time'),splitted_std[counter+i].squeeze())

        counter=counter+len(c_global)

    #latitudinal mean
    c_lat=c.extra_assi['lat_mean']
    if c_lat is not None and len(c_lat)>0:
        for i in range(len(c_lat)):
            name=names_short_vector[counter+i]
            ds[name]=(('time','lat'),splitted_mean[counter+i])
            ds[(name+'_std')]=(('time','lat'),splitted_std[counter+i])

        counter=counter+len(c_lat)

    #regional mean
    reg_vars=c.extra_assi['regionalmean']
    if reg_vars is not None and len(reg_vars)>0:
        for i in range(len(reg_vars)):
            name=names_short_vector[counter+i]
            ds[name]=(('time'),splitted_mean[counter+i].squeeze())
            ds[(name+'_std')]=(('time'),splitted_std[counter+i].squeeze())

        counter=counter+len(reg_vars)  

    #prec_weighted
    if c.extra_assi['prec_weighted']:
        ds[(names_short_vector[-1]+'mean')]=xr.DataArray(splitted_mean[-1],coords=dict(time=times_list[0],site=coordinates)).unstack('site')
        ds[(names_short_vector[-1]+'std')]=xr.DataArray(splitted_std[-1],coords=dict(time=times_list[0],site=coordinates)).unstack('site')

    #add the proxies at the different resolutions to the Dataset
    for i,l in enumerate(lisst):
        num=time_res[i]
        string_a='proxies_res_'+str(num)
        time_dim=('time_res_'+str(num))
        site_dim=('site_'+str(num))
        proxies_at_res=xr.DataArray(l.values,coords={time_dim:l.time.values,site_dim:l.site.values })        
        ds[string_a]=proxies_at_res

    #Add the original proxy values and their locations
    #first bring them together from pp_y_all
                              
    #not needed when working with pseudoproxies (already saved in the block above)
    if c.ppe['use']==False:
        pp_y_a=[]
        pp_y_lat=[]
        pp_y_lon=[]
        for pp_y in pp_y_all:
            pp_y_a.append(pp_y.values)
            pp_y_lat.append(pp_y.lat.values)
            pp_y_lon.append(pp_y.lon.values)

        pp_r_a=[]
        for pp_r in pp_r_all:
            pp_r_a.append(pp_r.values)    

        pp_r_a=np.concatenate(pp_r_a,axis=-1)
        pp_y_a=np.concatenate(pp_y_a,axis=-1)
        pp_y_lat=np.concatenate(pp_y_lat,axis=-1)
        pp_y_lon=np.concatenate(pp_y_lon,axis=-1)

        ds['proxies']=(('time','site'),pp_y_a)
        ds['proxies_r']=(('time','site'),pp_r_a)

        ds['proxies_lat']=(('site'),pp_y_lat)
        ds['proxies_lon']=(('site'),pp_y_lon)

    #add back attributes
    ds.attrs['prior']=str(attributes)
    #save config dictionary as one big string
    ds.attrs['cfg']=str(cfg)
    
    return ds

###########################################################################################

def anomaly(c,HXfull_all,pp_y_all,prior):
    """
    OLD!  I splitted this into two separate functions for proxies and the rest.
    
    Given the option in c.annomaly: ('BG','BG+HX','HX')
    When applying to BG:
        Apply to prior (all variables)
    When applying to (BG+HX,HX): 
        For pp_y_all:
            Apply annomaly correction for each timescale
        
    To-Do: 
        - For Precipitation, some studies don't remove the mean but divide by it. Further investigate that.
    """
    prior_a=prior.copy(deep=True)
    HXfull_all_a=HXfull_all.copy(deep=True)
    pp_y_all_a=pp_y_all.copy()
    
    def proxies_anomaly(proxy_list,start=None,end=None):
        pp_y_list=[]
        if start is not None:
            for pp in pp_y_all_a:
                m_p=pp.sel(time=slice(start,end)).mean('time',skipna=True)
                pp_y_list.append(pp-m_p)
        else:
            for pp in pp_y_all_a:
                m_p=pp.mean('time',skipna=True)
                pp_y_list.append(pp-m_p)
        
        return pp_y_list

    
    print('COMPUTE ANOMALIES IN MODE:',c.anomaly)
    #compute annomaly with respect to mean
    #only background/prior
    if isinstance(c.anomaly_time,list):
        start=c.anomaly_time[0]
        end=c.anomaly_time[1]
    
    if c.anomaly=='BG':
        if isinstance(c.anomaly_time,list):
            mean=prior_a.sel(time=slice(start,end)).mean('time',skipna=True)
        else:
            mean=prior_a.mean('time')
        prior_a=prior_a-mean
    #background / prior proxy estimates (and also the proxies!)
    elif (c.anomaly=='HX') | (c.anomaly=='BG+HX') : 
        if isinstance(c.anomaly_time,list):
            
            m_H=HXfull_all_a.sel(time=slice(start,end)).mean('time',skipna=True)
            HXfull_all_a=HXfull_all_a-m_H
            
            pp_y_all_a=proxies_anomaly(pp_y_all_a,start,end)
            
            if c.anomaly=='BG+HX':
                mean=prior_a.sel(time=slice(start,end)).mean('time',skipna=True)
                prior_a=prior_a-mean
                
        else:
            m_H=HXfull_all_a.mean('time',skipna=True)
            HXfull_all_a=HXfull_all_a-m_H

            pp_y_all_a=pp_y_all_a=proxies_anomaly(pp_y_all_a)
        
            if c.anomaly=='BG+HX':
                mean=prior_a.mean('time',skipna=True)
                prior_a=prior_a-mean
    
    return HXfull_all_a,pp_y_all_a,prior_a

def inf_fac(Y,HX):
    """
    Calculation of best inflation factor (not used currently, probably not useful).
    substracting square root of R is necessary to take into account the noise of the proxies.
    """
    nom=((Y.values-HX.mean(dim='time').values)**2).sum()-np.sqrt(R).sum()
    denom=HX.var('time').sum()
    r=(nom/denom).values
    if r < 1 :
        r=1
    return np.sqrt(r)#*1.05#np.sqrt(r)
