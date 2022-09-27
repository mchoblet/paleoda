import numpy as np
import xarray as xr
import pandas as pd
import cftime
import pickle

"""
The sisal database is already brought into the correct form (sisal_da.nc).
look at proxy_loader_testing Notebook to see how it is constructed.

Functions for constructing the table from iso 2k data are also saved here.
"""

def read_iso_pkl(path):
    """
    Input: path to pkl-File
    Output: Selected parts of file (glacier ice, d18O, permil, primary timeseries (or missing))
    """
    pickles = open(path,"rb")
    pTS=pickle.load(pickles)

    TS= pTS['TS']
    
    # function to extract data from structure 'TS'

    pullTsVariable = lambda ts, var: ts[var] if var in ts else 'missing'
    
    variableName = np.array(list(map(pullTsVariable,TS,['paleoData_variableName'] * len(TS))))

    # define units for each data series

    units = np.array(list(map(pullTsVariable,TS,['paleoData_units'] * len(TS))))

    # is the timeseries a primary time series for this site? pull only those records which are.

    primary = np.array(list(map(pullTsVariable,TS,['paleoData_iso2kPrimaryTimeseries'] * len(TS))))

    isd18O = np.where(variableName == 'd18O')[0]
    #isd2H = np.where(variableName == 'd2H')
    isPermil = np.where(units == 'permil')[0]
    isPrimary = np.where(primary == 'TRUE')[0]
    isPrimary_or_missing = np.where((primary == 'missing') | (primary == 'TRUE'))[0]
    
    # pull records which report d18O and dD in units of permil

    #d18Oin  = np.intersect1d(np.intersect1d(isPermil,isd18O),isPrimary)
    #d2Hin  = np.intersect1d(np.intersect1d(isPermil,isd2H),isPrimary)

    # exclude all non-isotopic data

    #isIso = np.union1d(d2Hin,d18Oin)

    #allIsoTS = np.asarray(TS)[isIso].tolist()

    # --------------------------------------------------------------------------
    # 4. EXPLORE ISOTOPE INTERPRETATION AND DATA TYPES
    # --------------------------------------------------------------------------

    description = np.array(list(map(pullTsVariable,TS,['paleoData_description'] * len(TS))))
    #get indices where glacier ice
    glacierice=np.where(description=='glacier ice')[0]
    
    output_index=list(set(glacierice).intersection(isPermil,isPrimary_or_missing,isd18O))
    x=len(glacierice)
    y=len(output_index)
    TS_out=np.array(TS)[output_index]
    print('Dataset reduced from ',x,' glacier ice records to ', y,'records')
    return TS_out
    
def make_table(dic,start=0,end=2013):
    """
    start, end: years to be considerend
    """
    years=[]
    values=[]
    for k in dic:
        if 'year start' in k:
            L=np.abs(k['year start'][0]-k['year start'][-1])
            y=k['year start']
            years.append(y)
            
        else:
            L=np.abs(k['year'][0]-k['year'][-1])
            y=k['year']
            years.append(y)
        k['length in years']=L
        values.append(np.array(k['paleoData_values']).astype(float))

    #save smallest resolution and mean resolution in time
    min_res=[]
    mean_res=[]
        
    for y in years:
        dist=np.abs(np.array(y)[1:]-np.array(y)[:-1])
        min_res.append(dist.min())
        mean_res.append(dist.mean())
    
    #special treatment where min_value <1 -> yearly mean
    
    #save lats and lons
    lats=[k['geo_meanLat'] for k in dic]
    lons=[k['geo_meanLon'] for k in dic]
    elevs=[k['geo_meanElev'] for k in dic]
    name=[k['geo_siteName'] for k in dic]
    lengths=[k['length in years'] for k in dic]
    dataset=[k['dataSetName'] for k in dic]
    
    flat_list = [item for sublist in years for item in sublist]
    rounded=[i for i in np.round_(np.array(flat_list).astype(float)).astype(int) if i>=start and i<= end]
    years_unique=np.unique(rounded).astype(int)

    #initialize array
    empty = np.empty((len(years_unique),len(lats)))
    empty.fill(np.nan)

    ##loop over records/years
    
    for i,k in enumerate(years):
        
        #make list k to array
        k_rounded=np.array(k).astype(int)
        k_unique=np.unique(k_rounded)
        
        for l,j in enumerate(k_unique):

            #values i to np-array
            vals=np.array(values[i])
            if j>=start and j<=end:
                
                #compute mean value (relevant when resolution is higher than annual), else just one value
                mean=vals[np.where(k_rounded==int(j))[0]].mean()
                
                
                empty[int(j),i]=mean
    
    dataarray=xr.DataArray(empty,dims=('time','site'))
    dataarray['time']=np.array([cftime.datetime(year,1,1,calendar='365_day') for year in years_unique])
    dataarray['site']=np.arange(0,len(lats))
    lats=[k['geo_meanLat'] for k in dic]
    lons=[k['geo_meanLon'] for k in dic]

    dataarray['lat']=(['site'],lats)
    dataarray['lon']=(['site'],lons)
    dataarray['elev']=(['site'],elevs)
    dataarray['site name']=(['site'],name)
    dataarray['min_time_resolution']=(['site'],min_res)
    dataarray['mean_time_resolution']=(['site'],mean_res)
    dataarray['length_of_record']=(['site'],lengths)
    #number of records actually counted only in cut_time function
    dataarray['number_of_records']=(['site'],lengths)
    
    return dataarray


def cut_time(dataarray,startyear,endyear):
    da=dataarray.sel(time=slice(cftime.datetime(startyear,1,1,calendar='365_day'),cftime.datetime(endyear,1,1,calendar='365_day')))
    
    #drop sites where no datapoints      
    #if dataarray has min/mean time resolution as coordinates adapt them. but might go wrong if there is only one record -> exception
    
    
    if 'min_time_resolution' in dataarray.coords:
        for s in range(len(da.site)):
            try:
                time=da.isel(site=s).dropna('time').time.dt.year
                #return time
                #.dt.year
                #print(time.values)
                #return time
                #time=time.t.dt.year

                dist=np.abs(np.array(time)[1:]-np.array(time)[:-1])
                da['min_time_resolution'][s]=dist.min()
                da['mean_time_resolution'][s]=dist.mean()
                #length of record
                da['length_of_record'][s]=np.max(time)-np.min(time)
                da['number_of_records'][s]=len(time)
            except:
                pass
            
    return da

def drop_empty(da):
    for s in da.site:
        data=da.sel(site=s)
        count=np.count_nonzero(~np.isnan(data))
        if count < 1:
            da=da.drop_sel(site=s)
    return da
    
def minimum_of_records(da,num,timeslice=None):
    """
    Function that drops sites from dataarray for which there are not at least 'num' records for
    time_slice in which we want to have at least 'num' records
    
    da: must have dimesion 'site'
    num: minimum number of records
    timeslice: None: (beginning to end), or tuple of (startyear,endyear)
    """
    
    #we need a cutted array where we count the records, but we drop records in the original (uncut) one
    if timeslice !=None:
        da_cut=cut_time(da,timeslice[0],timeslice[1])
    else:
        #change nothing
        da_cut=da
    
    for s in da_cut.site:
        data=da_cut.sel(site=s)
        count=len(data.dropna('time').time)#np.count_nonzero(~np.isnan(data))
        if count < num:
            da=da.drop_sel(site=s)
    
    return da

def read_iso_pkl(path):
    """
    Input: path to pkl-File
    Output: Selected parts of file (glacier ice, d18O, permil, primary timeseries (or missing))
    """
    pickles = open(path,"rb")
    pTS=pickle.load(pickles)

    TS= pTS['TS']
    
    # function to extract data from structure 'TS'

    pullTsVariable = lambda ts, var: ts[var] if var in ts else 'missing'
    
    variableName = np.array(list(map(pullTsVariable,TS,['paleoData_variableName'] * len(TS))))

    # define units for each data series

    units = np.array(list(map(pullTsVariable,TS,['paleoData_units'] * len(TS))))

    # is the timeseries a primary time series for this site? pull only those records which are.

    primary = np.array(list(map(pullTsVariable,TS,['paleoData_iso2kPrimaryTimeseries'] * len(TS))))

    isd18O = np.where(variableName == 'd18O')[0]
    #isd2H = np.where(variableName == 'd2H')
    isPermil = np.where(units == 'permil')[0]
    isPrimary = np.where(primary == 'TRUE')[0]
    isPrimary_or_missing = np.where((primary == 'missing') | (primary == 'TRUE'))[0]
    
    # pull records which report d18O and dD in units of permil

    #d18Oin  = np.intersect1d(np.intersect1d(isPermil,isd18O),isPrimary)
    #d2Hin  = np.intersect1d(np.intersect1d(isPermil,isd2H),isPrimary)

    # exclude all non-isotopic data

    #isIso = np.union1d(d2Hin,d18Oin)

    #allIsoTS = np.asarray(TS)[isIso].tolist()

    # --------------------------------------------------------------------------
    # 4. EXPLORE ISOTOPE INTERPRETATION AND DATA TYPES
    # --------------------------------------------------------------------------

    description = np.array(list(map(pullTsVariable,TS,['paleoData_description'] * len(TS))))
    #get indices where glacier ice
    glacierice=np.where(description=='glacier ice')[0]
    
    output_index=list(set(glacierice).intersection(isPermil,isPrimary_or_missing,isd18O))
    x=len(glacierice)
    y=len(output_index)
    TS_out=np.array(TS)[output_index]
    print('Dataset reduced from ',x,' glacier ice records to ', y,'records')
    return TS_out


############## UNUSED CODE


def coarsen_time(xarray,time_res):
    """
    Coarsen the time resolution of xarray-data using the resample method on a yearly time scale (binning values).
    If multiple values are available for one location in a time bin take the mean value
    
    input:
        - xarray with dimension time (monotonically increasing!)
        - time_res: positive integer
    """
    assert isinstance(xarray,xr.core.dataset.Dataset) | isinstance(xarray,xr.core.dataarray.DataArray), "Input must me xarray or dataset"
    assert isinstance(time_res,int) and time_res>0, "time_res must be integer"
    
    time_res=str(time_res)+'AS'
    resampled=xarray.resample(time=time_res).mean()
    
    return resampled

def lat_lon_multiindex(xarray):
    """
    X-Array needs coordinates lat and lon
    """
    lat=xarray.lat.values
    lon=xarray.lon.values
    coord_tuples=list(zip(lat,lon))

    multi=pd.MultiIndex.from_tuples(coord_tuples)
    xarray=xarray.assign_coords(loc=('site',multi))
    xarray.attrs['location coordinates']='(lat,lon)'
    return xarray

