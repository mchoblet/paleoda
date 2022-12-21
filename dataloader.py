import numpy as np
import xarray as xr
import cftime
import pandas as pd
import os


def load_prior(path,var,conversion=True):
    """
    Loads prior Data, renames latitude/longitude/time dime.
    Doesn't check if all dims that are needed are actually contained!
    If Longitudes go from -180 to 180 they are changed to 0-360

    Input: Path to nc-File, Variable
    conversion=True only relevant for precipitation (its better to have it in one consistent format, here mm/month)
    Output: DataArray
    """
    array=xr.open_dataset(path,use_cftime=True)[var]
    dims=array.dims
    if ('latitude' in dims) & ('longitude' in dims):
        array=array.rename({'latitude':'lat','longitude':'lon'})
    if ('t' in dims) & ('time' not in dims):
        array=array.rename({'t':'time'})
    if array.lon.min() < -10:
    	#convert lon from -180-180 to 0-360
    	array.coords['lon'] = array.coords['lon'] % 360
    	array= array.sortby(array.lon)
    
    #if we are dealing with precipitation and conversion is True convert it to mm/month
    #also do the same for evaporation!
    if var in ['pr','prec','precip','precipitation','pratesfc','evap']:
        if conversion:
            array=precipitation_conv(array)
    #remove some unnecearray lev... dimension of shape 1
    array=array.squeeze(drop=True)
    
    #bring into uniform shape
    array=array.transpose('time','lat','lon')
    
    #rename
    array=array.rename(var)
    
    return array




def precipitation_conv(array):
    """
    array: xarray which has unit attribute, else
    Converts monthly precipitation data into mm/month.
    Function automatically detects calendar type and the month lenghth (relevant for kg/m^2s)
    """
    #get unit and month length
    try:
        unit=array.attrs['units']
    except:
        raise AttributeError('Precipitation does not have a unit attribute. Conversion failed!')
    
    try:
        month_length =array.time.dt.days_in_month
    except:
        raise AttributeError('Data does not contain how many days there are in one month. Crucial')

    if unit=='mm/month':
        #print('No conversion needed!')
        return array
    
    if unit=='m/s':
        #assuming 30 days per month convert to mm/month
        array=array*1000*60*60*24*month_length
    
    elif unit=='mm/day':
        array=array*month_length
    
    elif unit in ['kg/m^2s','kg/m2s','kg/m^2/s','kg/m2/s','kg m-2 s-1']:
        #kg to mm,broadcasting done correctly by xarray
        array=array*(60*60*24*month_length)
    else:
        raise NameError('Unkown precipitation units! (kg/m^2s, kg/m2s, kg/m^2/s,kg/m2/s, kg m-2 s-1,mm/day or m/s)', unit)
    #change unit attribute   
    array.attrs['units']='mm/month'
    
    return array
    

def unite(*arrays):
    """
        CURRENTLY UNUSED
        concatenate/unite DataArrays by creating new Dataset from DataArrays   
    """
    Dataset=xr.Dataset()
    for a in arrays:
        #print(a.name)
        Dataset[a.name]=a
    return Dataset
    
def annual_mean(xarray,avg=None,check_nan=False):
    """
    Compute annual means from monthly input data. 
    Monthly data should be continous (no months missing), else for non-None avg parameter thinks might get weird.

    Input:
    - xarray: DataArray (can be 2D field or stacked to 1D, just needs time coordinate)
    - avg: 
        - None: means regular january to december mean
        - integer: month to start from (e.g 4, if you want)
        - list/array: months to be used (e.g [4,5,6], [11,12,1], ...). Make sure this months make sense and
          are in ascending order (11,12,1 is allowed).
    - check_nan:
        - removes nan-years. This is relevant for a model like iHadCM3 where there is missing data in between. I kept these years in the prior, because else the annual mean calculation as performed here doesn't work anymore.
        - Update 04.10.22: Do not remove, but copy the previous year. Workaround to not mess up Pseudoproxy calculation (prelimary workaround, year 1426,1427,1428 become equal to year 1425)
        
    Output:
    - Reduced Xarray, years that are not full are cut out
    """
    
    #take a deep copy of input array (else memory problems as input is manipulated in place)
    xarray=xarray.copy()
    
    #copy initial attributes
    xarray_time_copy=xarray.time
    xarray_attrs_copy=xarray.attrs
    name=xarray.name

    if avg==None:
        
        # Masking nan values
        cond = xarray.isnull()
        ones = xr.where(cond, 0.0, 1.0)

        #days per month
        month_length = xarray.time.dt.days_in_month
        #multiply ones array with ones (kind of a masking)
        month_length = month_length * ones

        #transform month-length to weights depending on number of days in year
        # Calculate the weights
        wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum('time')

        #calculate mean
        # Calculate the numerator #here resample so that we correctly get out time dimension
        xarray = (xarray * wgts).resample(time="AS").sum(dim='time')
        

        #check beginning/end, full year?    
        fm=xarray_time_copy.time.dt.month[0] #first month
        lm=xarray_time_copy.time.dt.month[-1] #last month
        if fm!=1 and fm!=2:
            xarray=xarray.isel(time=slice(1,None))
        if lm!=12 and lm!=11:
            xarray=xarray.isel(time=slice(None,-1))

        #add info about mean type in attributes
        xarray.attrs['Yearly mean type']='Standard January to December'
    
    #second case start month given (e.g. 4 for April to March averages)
    elif isinstance(avg,int): 

        #Slice the start
        #get first year/month
        fm=xarray.time.dt.month[0].values
        years=np.unique(xarray.time.dt.year.values)

        #first month in Array smaller than start month: slice from avg-month in first year
        #slicing on indices (assumes that no month is missing), else I would need to create a cftime.Datetime-object where I would also need to know the calendar type.
        #cal=xarray.time.to_index().calendar
        if fm<=avg:
            xarray=xarray.isel(time=slice(int(avg-fm),None))
        else:
            xarray=xarray.isel(time=slice(int(12-fm+avg),None))

        #Slice the End
        #last available month, needs to be==avg-1 (special case january=1 -> lm==12)
        lm=xarray.time.dt.month[-1].values
        #slice in last year
        if lm>=avg:
            xarray=xarray.isel(time=slice(None,int(avg-lm-1)))
        else:
            xarray=xarray.isel(time=slice(None,int(avg-lm-12-1)))

        # Masking nan values
        cond = xarray.isnull()
        ones = xr.where(cond, 0.0, 1.0)

        #days per month
        month_length = xarray.time.dt.days_in_month
        #multiply ones array with ones (kind of a masking)
        month_length = month_length * ones

        #weight month length by year length
        mw= month_length.resample(time="12M",closed='left',label='left').map(lambda a: a/a.sum('time'))
        #apply weighting
        xarray=(xarray*mw).resample(time="12M",closed='left',label='left').sum()

        #add info about mean type in attributes
        xarray.attrs['Yearly mean type']='Yearly mean starting in month:'+str(avg)
        
    if isinstance(avg,np.ndarray) | isinstance(avg,list):
        """
        calculate average for months given in list (e.g [4,5,6], [11,12,1], ...)
        WARNING: If a month is just not existent, this will go wrong (just using coarsen method)
        ENHANCEMENT: find a better way to do this
        """
        assert np.ndim(np.squeeze(avg))==1, "<avg> invalid, can be None, integer (1-12) or 1-d list/nd.array"
        #make sure avg is a list
        avg=list(avg)

        #Time-Slice
        #BEGINNING
        fm=xarray.time.dt.month[0]
        years=np.unique(xarray.time.dt.year.values)
        #start month in avg sequence
        sm=avg[0]
        #is start month included in first year?
        #cal=xarray.time.to_index().calendar

        #first month in Array smaller than start month: slice from avg-month in first year
        if fm<=sm:
            xarray=xarray.isel(time=slice(int(sm-fm),None))
        #slice in second year
        else:
            xarray=xarray.isel(time=slice(int(12-fm+sm),None))

        #Slice the End
        #last available month, needs to be==avg-1
        lm=xarray.time.dt.month[-1].values
        #month to end with
        em=avg[-1]
        #slice in last year
        if lm>em:
            #if last month in prior larger than end month in avg slice from the end
            xarray=xarray.isel(time=slice(None,int(em-lm)))
        elif lm<em:
            #else go one year further back
            xarray=xarray.isel(time=slice(None,int(em-lm-12)))

        #### Select months
        xarray=xarray.where(xarray.time.dt.month.isin(avg), drop=True)

        # Masking nan values
        cond = xarray.isnull()
        ones = xr.where(cond, 0.0, 1.0)

        #days per month weighting.  
        #Ugly workaround, not so x-array-esque, but the reduce method didn't work with my custom function
        #as the map function did not work, this is the only way I got it to work without a list comprehension (maybe slow)
        month_length = xarray.time.dt.days_in_month
        month_coarse=(ones*month_length).coarsen(time=len(avg)).construct(time=('year','month'))
        month_sum=month_coarse.sum('month')
        #mw=month_coarse/month_sum.values[:,None,:,:]
        #assure format and weight
        mw=month_coarse.transpose('year','month',...)/month_sum.transpose('year',...).values[:,None]
        #get in good shape for multiplying with xarray
        mw=mw.stack(time=('year','month')) #(mw.sizes['year']*mw.sizes['month'],mw.sizes['lat'],mw.sizes['lon'])
        mw=mw.transpose('time',...)
        #the month label is the middle month (coarsen has no label='left' or 'right' option)
        xarray=(xarray*mw.values).coarsen(time=len(avg)).sum('time')

        #add info about mean type in attributes
        xarray.attrs['Yearly mean type']='Yearly mean based on months:'+str(avg)
   
    #add year index 
    xarray=xarray.assign_coords(year=('time',xarray.time.dt.year.values))
    #add original attributes (get lost throughout the process)
    xarray.attrs=xarray_attrs_copy
    xarray=xarray.rename(name)
    
    #drop years where a nan is included (this would break the code). This takes some time therefore only an option.
    #update: copy previous year
    if check_nan:
        print('Checking prior for nans')
        for i,t in enumerate(xarray.time):
            x=xarray.sel(time=t)
            nans=np.count_nonzero(np.isnan(x))
            if nans>0:
                #print('Dropped year', t.values, 'due to nans')
                #xarray=xarray.where(xarray.time!=t, drop=True)
                print('Only nans in year', t.values, '. Replaced values with previous year')
                xarray.loc[dict(time=t)]=xarray.isel(time=(i-1))
   
    return xarray


def random_indices(nmem,length,reps=1,seed=None):
    """
    Produces array of random indices (for prior selection in Monte Carlo approach or proxy randomization)
    
    Input:
        nmem: how many members of length-members are randomly selected
        length: length of initial array (e.g. 1000 for full prior, or 163 for sisal database)
        reps: repetitions
        seed: Integer (Number you can set to make reproducable results (works over loops))
    
    Output: [reps,nmem] 2d nparray (loop over first index to get the members)
    """
    rng=np.random.default_rng(seed)
    array=np.zeros((reps,nmem))
    for i in range(reps):
        array[i,:]=rng.choice(length,size=nmem, replace=False)
    #if only one repetition get rid of unwanted extra dim <- No, need it for the wrapper!
    #array=array.squeeze()
    return array.astype(int)
    
def checkfile(path):
    #adapted from https://stackoverflow.com/questions/29682971/auto-increment-file-name-python
    path=os.path.expanduser(path)
    
    if not os.path.exists(path):
        print('Output written to {}'.format(os.path.basename(path)))
        return path

    root, ext = os.path.splitext(os.path.expanduser(path))
    dir       = os.path.dirname(root)
    fname     = os.path.basename(root)
    candidate = fname+ext
    index     = 0
    ls        = set(os.listdir(dir))
    while candidate in ls:
             candidate = "{}_{}{}".format(fname,index,ext)
             index    += 1
    new_path=os.path.join(dir,candidate)
    print('Warning: Output written to {}, because {} already exists'.format(os.path.basename(new_path),os.path.basename(path)))
    return new_path


################### BELOW: USEFUL CODE WHICH IS NOT USED CURRENTLY

def clm_monthly(da,start,end):
    """
    CURRENTLY UNUSED
    Compute mean climatology for each month in [start,end]-year-range
    Input:
        da: Dataarray/Dataset (Monthly!)
        time: e.g [1800,1850]
    """
    clm=da[(da['time.year']>=start) & (da['time.year']<=end)].groupby('time.month').mean(dim='time')
    return clm

def clm_annual(da,start,end):
    """
    CURRENTLY UNUSED
    Compute mean climatology in [start,end]-year range
    Input:
        da: Dataarray/Dataset
        time: e.g [1800,1850]
    """
    clm=da[(da['time.year']>=start) & (da['time.year']<=end)].mean(dim='time')
    return clm

def anom(da,start,end,typ='m'):
    """
        CURRENTLY UNUSED
    """
    assert typ in ['m','y'], "Error. Typ needs to be 'm'(onthly) or 'y'(yearly)!"
    
    if typ=='m':
        clm=clm_monthly(da,start,end)
        out=da.groupby('time.month')-clm
    elif typ=='y':
        clm=clm_annual(da,start,end)
        out=da-clm
    return out
    
    
def replacemean(da,mean):
    """
        CURRENTLY UNUSED
        
    For each gridbox replace the monthly mean by another mean (Steiger 2018 did something like this for biascorrection)
    Input:
       - da: monthly (!) dataset
       - mean: Array with same dimension as
    """ 
    #remove monthly mean from input da-Array
    clm=da.groupby('time.month').mean('time')
    da=da.groupby('time.month')-clm
    #add mean
    out=da+mean
    return out

def detrend_time(da,deg=1):
    """
        CURRENTLY UNUSED
    """
    coef=da.polyfit(dim='time', deg=deg).polyfit_coefficients
    vals=xr.polyval(coord=da['time'],coeffs=coef)
    out=da-vals
    return out

def stack(xarray):
    """
    CURRENTLY UNUSED
    Stack xarray in order to get 2D Vector (state variables*time)
    """
    if ('lat' and 'lon') in xarray.dims:
        xarray=xarray.stack(z=('lat','lon'))
    elif ('latitude' and 'longitude') in xarray.dims:
        xarray=xarray.stack(z=('latitude','longitude'))
    else: 
        print('Error in lat/lon dimension definition, could not stack!')
    #transpose for good shape
    return xarray.T


def select_prior(da,nens,seed=None,mean_zero=False):
    """
    CURRENTLY UNUSED
    Select nens years from da to create reduced background prior    
    - Input:
        - Datarray, Dataset (annual, full)
        - Nens: size of final sample
        - seed=None
        - mean_zero=False (If you want to make sure the posterior ensemble has mean zero)
    - Output:
        - reduced Dataset, indices used are stored as coodinate
        #Unused indices are stored in attributed
    """

    rng=np.random.default_rng(seed)
    #original prior length
    l=len(da.time)
    indices=rng.choice(l, nens, replace=False)
    #select indices in full prior
    bg=da.isel(time=indices)
    #add indices to coordinates
    bg=bg.assign_coords(indices=('time',indices))
    #compute unused indices
    unused=np.setdiff1d(np.arange(l),indices)
    #store unused indices in attrs (sorted for direct further used)
    bg.attrs['unused']=np.sort(unused)
    bg.attrs['initial timelength']=len(da.time)
    
    
    #if you want a zero mean prior, make sure new mean is also zero
    if mean_zero:
        bg=bg-bg.mean(dim='time')
    
    return bg


    
