import xarray as xr
import numpy as np
from types import SimpleNamespace

import dataloader
import utils
import psm_pseudoproxy
import kalman_filters
import evaluation

import warnings
warnings.filterwarnings('ignore')


import tqdm
import os
import warnings
import copy

from numba import njit,prange,vectorize
import xskillscore as xss
import sys
import json

def paleoda_wrapper(cfg):
    #20.09.22 changed the structure a bit to bring in the multi-model-prior more easily
    #it's first loading the proxies + resampling them. then loading models, psm, annomaly correction, values_vector
    # then we create pseudoproxies and correct the proxies if we have the anomaly option.
    #that way I can easily wrap around the center part


    #CONFIG CHECK
    #check config and load into namespace
    cfg=utils.config_check(cfg)
    c=SimpleNamespace(**cfg)

    #proxy databases and errors list of DAs, final time given by c.proxy_time.
    #site has 0.+..., 1.+... to mark the distinct proxy dbs
    
    pp_y_all,pp_r_all=utils.proxy_load(c)

    #RESAMPLE PROXIES
    pp_y_all,pp_r_all,times_list,lisst,lisst_r=utils.resample_wrapper(c,pp_y_all,pp_r_all)
    #anomaly correction for proxies is done after pseudoproxies part

    #timescales to reconstruct
    if c.ppe['use']==False:
        time_res=c.timescales
    else:
        #if using pseudoproxies: I implemented the option of having different TS for each Database, to facilitate some experiments where some locations only have specif timescale information
        timescales=np.unique(np.hstack(c.ppe['multi_timescale']))
        if 1 not in timescales:
            timescales=np.insert(timescales,0,1)
        time_res=np.sort(np.array(timescales))

    #block size is largest time resolution
    bs=time_res[-1]
    #time resolutions without bs, reversed (backwards)
    time_res_r=time_res[:-1][::-1]

    if c.multi_model_prior is None:
        #set range to one such that the loop is just ran once (same as not multi-model prior)
        ran=1
    else:
        print('Will compute a multi-model-prior!')
        try:
            dicts_paths=c.multi_model_prior[0]
        except:
            dicts_paths=c.multi_model_prior
        #eventually oros not provided (and also not needed)
        try:
            oros=c.multi_model_prior[1]
        except:
            pass
        ran=len(dicts_paths)
        try:
            model_names=list(c.multi_model_prior[0].keys())
        except:
            model_names=list(c.multi_model_prior.keys())
        
    #empty list where I am going to save the values vectors!
    values_vector_list=[]
    MC_idx_list=[]
        
    for i in range(ran):
        #for multi-model prior option change the paths and the 
        if c.multi_model_prior is not None:
            current_mod=model_names[i]
            cfg_copy=copy.deepcopy(cfg)
            cfg_copy['vp']=dicts_paths[current_mod]
            try:
                cfg_copy['oro']=oros[current_mod]
            except:
                #oros probably not deeded
                pass
            
            ###
            #workaround to make sure the broken iHadCM3 years are removed
            if current_mod!='iHadCM3':
                cfg_copy['check_nan']=False
                
            elif current_mod=='iHadCM3':
                cfg_copy['check_nan']=True
            ###
            
            
            utils.config_check(cfg)
            c2=SimpleNamespace(**cfg_copy)    
        else:
            c2=c        

        #PRIOR AND PROXY LOAD
        #annual and monthly priors
        prior, attributes, prior_raw=utils.prior_preparation(c2)

        #PSM
        #apply psm. List of DAs
        HXfull_all=utils.psm_apply(c2,prior,prior_raw, pp_y_all)

        #### BRING DATA TOGETHER
        #we have assured in the beginning, that the time-axis is the same for all proxy-dbs (.reindex-method)
        #at that step we lose the proxy-db specific metadata. The meta data is the reason why we can not concatenate the xarrays directly,
        #cause we need it for the PSMs

        # 2. Bring HXfull_all together
        #loop over dbs, bring into list
        data=[]
        sites=[]
        lons=[]
        lats=[]

        
        for db in HXfull_all:
                data.append(db.values)
                sites.append(db.site.values.tolist())
                lons.append(db.lon.values.tolist())
                lats.append(db.lat.values.tolist())
                #time_l.append(i.time)

        ax=db.get_axis_num('site')
        data=np.concatenate(data,axis=ax)
        sites=np.concatenate(sites,axis=0)
        lons=np.concatenate(lons,axis=0)
        lats=np.concatenate(lats,axis=0)

        #create dataarray
        HXfull_all_fin=xr.DataArray(data,coords=dict(time=HXfull_all[0].time.values,site=sites))
        HXfull_all_fin['lat']=('site',lats)
        HXfull_all_fin['lon']=('site',lons)
        #count databasemembers and add it as an attribute
        integers=(list(map(int,list(map(float,sites)))))
        HXfull_all_fin.attrs['DB_members']=np.unique(integers,return_counts=True)[1]        

        #eventually cut the prior regionally:
        #HX has already been limited in case c.only_regional_proxies==True, else all proxy estimates are taken
        bounds=c.regional_bounds #[[latS,latN],[latW,latE]]
        if bounds!=False and bounds!=None:
            #latitudes selection
            prior=prior.where((prior.lat>= bounds[0][0] ) & (prior.lat <= bounds[0][1]), drop=True)            
            #longitude selection
            #needs special treatment (e.g 330, 30) or (0,50). 
            lons=bounds[1]
            lon=prior.lon
            if lons[1]<lons[0]:          
                sel_lon_1= lon.where( (lon >= lons[0] ), drop=True)
                sel_lon_2= lon.where( (lon <= lons[1]), drop=True)
                sel_lon=np.concatenate([sel_lon_1.values,sel_lon_2.values])
                prior=prior.where(prior.lon==sel_lon,drop=True)
            else:    
                prior=prior.where((lon >= lons[0] ) & (lon <= lons[1]),drop=True)          
        
        ###ANOMALY FOR EVERYTHING EXCEPT THE PROXIES.
        if c.anomaly is not None or c.anomaly!=False:
            HXfull_all_a,prior_a=utils.anomaly_noproxies(c,HXfull_all_fin,prior)
            #assign to original arrays
            HXfull_all_fin=HXfull_all_a
            prior=prior_a
        
        ####EXTRA ASSIMILATATED VALUES
        extra_list,names,lengths,names_short=utils.extra_assimil(c2,prior,prior_raw,HXfull_all_fin)

        #CREATING THE PRIOR VECTOR
        #we stack all the variables we want to assimilate together into one vector.
        #it is faster to assimilate all variables at once than to repeat the calculations for tsurf, prec, d18O... separately
        #This is a structure we can not easily pack into a useful xarray-DataArray (that keeps track of coordinates and stuff)
        #Therefore from that point on I store the relevant info on which part of the vector is what in some additional vectors:
        #length_vector, names_vector, names_short vector
        #cfg['reconstruct'] contains all the variables are goint to be reconstructed

        #If you are only interested into a global/regional/latitudinal/mean', set 'full_fields' in config to false.
        #This way we don't reconstruct the field, which is equivallent as long as no localization is used.

        #The proxy estimates from HXfull_all_fin are carried within that vector, because I need to use the updated proxy estimates
        #for the Multi-time-scale DA.

        #HOW TO SPLIT AGAIN AFTERWARDS
        #splitted=np.split(values_vector,split_vector,axis=-1)
        #xr.DataArray(splitted[0],coords=dict(time=time_array,site=coordinates)).unstack('site')

        if c2.full_fields:
            #stack prior variables, secure right shape
            prior_stacked=prior.stack(site=('lat','lon')).transpose('time','site')

            #save coordinates for later when remaking the prior xarray
            coordinates=prior_stacked['site']

            #store number of elements per field, coordinates and short name for each part of the vector
            length_init=np.repeat(len(coordinates),len(c.reconstruct))
            names_init=np.repeat(coordinates.values,len(c.reconstruct))
            names_short_init=list(c.reconstruct)

            #bring stacked variables into one vector and concatenate that list
            #depending on 

            values=[]
            #for v in prior_stacked:
            for v in c2.reconstruct:
                values.append(prior_stacked[v].values)
            values=np.concatenate(values,axis=-1)

            #add extra_list from previous step
            values_vector=np.concatenate([values,extra_list],axis=-1)

        else:
            #initialize lists as empty such that it fits the rest
            length_init=[]
            names_init=[]
            names_short_init=[]
            #values_vector starts with extra_list from previous step
            values_vector=extra_list
            coordinates=None
            
        names_vector=np.concatenate([names_init,names],axis=-1)
        length_vector=np.concatenate([length_init,lengths])
        names_short_vector=np.concatenate([names_short_init,names_short])
        split_vector=np.cumsum(length_vector,dtype=int)[:-1] #can be applied to values_vector with numby split


        #get position where proxy-estimates start in the vector
        if c2.full_fields:
            proxy_start=int(split_vector[len(c.reconstruct)-1])
            try:
                proxy_end=int(split_vector[(len(c.reconstruct)-1+len(c.obsdata))])
            except:
                #case that only proxy estimates saved
                proxy_end=None
        else:
            proxy_start=0
            proxy_end=int(split_vector[(len(c.obsdata))])

        #if c.multi_model_prior is not None:
            #save al values_vectors in a list. proxy_names and stuff irrelevant because its the same everytime
            #(models were regridded to a common grid)
        values_vector_list.append(values_vector)
        
        #compute separate monte carlo indices for each model (brings in more randomnes)
        MC_idx=dataloader.random_indices(c.nens,prior.time.shape[0]-bs-1,c.reps,seed=c.seed + i)
        MC_idx_list.append(MC_idx)
        
        #import pdb
        #pdb.set_trace()

    MC_idx_list=np.array(MC_idx_list)


    #finish. 

    #GENERATE PSEUDOPROXIES
    if c.ppe['use']:
        prior_ext, HXfull_all_ext,lisst,lisst_r,times_list=utils.pseudoproxy_generator(cfg,HXfull_all,pp_y_all,times_list) #Hxfull_all_ext is not used further
        if c.ppe['source']=='external':
            truth=prior_ext.copy(deep=True)
            
        else:
            truth=prior.copy(deep=True)
    else:
        truth=None

    #Create list of availables sites at each timescale
    #this is necessary for getting the right proxy estimates during the Multi-time-scale DA
    sites_avail=[]
    for l in lisst:
        sites_avail.append(l.site.values)
    
    
    
    #Get indices in values_vector for available sites at each timescale
    #mini hack to speed up the search of the indices: only look into names_vector part where there actually proxies
    #reduce names vector to proxy-names
    proxy_names=names_vector[proxy_start:proxy_end]
    if 1 not in c.timescales:
        idx_list=[[proxy_start+np.argwhere(proxy_names==al)[0][0] for al in b] for b in sites_avail[1:]]
        idx_list.insert(0,[])
        
        #second list for the covariance localization matrix 
        idx_list_2=[[np.argwhere(proxy_names==al)[0][0] for al in b] for b in sites_avail[1:]]
        idx_list_2.insert(0,[])
        
    else:
        idx_list=[[proxy_start+np.argwhere(proxy_names==al)[0][0] for al in b] for b in sites_avail]
        idx_list_2=[[np.argwhere(proxy_names==al)[0][0] for al in b] for b in sites_avail]
    
    if c.anomaly is not None and c.anomaly!= False:
        print('Compute anomaly for proxies')
        pp_y_all_a=utils.anomaly_proxies(c,lisst)
        #reasign original array
        lisst=pp_y_all_a
        
        #also calculate anomaly for 'truth'. HXfull_all_fin only passed as a dummy here
        if c.ppe['use']==True and c.ppe['source']=='external':
            _,truth=utils.anomaly_noproxies(c,HXfull_all_fin,truth)

    #precompute decorrelation matrices for localized kalman filter
    
    if c.cov_loc is not None:
        if c.full_fields:
            print('Precalculating covariance distance matrices for localization')
            lats=HXfull_all_fin['lat']
            lons=HXfull_all_fin['lon']
            #select first available data variable. only its latitudes + longitudes are relevant
            model_dat=prior[list(prior.data_vars)[0]]
            PH_loc, HPH_loc= kalman_filters.covariance_loc(model_dat,proxy_lat=lats,proxy_lon=lons, cov_len=c.cov_loc)
            
            #total_len=values_vector.shape[1]
            #grid_boxes=length_vector[0]

            #repeat PH_loc given number of variables
            PH_loc=np.repeat(PH_loc,len(c.reconstruct),axis=0)
            #append the localization for the proxy estimates
            PH_loc=np.concatenate([PH_loc,HPH_loc],axis=0)

            total_len=values_vector.shape[1]

            #how many ones are left to be appended to PH_loc (for example for GMT, which should not be affected by covariance localization)
            num_ones=total_len-PH_loc.shape[0]
#            print(PH_loc.shape)
            
#            print('append',num_ones, ' ones')
            ones_mat=np.ones((num_ones,PH_loc.shape[1]))
            
            if num_ones>0:
                PH_loc=np.concatenate([PH_loc,ones_mat],axis=0)
        else:
            print('Can not use covariance localization when not assimilating full climate fields. Abort')
            raise ValueError


    ##PROXY FRAC:
    #the idea is too only use a fraction of proxy measurements (e.g. 75%) and repeat the reconstruction cfg['reps']-timexs.
    #We thus create a list of proxy indices to use in each reconstruction
    #the concept is extended to multi-timescale DA applying to computing this list for every timescale available
    #the current implementation doesn't exclude that a proxy is not used in a smaller timescale, but on a higher one (due to the reuse option)
    #this would be a bit more complicated to implement

    if c.proxy_frac is not None:
        empty=[]
        for i_lis, lis in enumerate(lisst):
            #proxy frac can either be a fraction, or an absolute number (the latter is especially relevant for PPEs when comparing different timescales)
            l=len(lis.site)
            if c.proxy_frac<1:
                prox_mems=int(c.proxy_frac*l) #e.g 0.75*163=122
            else:
                prox_mems=int(c.proxy_frac) # absolute number
                if prox_mems>l:
                    print('Not enough proxies for timescale ', i_lis)
                    print('Setting number of available proxies to maximum: ', l)
                    prox_mems=l
            prox_idx=dataloader.random_indices(prox_mems,l,c.reps,seed=c.seed)
            empty.append(prox_idx)
        #proxy frac is the list
        proxy_frac=empty 
        #reversed proxy_frac list without last, not needed
        #prox_frac_r=proxy_frac[:-1].reverse()

    #MULTI-TIMESCALE DATAASSIMILATION LOOP
    #Brief description:
        # The concept follows the Steiger,Hakim 2016 Multi-timescale data assimilation concept in a slightly adapted and computationally optimized form
        # We repeat the reconstruction as many times as given by the cfg['reps'] parameter. The idea is to do some Monte Carlo,
        # where the selected proxies and the prior is (slightly) different in each step.
        # In each repetion we create a prior_block matrix for all variables that are going to be assimilated. Block refers to the fact, that
        # we store consecutive years from the prior in order to create meaningful multi-year averages and covariances.
        # the thus reconstruct the whole timeseries block and subblockwise, where for the (sub)blocks x-year averages are assimilated and the annomalies
        # are then added back. 
        # The proxy data has been resampled and brought into lissts for each timescale in the Resampling step before (Check that part for details)
        # This preprocessing of the proxies has the advantage, that we do not need do assimilate each proxy one after another as originally
        # proposed in Steiger,Hakim 2016. Still, we have obvioulsy more Kalman-Filter calculations than in the single timescale DA.
        # The other, even more expensive computation step is the annomaly and mean calculation, which I optimized with numba 
        # (with a very modern machine, there might no much speed gain, but on my current computer it is up to an order of magnitude faster than
        # the straight forward numpy X-X.mean() solution.)
        # We start assimilating from the large timescales to the small ones (this way we can do the annomaly mean calculation for the block size only once for
        # each monte carlo repetion. Changing this order would require some changes in the code. 
        # To-Do:
            # Count if number of total Kalman Filter Calculations is correct -> Yes.
            # test at least once that the order of assimilation (large to small timescales) does not change so much


    #Comments
        # This loop (should) also works when using only one, for instance annual, timescale
        # Tried to clean up the code and introduce short comments on what is going on.
        # Be careful with naming conventions, and using deep copies of arrays when necessary
        # I added the option that no proxies are available in one time-slice (then, the prior_block is taken)
        # The info which proxies are available at which resolution is stored in idx_list, and I additionaly look which proxies are really available via xr.dropna().
        # A bit convoluted this way, but it works

    #indices of proxy records available for largest timescale
    idx_bs=idx_list[-1]
    #indices of proxy records available for largest timescale (for covariance loc)
    idx_bs_2=idx_list_2[-1]
    
    #backwards proxy and proxy error list
    lisst_bw=lisst[:-1][::-1]
    lisst_r_bw=lisst_r[:-1][::-1]
    # number of assimilated values, years and blocks to reconstruct, repetitions and ensemble members
    num_vals=values_vector.shape[1]
    num_times=len(times_list[0]) #basic time scale needs to be annual
    num_blocks=int(np.ceil(num_times/bs)) #  this was incorrect: num_times//bs
    reps=c.reps
    
    if c.multi_model_prior is None:
        nens=c.nens
    else:
        nens=c.nens*ran

    #RANDOM INDICES FOR PRIOR
    #MC_idx=dataloader.random_indices(c.nens,prior.time.shape[0]-bs-1,c.reps,seed=None)

    #Initialize mean and std array for saving reconstruction
    mean_array=np.empty((reps,num_times,num_vals))
    std_array=np.empty((reps,num_times,num_vals))

    print('Start Multitimescale DA loop.')
    
    #Monte carlo repetitions
    for r in tqdm.tqdm(range(reps)):
        #create prior_block form values vector, in the loops we create a similar variable called prior_block (be careful to take deepcopy if necessary)

        
        prior_b=utils.prior_block_mme(values_vector_list,bs,MC_idx_list[:,r])

        #pdb.set_trace()
        #proxy selection for that repetion

        if c.proxy_frac is not None:
            proxy_frac_idx = [p[r] for p in proxy_frac]
            #reversed proxy_frac_idx without last list
            proxy_frac_idx_r=list(reversed(proxy_frac_idx[:-1]))

        #compute mean and anomaly for prior_block, to that end save the shape, flatten, compute, unflatten
        #flattening just keeping the blocksize dimension (along which mean is calculated)
        shape=prior_b.shape
        prior_flat=prior_b.reshape(bs,-1)
        m_bs,a_bs=utils.anomean_with_numba(prior_flat,bs)
        m_bs=m_bs.reshape((1,shape[1],shape[2]))
        a_bs=a_bs.reshape(shape)

        #Available proxy estimates for largest time_scale
        HXf_bs_m=m_bs[:,:,idx_bs]

        #Optional dictionaries used for saving rank histograms if they are to be calculated
        rank_dic={}
        if 'rank_histogram' in c.metrics:
            for tres in time_res:
                rank_dic[str(tres)]=[]
        rank_dic_post={}
        if 'rank_histogram_posterior' in c.metrics:
            for tres in time_res:
                rank_dic_post[str(tres)]=[]
        
        #import pdb
        #pdb.set_trace()
        
        #loop over blocks
        for i in range(num_blocks):
            #assimilate block size means directly (saves one anomean calculation)
            current_time=times_list[-1][i]
            Y=lisst[-1].isel(time=i)
            R=lisst_r[-1].isel(time=i)

            #eventually only select some proxies
            if c.proxy_frac is not None:
                Y=Y[proxy_frac_idx[-1]]
                R=R[proxy_frac_idx[-1]]

            #indices where Y is not nan
            mask=~np.isnan(Y)

            Y=Y.dropna('site').values
            R=R.dropna('site').values

            #Get prior forecast (Ne x Nx)
            Xf=m_bs[0].copy() 

            if len(Y)>0:
                #Additionaly mask the the prior estimates as given by availability
                HXf=copy.deepcopy(HXf_bs_m[0])
                if c.proxy_frac is not None:
                    HXf=HXf[:,proxy_frac_idx[-1]]
                #option for covariance localization   
                if c.cov_loc is not None:
                    #necessary copy operations can take up to a ms!
                    PH_loc_bs=copy.deepcopy(PH_loc)
                    HPH_loc_bs=copy.deepcopy(HPH_loc)
                    
                    #proxies which are available in general for that timescale
                    PH_loc_bs=PH_loc_bs[:,idx_bs_2]
                    HPH_loc_bs=HPH_loc_bs[:,idx_bs_2][idx_bs_2]
                    
                    if c.proxy_frac is not None:
                        PH_loc_bs=PH_loc_bs[:,proxy_frac_idx[-1]]
                        HPH_loc_bs=HPH_loc_bs[proxy_frac_idx[-1]][:,proxy_frac_idx[-1]]
                    
                    PH_loc_bs=PH_loc_bs[:,mask]
                    HPH_loc_bs=HPH_loc[:,mask][mask]
                    
                HXf=HXf[:,mask] # Ne x Ny 
                #posterior (Has shape Nx )
                
                if c.cov_loc is not None:
                    Xf_post=np.real(kalman_filters.ENSRF_direct_loc(Xf.T,HXf.T,Y,R*c.error_scale,PH_loc_bs, HPH_loc_bs))
                else:
                    Xf_post=kalman_filters.ETKF(Xf.T,HXf.T,Y,R*c.error_scale) #input (Nx,Ne),(Ny,Ne),(Ny,1),(Ny,1)
                
                #add assimilated mean back to previous anomalies
                #prior_block=utils.mean_plus_anoms(Xf_post.T,a_bs)
                prior_block=Xf_post.T + a_bs
            else:
                prior_block=copy.deepcopy(prior_b)

            #eventually compute rank histograms, requires xr-DataArrays (time dimension is a dummy here)

            if 'rank_histogram' in c.metrics:
            
                HXf_xr=xr.DataArray(HXf,dims=('time','site'))
                Y_xr=xr.DataArray(Y,dims='site')
                rank=xss.rank_histogram(Y_xr,HXf_xr,dim='site',member_dim='time')
                rank_dic[str(bs)].append(rank)

            if 'rank_histogram_posterior' in c.metrics:
                Y_xr=xr.DataArray(Y,dims='site')
                idx_res=idx_list[-1]
                HXf_post=Xf_post[idx_res,:][mask].T
                HXf_post_xr=xr.DataArray(HXf_post,dims=('time','site'))
                rank=xss.rank_histogram(Y_xr,HXf_post_xr,dim='site',member_dim='time')
                rank_dic_post[str(bs)].append(rank)


            #loop over all other resolutions (backwards)
            for ii,res in enumerate(time_res_r):
                #proxy indices for that time_res
                idx_res=idx_list[:-1][::-1][ii]
                
                #proxy indices in the covariance localization matrices
                idx_res_2=idx_list_2[:-1][::-1][ii]
                
                if res!=1:
                    #compute anomalies and mean
                    prior_flat=prior_block.reshape(bs,-1)
                    mean_res,anom_res=utils.anomean_with_numba(prior_flat,res) #prior_flat not changed
                    
                    mean_res=mean_res.reshape((bs//res,nens,num_vals))
                    anom_res=anom_res.reshape(shape)
                else:
                    anom_res=np.zeros((shape))
                    mean_res=prior_block

                #loop over sub_index in block, computed via true divison (e.g. 50/25 = 2)
                bs_mod_res=bs//res
                for sub_index in range(bs_mod_res):
                    
                    #issue of endings not matching (e.g. when running from 851-1849 in 20 year blocks, last block is not complete)
                    #this is not a super nice solution. better would be to adapt the list where the subindices come from
                    #probably better to build some extra code for last iteration and checking here everytime
                    
                    
                    if sub_index * res + i*bs < num_times:
                 
                        #get the current proxies at the right time
                        Y=lisst_bw[ii][i*bs_mod_res+sub_index]

                        
                        #eventually slice
                        if c.proxy_frac is not None:
                            Y=Y[proxy_frac_idx_r[ii]]


                        #which proxies are available?
                        mask=~np.isnan(Y)
                        Y=Y.dropna('site').values
                            
                        
                        if len(Y)>0:
                            R=lisst_r_bw[ii][i*bs_mod_res+sub_index]
                            #eventually slice
                            if c.proxy_frac is not None:
                                R=R[proxy_frac_idx_r[ii]]

                            R=R.dropna('site').values
                            Xf=mean_res[sub_index,:]
                            #get averaged proxy estimates + available proxies
                            HXf=mean_res[sub_index,:][:,idx_res]

                            #apply proxy fraction
                            if c.proxy_frac is not None:
                                HXf=HXf[:,proxy_frac_idx_r[ii]]
                            
                            if c.cov_loc is not None:
                                #necessary copy operations can take up to a ms!
                                PH_loc_bs=copy.deepcopy(PH_loc)
                                HPH_loc_bs=copy.deepcopy(HPH_loc)
                                
                                #proxies which are available in general for that timescale
                                PH_loc_bs=PH_loc_bs[:,idx_res_2]
                                HPH_loc_bs=HPH_loc_bs[:,idx_res_2][idx_res_2]
                                
                                if c.proxy_frac is not None:
                                    PH_loc_bs=PH_loc_bs[:,proxy_frac_idx[-1]]
                                    HPH_loc_bs=HPH_loc_bs[proxy_frac_idx[-1]][:,proxy_frac_idx[-1]]
                                    
                                PH_loc_bs=PH_loc_bs[:,mask]
                                HPH_loc_bs=HPH_loc_bs[:,mask][mask]
                        
                            
                            
                            #slice according to mask
                            HXf=HXf[:,mask]  # Ne x Ny 
                            
                            if c.cov_loc is not None:
                                Xf_post=kalman_filters.ENSRF_direct_loc(Xf.T,HXf.T,Y,R*c.error_scale,PH_loc_bs, HPH_loc_bs)
                            else:
                                Xf_post=kalman_filters.ETKF(Xf.T,HXf.T,Y,R*c.error_scale) #input needs to be (Nx,Ne),(Ny,Ne),(Ny,1),(Ny,1)            
                            
                        #case that no proxies are available
                        else:
                            Xf_post=mean_res[sub_index,:].T

                        #replace prior_block values
                        start=sub_index*res
                        end=(sub_index+1)*res
                        
                        try:
                            #numba doesn't work wor some weird reason when using covariance localization. Also seems to be slower on ravenclaw for some reason.
                            #prior_block[start:end]=utils.mean_plus_anoms(Xf_post.T,anom_res[start:end])
                            prior_block[start:end]=Xf_post.T + anom_res[start:end]
                        except:
                            import pdb
                            pdb.set_trace()
                        
                        #eventually compute rank histograms, requires xr-DataArrays (time dimension is a dummy here)
                        if 'rank_histogram' in c.metrics:
                            HXf_xr=xr.DataArray(HXf,dims=('time','site'))
                            Y_xr=xr.DataArray(Y,dims='site')
                            rank=xss.rank_histogram(Y_xr,HXf_xr,dim='site',member_dim='time')
                            rank_dic[str(res)].append(rank)

                        if 'rank_histogram_posterior' in c.metrics:
                            Y_xr=xr.DataArray(Y,dims='site')
                            HXf_post=Xf_post[:,idx_res][:,mask]
                            HXf_post_xr=xr.DataArray(HXf_post,dims=('time','site'))
                            rank=xss.rank_histogram(Y_xr,HXf_post_xr,dim='site',member_dim='time')
                            rank_dic_post[str(res)].append(rank)                   
                    else:
                        pass
                                
            
            #compute mean values in block (along ensemble)
            mean_block=np.mean(prior_block,axis=1)
            std_block=np.std(prior_block,axis=1)

            #fill mean_array at that part for that repetition
            block_start=bs*i
            block_end=bs*(i+1)
            
            if block_end>num_times:
                
                mean_array[r,block_start:block_end,:]=mean_block[:bs-(block_end-num_times),:]
                std_array[r,block_start:block_end,:]=std_block[:bs-(block_end-num_times),:]
            else:
                mean_array[r,block_start:block_end,:]=mean_block
                std_array[r,block_start:block_end,:]=std_block
            
    #take mean along Monte Carlo
    mean_array_final=mean_array.mean(axis=0)
    std_array_final=std_array.mean(axis=0)      
    print('Finished multitimescale DA')
    
    
    #SAVING: SPLITTING UP THE VECTOR
    #Now we have to resplit everything, and eventually also calculate PPE evaluation metrics
    splitted_mean=np.split(mean_array_final,split_vector,axis=-1)
    splitted_std=np.split(std_array_final,split_vector,axis=-1)

    if c.full_fields:
        num_vars=len(c.reconstruct)
    else:
        num_vars=0

    #Create output folder if it doesn't already exist
    #cwd=os.getcwd()
    cwd=c.basepath
    
    pat=os.path.dirname(cwd)+'/results/experiments/'
    
    base_path=os.path.join(pat,c.output_folder)
    if not os.path.exists(base_path):
            os.mkdir(base_path)
      
    
    ds=utils.evaluation_saving(c, num_vars, names_short_vector, splitted_mean, splitted_std, times_list, coordinates, truth, prior, lisst, HXfull_all_fin, rank_dic, rank_dic_post, MC_idx_list, sites, prior_block, attributes, pp_y_all,pp_r_all,split_vector,time_res,cfg,base_path)
    
    
    #add missing '.nc'
    path1=c.output_file+'.nc'
    #increment path name by number if it already exits
    path1=dataloader.checkfile(os.path.join(base_path,path1))    
    
    ds.to_netcdf(path=path1)

    return ds
