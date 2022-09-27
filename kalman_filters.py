### Ensemble Kalman Filters implemented by me: https://github.com/mchoblet/ensemblefilters

import numpy as np
import scipy.linalg

def ESTKF(Xf, HXf, Y, R):
    """
    Error-subspace transform Kalman Filter
    
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 12, see section 5.9
    Errors: 
        5th line: A instead of L (A needs to be created)
        Last line: W_A instead of W'
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (assumed uncorrelated) (N_y x 1) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations (N_y x N_e)
    - Y: Observation vector (N_y x 1)
    Output:
    - Analysis ensemble (N_x, N_e)
    """
    
    # number of ensemble members
    Ne=np.shape(Xf)[1]
    
    #Obs error matrix
    Rmat=np.diag(R)
    Rmat_inv=np.diag(1/R)
    #Mean of prior ensemble for each state vector variable 
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    
    #Mean of model values in observation space
    mY = np.mean(HXf, axis=1)
    d=Y-mY

    """
    Create projection matrix:
    - create matrix of shape Ne x Ne-1 filled with off diagonal values
    - fill diagonal with diagonal values
    - replace values of last row
    """

    sqr_ne=-1/np.sqrt(Ne)
    off_diag=-1/(Ne*(-sqr_ne+1))
    diag=1+off_diag

    A=np.ones((Ne,Ne-1))*off_diag
    np.fill_diagonal(A,diag)
    A[-1,:]=sqr_ne

    #error in pseudocode, replace L by A
    HL=HXf @ A
    B1=Rmat_inv @ HL
    C1=(Ne-1)*np.identity(Ne-1)
    C2=C1+HL.T @ B1
    
    #EVD of C2, assumed symmetric
    eigs,U=np.linalg.eigh(C2)
    
    d1=B1.T @ d
    d2=U.T @ d1
    d3=d2/eigs
    T=U @ np.diag(1/np.sqrt(eigs)) @ U.T
    
    #mean weight
    wm=U @ d3
    #perturbation weight
    Wp=T @ A.T*np.sqrt((Ne-1))
    #total weight matrix + projection matrix transform
    W=wm[:,None]+Wp
    Wa = A @ W

    #Analysis ensemble
    Xa = mX[:,None] + Xfp @ Wa

    return Xa

def ETKF(Xf, HXf, Y, R):
    """
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 7, see section 5.4.
    Errors: Calculation of W1 prime, divide by square root of eigenvalues. The mathematical formula in the paper has an error already.
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) ($N_y$ x 1$) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector ($N_y$ x 1)

    Output:
    - Analysis ensemble (N_x, N_e)
    """
    # number of ensemble members
    Ne=np.shape(Xf)[1]

    #Obs error matrix
    #Rmat=np.diag(R)
    Rmat_inv=np.diag(1/R)
    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]

    C=Rmat_inv @ HXp
    A1=(Ne-1)*np.identity(Ne)
    A2=A1 + (HXp.T @ C)

    #eigenvalue decomposition of A2, A2 is symmetric
    eigs, ev = np.linalg.eigh(A2) 

    #compute perturbations
    Wp1 = np.diag(np.sqrt(1/eigs)) @ ev .T
    Wp = ev @ Wp1 * np.sqrt(Ne-1)

    #differing from pseudocode
    d=Y-mY
    D1 = Rmat_inv @ d
    D2 = HXp.T @ D1
    wm=ev @ np.diag(1/eigs) @ ev.T @ D2  #/ np.sqrt(Ne-1) 

    #adding pert and mean
    W=Wp + wm[:,None]

    #final adding up (most costly operation)
    Xa=mX[:,None] + Xfp @ W

    return Xa
   

def ETKF_livings(Xf, HXf, Y, R):
    """
    Adaption of the ETKF proposed by David Livings (2005)
    
    Implementation adapted from
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) ($N_y$ x 1$) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector ($N_y$ x 1)

    Output:
    - Analysis ensemble (N_x, N_e)
    """
    # number of ensemble members
    Ne=np.shape(Xf)[1]
    Ny=np.shape(Y)[0]

    #Obs error matrix
    Rmat=np.diag(R)
    Rmat_inv=np.diag(1/R)
    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]
    
    #Scaling of perturbations proposed by Livings (2005), numerical stability
    S_hat=np.diag(1/np.sqrt(R)) @ HXp/np.sqrt(Ne-1)
    
    #svd of S_hat transposed
    U,s,Vh=np.linalg.svd(S_hat.T)
    
    C=Rmat_inv @ HXp
    #recreate singular value matrix
    Sig=np.zeros((Ne,Ny))
    np.fill_diagonal(Sig,s)
    
    #perturbation weight
    mat=np.diag(1/np.sqrt(1+np.square(s)))
    Wp1=mat @ U.T
    Wp=U @ Wp1
    
    #innovation
    d=Y-mY
    #mean weight
    D = np.diag(1/np.sqrt(R)) @ d
    D2= Vh @ D
    D3 = np.diag(1/(1+np.square(s))) @ Sig @ D2
    wm= U @ D3 / np.sqrt(Ne-1)

    #adding pert and mean
    W=Wp + wm[:,None]

    #final adding up (most costly operation)
    Xa=mX[:,None] + Xfp @ W
    
    return Xa

def EnSRF(Xf, HXf, Y, R):
    """
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 9, see section 5.6. Pseudocode has some errors, eg. in step 7 it should be sqrt(Lambda).
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) (N_y x 1) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations (N_y x N_e)
    - Y: Observation vector (N_y x 1)

    Output:
    - Analysis ensemble (N_x, N_e)
    """
    #Obs error matrix
    Rmat=np.diag(R)
    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]

    #Gram matrix of perturbations
    I1=HXp @ HXp.T
    Ny=np.shape(Y)[0]
    Ne=np.shape(Xf)[1]

    I2=I1+(Ne-1)*Rmat
    #compute eigenvalues and eigenvectors (use that matrix is symmetric and real)
    eigs, ev = np.linalg.eigh(I2) 

    #Error in Pseudocode: Square Root + multiplication order (important!)
    G1=ev @ np.diag(np.sqrt(1/eigs)) 
    G2=HXp.T @ G1

    U,s,Vh=np.linalg.svd(G2)
    #Compute  sqrt of matrix, Problem of imaginary values?? (singular values are small)
    rad=(np.ones(Ne)-np.square(s)).astype(complex)
    rad=np.sqrt(rad)
    A=np.diag(rad)

    W1p=U @ A
    W2p=W1p@U.T

    d=Y-mY

    w1=ev.T @ d
    w2=np.diag(1/eigs).T @ w1
    w3=ev @ w2
    w4=HXp.T @ w3
    W=W2p+w4[:,None]
    Xa=mX[:,None]+Xfp @ W

    return Xa

def ENSRF_direct(Xf, HXf, Y, R):
    """
    direct calculation of Ensemble Square Root Filter from Whitaker and Hamill
    As for instance done in Steiger 2018: "A reconstruction of global hydroclimate and dynamical variables over the Common Era".
    
    In comparison to the code for that paper [1], the matrix multiplications are performed  consequently from left to right and 
    the kalman gain is not explicitely computed, because this would be inefficient when we are just interested in the posterior ensemble.
    One could also avoid computing the matrix inverses and solve linear systems instead (one could even use Cholesky decomposition
    because the covariance matrices are positive definite), but as the number of observations is small the speed up is insignificant.
    When using many observations (>1000) one should consider doing it. Here, the main computation effort comes from the matrix square root
    (potentially numerically unstable) and unavoidable matrix - matrix multiplications.
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) ($N_y$ x 1$) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector ($N_y$ x 1)
    Output:
    - Analysis ensemble (N_x, N_e)
    
    [1] https://github.com/njsteiger/PHYDA-v1/blob/master/M_update.m
    """
    Ne=np.shape(Xf)[1]

    #Obs error matrix, assumption that it's diagonal
    Rmat=np.diag(R)
    Rsqr=np.diag(np.sqrt(R)) 

    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]
    #innovation
    d=Y-mY

    #compute matrix products directly
    #BHT=(Xfp @ HXp.T)/(Ne-1) #avoid this, it's inefficient to compute it here
    HPHT=(HXp @ HXp.T)/(Ne-1)

    #second Kalman gain factor
    HPHTR=HPHT+Rmat
    #inverse of term
    HPHTR_inv=np.linalg.inv(HPHTR)
    #matrix square root of denominator
    HPHTR_sqr=scipy.linalg.sqrtm(HPHTR)

    #Kalman gain for mean
    xa_m=mX + (Xfp @ (HXp.T /(Ne-1) @ (HPHTR_inv @ d)))

    #Perturbation Kalman gain
    #inverse of square root calculated via previous inverse: sqrt(A)^(-1)=sqrt(A) @ A^(-1)
    HPHTR_sqr_inv=HPHTR_sqr @ HPHTR_inv
    fac2=HPHTR_sqr + Rsqr
    factor=np.linalg.inv(fac2)

    #right to left multiplication!
    pert = (Xfp @ (HXp.T/(Ne-1) @ (HPHTR_sqr_inv.T @ (factor @ HXp))))
    Xap=Xfp-pert
    
    return Xap+xa_m[:,None]


def EnSRF_serial(Xf, HXf, Y, R):
    """
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    algorithm 10, see section 5.7.
    Errors: Line 1 must be inside of loop, in HPH^T the divisor Ne-1 is missing.
    This version uses the appended state vector approach, which also updates the precalculated observations from the model.
    
    
    Dimensions: N_e: ensemble size, N_y: Number of observations: N_x: State vector size (Gridboxes x assimilated variables)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) ($N_y$ x 1$) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector ($N_y$ x 1)
    Output:
    - Analysis ensemble (N_x, N_e)
    """

    # augmented state vector with Ye appended
    Xfn = np.append(Xf, HXf, axis=0)
    
    # number of state variables
    Nx= np.shape(Xf)[0]
    # number of ensemble members
    Ne=np.shape(Xf)[1]
    #Number of measurements
    Ny=np.shape(Y)[0]
    for i in range(Ny):
        #ensemble mean and perturbations
        mX = np.mean(Xfn, axis=1)
        Xfp=np.subtract(Xfn,mX[:,None])
        
        #get obs from model
        HX=Xfn[Nx+i,:]
        #ensemble mean for obs
        mY=np.mean(HX)
        #remove mean
        HXp=(HX-mY)[None]

        HP=HXp @ Xfp.T /(Ne-1)
        
        #Variance at location (here divisor is missing in reference!)
        HPHT=HXp @ HXp.T/(Ne-1)

        ##Localize HP ?
        
        #compute scalar
        sig=R[i]
        F=HPHT + sig
        K=(HP/F)

        #compute factors for final calc
        d=Y[i]-mY
        a1=1+np.sqrt(sig/F)
        a2=1/a1
        
        #final calcs
        mXa=mX+np.squeeze((K*d))
        Xfp=Xfp-a2*K.T @ HXp
        Xfn=Xfp+mXa[:,None]
        
    return Xfn[:Nx,:]

def covariance_loc(model_data,proxy_lat,proxy_lon, cov_len):    
    """
    Function that returns the matrices needed for the Covariance Localization in the direct EnSRF solver by Hadamard (element-wise) product.
    These are the terms called W_loc and Y_loc here: https://www.nature.com/articles/s41586-020-2617-x#Sec7 (Data Assimilation section).
    The idea is to compute these matrices once in the beginning for all available proxy locations, and later in the DA loop one only selects
    the relevant columns of W_loc / rows and columns of Y_loc for the localized simultaneous Kalman Filter Solver.
    
    Input:
       - model_data from which the grid point locations are extracted. Here I use the stack function, which I also use when constructing the
       prior vector. In brings all gridpoints in a vector form (xarray-DataArray such that stack can be applied, N_x grid points)
       - proxy_lat, proxy_lon are the latitudes and longitudes of the proxy locations (np.arrays, length = N_y). Make sure they have the same ordering as
       the entries of your Observations-from-Model (HXf) in the Kalman Filter.
       - cov_len: Radius for Gaspari Cohn function [float, in km ]
       
    Ouput:
        - PH_loc: Matrix for localization of PH^T (N_x * N_y)
        - HPH_loc: Matrix for localization of HPH^T (N_y * N_y)
    """
    from haversine import haversine_vector, Unit
    
    #bring coordinates of model (field) and proxy (individual locations) into the right form
    #the method we use are different due to this different structures (field and individual locations)
    loc=np.array([[lat,lon] for lat,lon in zip(proxy_lat,proxy_lon)])
    stacked=model_data.stack(z=('lat','lon')).transpose('z','time')
    coords=[list(z) for z in stacked.z.values]
    
    #model-proxy distances
    dists_mp=haversine_vector(loc,coords, Unit.KILOMETERS,comb=True)
    dists_mp_shape=dists_mp.shape
    
    #proxy-proxy distances
    dists_pp=haversine_vector(loc,loc, Unit.KILOMETERS,comb=True)
    dists_pp_shape=dists_pp.shape
    
    def gaspari_cohn(dists,cov_len):
        """
        Gaspari Cohn decorrelation function https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.49712555417 page 26
        dists: need to be a 1-D array with all the distances (Reshape to your needs afterwards)
        cov_len: radius given in km
        """
        dists = np.abs(dists)
        array = np.zeros_like(dists)
        r = dists/cov_len
        #first the short distances
        i=np.where(r<=1.)[0]
        array[i]=-0.25*(r[i])**5+0.5*r[i]**4+0.625*r[i]**3-5./3.*r[i]**2+1.
        #then the long ones
        i=np.where((r>1) & (r<=2))[0]
        array[i]=1./12.*r[i]**5-0.5*r[i]**4+0.625*r[i]**3+5./3.*r[i]**2.-5.*r[i]+4.-2./(3.*r[i])

        array[array < 0.0] = 0.0
        return array
    
    #flatten distances, apply to Gaspari Cohn and reshape
    PH_loc=gaspari_cohn(dists_mp.reshape(-1),cov_len).reshape(dists_mp_shape)
    HPH_loc=gaspari_cohn(dists_pp.reshape(-1),cov_len).reshape(dists_pp_shape)
    
    return PH_loc, HPH_loc



def ENSRF_direct_loc(Xf, HXf, Y, R,PH_loc, HPH_loc):
    """
    direct calculation of Ensemble Square Root Filter from Whitaker and Hamill
    applying localization matrices to PH^T and HPH^T as in Tierney 2020: 
    https://www.nature.com/articles/s41586-020-2617-x#Sec7 (Data Assimilation section).
    This is less efficient than without localization, becaue PH needs to be explicitely calculated for the entry-wise hadamard product
    (https://en.wikipedia.org/wiki/Hadamard_product_(matrices)). 
    However, this is still better than using the serial EnSRF formulation (At least an order of magnitude faster).
    It is important to not compute the Kalman gains explicitely.
    As commented in the docstring of ENSRF_direct, avoiding inverting matrices could be done, but the speed up is insignificant in comparison to the rest.
    
    I propose to compute PH_loc and HPH_loc once for all possible proxy locations, and here only select the 
    relevant columns (for PH_loc) and the relvant rows and columns for HPH_loc using fancy indexing:
    PH_loc -> PH_loc[:,[column_indices]]
    HPH_loc -> HPH_loc[[row_indices]][:,[column_indices]],
    given which proxies are available at one timestep.
    
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) (N_y) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations ($N_y$ x $N_e$)
    - Y: Observation vector (N_y)
    - PH_loc: Matrix for localization of PH^T (N_x * N_y)
    - HPH_loc: Matrix for localization of HPH^T (N_y * N_y)
    
    Output:
    - Analysis ensemble (N_x, N_e)
    """
    
    Ne=np.shape(Xf)[1]

    #Obs error matrix, assumption that it's diagonal
    Rmat=np.diag(R)
    Rsqr=np.diag(np.sqrt(R)) 

    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]
    #innovation
    d=Y-mY

    #compute matrix products directly
    #entry wise product of covariance localization matrices
    PHT= PH_loc * (Xfp @ HXp.T/(Ne-1))
    HPHT= HPH_loc * (HXp @ HXp.T/(Ne-1))
    
    #second Kalman gain factor
    HPHTR=HPHT+Rmat
    #inverse of factor
    HPHTR_inv=np.linalg.inv(HPHTR)
    #matrix square root of denominator
    HPHTR_sqr=scipy.linalg.sqrtm(HPHTR)

    #Kalman gain for mean
    xa_m=mX + PHT @ (HPHTR_inv @ d)

    #Perturbation Kalman gain
    #inverse of square root calculated via previous inverse: sqrt(A)^(-1)=sqrt(A) @ A^(-1)
    HPHTR_sqr_inv=HPHTR_sqr @ HPHTR_inv
    fac2=HPHTR_sqr + Rsqr
    factor=np.linalg.inv(fac2)

    # right to left multiplication!
    pert = PHT @ (HPHTR_sqr_inv.T @ (factor @ HXp))
    Xap=Xfp-pert
    
    return Xap+xa_m[:,None]

def SEnKF(Xf, HXf, Y, R):
    """
    Stochastic Ensemble Kalman Filter
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    
    Changes: The pseudocode is not consistent with the description in 5.1, where the obs-from-model are perturbed, but in the pseudocode it's the other way round.
    Hence the 8th line D= ... is confusing if we would generate Y as described in the text.
    Last line needs to have 1/(Ne-1)
    
    Input:
    - Xf:  the prior ensemble (N_x x N_e) 
    - R: Measurement Error (Variance of pseudoproxy timerseries) (N_y x 1) -> converted to Ny x Ny matrix
    - HX^f: Model value projected into observation space/at proxy locations (N_y x N_e)
    - Y: Observation vector (N_y x 1)
    Output:
    - Analysis ensemble (N_x, N_e)
   
    """
    # number of ensemble members
    Ne=np.shape(Xf)[1]
    Ny=np.shape(R)[0]
    #Obs error matrix
    Rmat=np.diag(R)
    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]

    HPH=HXp@HXp.T /(Ne-1)

    A=HPH + Rmat

    rng = np.random.default_rng(seed=42)
    Y_p=rng.standard_normal((Ny, Ne))*np.sqrt(R)[:,None]

    D= Y[:,None]+Y_p - HXf
    
    #solve linear system for getting inverse
    C=np.linalg.solve(A,D)
    
    E=HXp.T @ C
    
    Xa=Xf+Xfp@(E/(Ne-1))
    
    return Xa


    
def SEnKF_loc(Xf, HXf, Y, R,PH_loc, HPH_loc):
    """
    Stochastic Ensemble Kalman Filter that can do localisation. Changes the order of calculations with
    respect to optimized SEnKF.
    Implementation adapted from pseudocode description in
    "State-of-the-art stochastic data assimialation methods" by Vetra-Carvalho et al. (2018),
    
    for the calculation of PH_loc/HPH_loc look at the function in ensrf_direct_loc.py
    
    Changes: see SEnKF function
    """
    # number of ensemble members
    Ne=np.shape(Xf)[1]
    Ny=np.shape(R)[0]
    #Obs error matrix
    Rmat=np.diag(R)
    #Mean of prior ensemble for each gridbox   
    mX = np.mean(Xf, axis=1)
    #Perturbations from ensemble mean
    Xfp=Xf-mX[:,None]
    #Mean and perturbations for model values in observation space
    mY = np.mean(HXf, axis=1)
    HXp = HXf-mY[:,None]

    #Hadamard product for localisation
    HPH=HPH_loc * (HXp@HXp.T /(Ne-1))

    A=HPH + Rmat

    rng = np.random.default_rng(seed=42)
    Y_p=rng.standard_normal((Ny, Ne))*np.sqrt(R)[:,None]

    D= Y[:,None]+Y_p - HXf
    
    #solve linear system for getting inverse
    C=np.linalg.solve(A,D)
    
    Pb=PH_loc*(Xfp @ HXp.T/(Ne-1)) 
    
    Xa=Xf + Pb @ C
    
    return Xa

