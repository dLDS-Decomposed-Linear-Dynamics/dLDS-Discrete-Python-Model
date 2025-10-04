"""
Decomposed Linear Dynamical Systems (dLDS) for learning the latent components of neural dynamics
@code author: noga mudrik
"""

"""
Imports
"""

# simple imports
import matplotlib
import numpy as np
from scipy import linalg
import pandas as pd
#import random

# Plotting imports
from webcolors import name_to_rgb
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from colormap import rgb2hex

# Linear algebra imports
from numpy.linalg import matrix_power
from scipy.linalg import expm
from sklearn import linear_model
try:
    import pylops
except:
    print('itertools was not uploaded')    
import itertools

# os and files loading imports
import os
import dill   
import mat73
import warnings
import pickle
sep = os.sep


#%% FHN model
 
   
def create_FHN(dt = 0.01, max_t = 100, I_ext = 0.5, b = 0.7, a = 0.8 , tau = 20, v0 = -0.5, w0 = 0, 
               params = {'exp_power' : 0.9, 'change_speed': False}):
    time_points = np.arange(0, max_t, dt)
    if params['change_speed']:
        time_points = time_points**params['exp_power']    
        
    w_full = []
    v_full = []
    v = v0
    w = w0
    for t in time_points:
        v, w =  cal_next_FHN(v,w, dt , max_t , I_ext , b, a , tau)
        v_full.append(v)
        w_full.append(w)
    return v_full, w_full


        
def cal_next_FHN(v,w, dt = 0.01, max_t = 300, I_ext = 0.5, b = 0.7, a = 0.8 , tau = 20) :
    v_next = v + dt*(v - (v**3)/3 - w + I_ext)
    w_next = w + dt/tau*(v + a - b*w)
    return v_next, w_next

#%% Lorenz attractor dynamics definition  
    
def lorenz(x, y, z, s=10, r=25, b=2.667):
    """
    Inputs:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Outputs:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def create_lorenz_mat(t = [], initial_conds = (0., 1., 1.05) , txy = []):
  """
  Create the lorenz dynamics
  """
  if len(t) == 0: t = np.arange(0,1000,0.01)
  if len(txy) == 0: txy = t

  xs = np.zeros(len(t)-1)
  ys = np.zeros(len(t)-1)
  zs = np.zeros(len(t)-1)

  # Set initial values
  xs[0], ys[0], zs[0] = initial_conds


  for i in range(len(t[:-2])):
      dt_z = t[i+1] - t[i]
      dt_xy =  txy[i+1] - txy[i]
      x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
      xs[i + 1] = xs[i] + (x_dot * dt_xy)
      ys[i + 1] = ys[i] + (y_dot * dt_xy)
      zs[i + 1] = zs[i] + (z_dot * dt_z)
  return xs, ys, zs

def load_mat_file(mat_name , mat_path = '',sep = sep):
    """
    Function to load mat files. Useful for uploading the c. elegans data. 
    Example:
        load_mat_file('WT_Stim.mat')
    """
    data_dict = mat73.loadmat(mat_path+sep+mat_name)
    return data_dict    

def create_dynamics(type_dyn = 'cyl', max_time = 1000, dt = 0.01, 
                    params_ex = {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}):
  """
  Create ground truth dynamics. 
  Inputs:
      type_dyn          = Can be 'cyl', 'lorenz','FHN', 'multi_cyl', 'torus', 'circ2d', 'spiral'
      max_time          = integer. Number of time points for the dynamics. Relevant only if data is empty;
      dt                = time interval for the dynamics.
      params_ex         = dictionary of parameters for the dynamics.  {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}):
    
      
  Outputs:
      dynamics: k X T matrix of the dynamics 
      
  """
  t = np.arange(0, max_time, dt)
  if type_dyn == 'cyl':
    x = params_ex['radius']*np.sin(t)
    y = params_ex['radius']*np.cos(t)
    z = t     + params_ex['bias']


    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'spiral':
    x = t*np.sin(t)
    y = t*np.cos(t)
    z = t 

    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'lorenz':    
    txy = t

    x,y,z  = create_lorenz_mat(t, txy = txy)
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'torus':
    R=5;    r=1;
    u=np.arange(0,max_time,dt);
    v=np.arange(0,max_time,dt);
    [u,v]=np.meshgrid(u,v);
    x=(R+r*np.cos(v)) @ np.cos(u);
    y=(R+r*np.cos(v)) @ np.sin(u);
    z=r*np.sin(v);
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'circ2d':
    x = params_ex['radius']*np.sin(t)
    y = params_ex['radius']*np.cos(t)
    dynamics = np.vstack([x.flatten(),y.flatten()]) 
  elif type_dyn == 'multi_cyl':
    dynamics0 = create_dynamics('cyl',max_time = 50,params_ex = params_ex)
    list_dyns = []
    for dyn_num in range(params_ex['num_cyls']):
        np.random.seed(dyn_num)
        random_trans = np.random.rand(dynamics0.shape[0],dynamics0.shape[0])-0.5
        transformed_dyn = random_trans @ dynamics0
        list_dyns.append(transformed_dyn)
    dynamics = np.hstack(list_dyns)
  elif type_dyn == 'c_elegans':
      mat_c_elegans = load_mat_file('WT_NoStim.mat','E:\CoDyS-Python-rep-\other_models') # 
      dynamics = mat_c_elegans['WT_NoStim']['traces'].T
  elif type_dyn == 'lorenz_2d':
    txy = t

    x,y,z  = create_lorenz_mat(t, txy = txy)
    dynamics = np.vstack([x.flatten(),z.flatten()]) 
  elif type_dyn.lower() == 'fhn':
    v_full, w_full = create_FHN(dt = dt, max_t = max_time, I_ext = 0.5, 
                                b = 0.7, a = 0.8 , tau = 20, v0 = -0.5, w0 = 0,
                                params = {'exp_power' : params_ex['exp_power'], 'change_speed': False})      
    
    dynamics = np.vstack([v_full, w_full])
  return    dynamics



#%% Basic Model Functions
#%% Main Model Training
def train_model_include_D(max_time = 500, dt = 0.1, dynamics_type = 'cyl',num_subdyns = 3, 
                          error_reco = np.inf,  data = [], step_f = 30, GD_decay = 0.85, max_error = 1e-3, 
                          max_iter = 3000, F = [], coefficients = [], params= {'update_c_type':'inv','reg_term':0,'smooth_term':0}, 
                          epsilon_error_change = 10**(-5), D = [], x_former =[], latent_dim = None, include_D  = False,step_D = 30, reg1=0,reg_f =0 , 
                          max_data_reco = 1e-3,  sigma_mix_f = 0.1,  action_along_time = 'median', to_print = True, seed = 0, seed_f = 0, 
                          normalize_eig  = True,  params_ex = {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}, 
                          init_distant_F = False,max_corr = 0.1, decaying_reg = 1, other_params_c={}, include_last_up = False):
    
  """
  This is the main function to train the model! 
  Inputs:
      max_time      = Number of time points for the dynamics. Relevant only if data is empty;
      dt            =  time interval for the dynamics
      dynamics_type = type of the dynamics. Can be 'cyl', 'lorenz', 'multi_cyl', 'torus', 'circ2d', 'spiral'
      num_subdyns   = number of sub-dynamics
      error_reco    = intial error for the reconstruction (do not touch)
      data          = if one wants to use a pre define groud-truth dynamics. If not empty - it overwrites max_time, dt, and dynamics_type
      step_f        = initial step size for GD on the sub-dynamics
      GD_decay      = Gradient descent decay rate
      max_error     = Threshold for the model error. If the model arrives at a lower reconstruction error - the training ends.
      max_iter      = # of max. iterations for training the model
      F             = pre-defined sub-dynamics. Keep empty if random.
      coefficients  = pre-defined coefficients. Keep empty if random.
      params        = dictionary that includes info about the regularization and coefficients solver. e.g. {'update_c_type':'inv','reg_term':0,'smooth_term':0}
      epsilon_error_change = check if the sub-dynamics do not change by at least epsilon_error_change, for at least 5 last iterations. Otherwise - add noise to f
      D             = pre-defined D matrix (keep empty if D = I)
      latent_dim    =  If D != I, it is the pre-defined latent dynamics.
      include_D     = If True -> D !=I; If False -> D = I
      step_D        = GD step for updating D, only if include_D is true
      reg1          = if include_D is true -> L1 regularization on D
      reg_f         = if include_D is true ->  Frobenius norm regularization on D
      max_data_reco = if include_D is true -> threshold for the error on the reconstruction of the data (continue training if the error (y - Dx)^2 > max_data_reco)
      sigma_mix_f            = std of noise added to mix f
      action_along_time      = the function to take on the error over time. Can be 'median' or 'mean'
      to_print               = to print error value while training? (boolean)
      seed                   = random seed
      seed_f                 = random seed for initializing f
      normalize_eig          = whether to normalize each sub-dynamic by dividing by the highest abs eval
      params_ex              = parameters related to the creation of the ground truth dynamics. e.g. {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}
      init_distant_F         = when initializing F -> make sure that the correlation between each pair of {f}_i does not exeed a threshold
      max_corr               = max correlation between each pair of initial sub-dyns (relevant only if init_distant_F is True)
      decaying_reg           = decaying factor for the l1 regularization on the coefficients. If 1 - there is no decay. (should be a scalar in (0,1])
      other_params_c         = additional parameters for the update step of c
      include_last_up        = add another update step of the coefficients at the end
  """  
  # if D != I
  if not include_D and len(data) > 1: latent_dyn = data   

  step_f_original = step_f
  
  # Define data and number of dyns
  if len(data) == 0 :
    data            = create_dynamics(type_dyn = dynamics_type, max_time = max_time, dt = dt, params_ex = params_ex)
    if not include_D: latent_dyn = data
  else:
    if isinstance(data, np.ndarray) and len(data) > 1: 
      if not include_D: latent_dyn = data
    else:       
      if len(data) == 1:
        data = data[0]
        if not include_D: latent_dyn = data 
      else:
          raise ValueError('The parameter "data" is invalid')
  
  # Define # of time points 
  n_times = data.shape[1]
  
  # Default value for the latent dimension if D != I     
  if include_D and np.isnan(latent_dim):
      latent_dim = int(np.max([data.shape[0] / 5,3]))
  else:      latent_dim = data.shape[0]; 

  if include_D: # If model needs to study D
      if len(D) == 0: D = init_mat(size_mat = (data.shape[0], latent_dim) , dist_type ='sparse', init_params={'k':4})
      elif D.shape[0] != data.shape[0]: raise ValueError('# of rows in D should be = # rows in the data ')

  else:
    latent_dyn = data

  if len(F) == 0:            
      F = [init_mat((latent_dim, latent_dim),normalize=True,r_seed = seed_f+i) for i in range(num_subdyns)]
      # Check that initial F's are far enough from each other
      if init_distant_F:
           F = check_F_dist_init(F, max_corr = max_corr)
 
  """
  Initialize Coeffs
  """
  if len(coefficients) == 0: 
                             coefficients   = init_mat((num_subdyns,n_times-1))
  if len(params) == 0:       params         = {'update_c_type':'inv','reg_term':0,'smooth_term':0}
  
  if not include_D:
      cur_reco              = create_reco(latent_dyn=latent_dyn, coefficients= coefficients, F=F)

  
  data_reco_error  = np.inf
  
    
  counter = 1
 
  error_reco_array = []

  while  data_reco_error > max_data_reco and (counter < max_iter):
      
    ### Store Iteration Results

     
    """
    Update x
    """
    
    if include_D:
      latent_dyn = update_X(D, data,random_state=seed)

      
    """
    Decay reg 
    """
    if params['update_c_type'] == 'lasso':
        params['reg_term'] = params['reg_term']*decaying_reg 
    
    """
    Update coefficients
    """
    if counter != 1:        coefficients = update_c(F,latent_dyn, params,random_state=seed,other_params=other_params_c)

        
    
    """
    Update D
    """
    
    if include_D:
      one_dyn: D = update_D(D, step_D, latent_dyn, data, reg1,reg_f) 

    

    """
    Update F
    """   

    F = update_f_all(latent_dyn,F,coefficients,step_f,normalize=False, action_along_time= action_along_time,    normalize_eig = normalize_eig )  

    step_f *= GD_decay
    
    if include_D:
        data_reco_error = np.mean((data - D @ latent_dyn)**2)
    mid_reco = create_reco(latent_dyn, coefficients, F)
    error_reco = np.mean((latent_dyn -mid_reco)**2)

    error_reco_array.append(error_reco)
   
    if np.mean(np.abs(np.diff(error_reco_array[-5:]))) < epsilon_error_change:
      F = [f_i + sigma_mix_f*np.random.randn(f_i.shape[0],f_i.shape[1]) for f_i in F]
      print('mixed F')

    if to_print:
        print('Error = %s'%str(error_reco) )
        if include_D:            print('Error reco y = %s'%str(data_reco_error))

    counter += 1
    if counter == max_iter: print('Arrived to max iter')
 

  # Post training adjustments
  if include_last_up:
      coefficients = update_c(F, latent_dyn,params,  {'reg_term': 0, 'update_c_type':'inv','smooth_term' :0, 'num_iters': 10, 'threshkind':'soft'})
  else:
      coefficients = update_c(F, latent_dyn, params,other_params=other_params_c)  
        
  print(error_reco_array)
  if not include_D: 
      D = [];
  return coefficients, F, latent_dyn, error_reco_array, D
      

def update_D(former_D, step_D , x, y, reg1 = 0, reg_f= 0) :
  """
  Update the matrix D by applying GD. Relevant just in case where D != I
  """
  
  if reg1 == 0 and reg_f ==0:
    D = y @ linalg.pinv(x)
  else:
    basic_error = -2*(y - former_D @ x ) @ x.T
    if reg1 != 0:      reg1_error = np.sum(np.sign(former_D))
    else: reg1_error = 0      
    if reg_f != 0:      reg_f_error = 2*former_D
    reg_f_error = 0
    D = former_D - step_D *(basic_error + reg1*reg1_error + reg_f* reg_f_error)
  return D

def update_X(D, data, reg1 = 0, former_x = [], random_state = 0, other_params ={}):  
  """
  Update the latent dynamics. Relevant just in case where D != I
  """
  if reg1 == 0 :
    x = linalg.pinv(D) @ data
  else:
    clf = linear_model.Lasso(alpha=reg1,random_state=random_state, **other_params)
    clf.fit(D,data)
    x = np.array(clf.coef_)
  return x

def check_F_dist_init(F, max_corr = 0.1):
    """
    This function aims to validate that the matrices in F are far enough from each other
    """
    combs = list(itertools.combinations(np.arange(len(F)),2))
    corr_bool = [spec_corr(F[comb_s[0]],F[comb_s[1]]) > max_corr for comb_s in combs]
    counter= 100
    while (corr_bool == False).any():
        counter +=1
        for comb_num,comb in enumerate(combs):
            if spec_corr(F[comb[0]],F[comb[1]])  > max_corr:
                fi_new = init_mat(np.shape(F[0]),dist_type = 'norm',r_seed = counter)
                F[comb[0]] = fi_new
    return F
        
def spec_corr(v1,v2):
  """
  absolute value of correlation
  """
  corr = np.corrcoef(v1[:],v2[:])
  return np.abs(corr[0,1])    
    

def init_mat(size_mat, r_seed = 0, dist_type = 'norm', init_params = {'loc':0,'scale':1}, normalize = False):
  """
  This is an initialization function to initialize matrices like G_i and c. 
  Inputs:
    size_mat    = 2-element tuple or list, describing the shape of the mat
    r_seed      = random seed (should be integer)
    dist_type   = distribution type for initialization; can be 'norm' (normal dist), 'uni' (uniform dist),'inti', 'sprase', 'regional'
    init_params = a dictionary with params for initialization. The keys depends on 'dist_type'.
                  keys for norm -> ['loc','scale']
                  keys for inti and uni -> ['low','high']
                  keys for sparse -> ['k'] -> number of non-zeros in each row
                  keys for regional -> ['k'] -> repeats of the sub-dynamics allocations
    normalize   = whether to normalize the matrix
  Output:
      the random matrix with size 'size_mat'
  """
  np.random.seed(r_seed)
  random.seed(r_seed)
  if dist_type == 'norm':
    rand_mat = np.random.normal(loc=init_params['loc'],scale = init_params['scale'], size= size_mat)
  elif dist_type == 'uni':
    if 'high' not in init_params.keys() or  'low' not in init_params.keys():
        raise KeyError('Initialization did not work since low or high boundries were not set')
    rand_mat = np.random.uniform(init_params['low'],init_params['high'], size= size_mat)
  elif dist_type == 'inti':
    if 'high' not in init_params.keys() or  'low' not in init_params.keys():
      raise KeyError('Initialization did not work since low or high boundries were not set')
    rand_mat = np.random.randint(init_params['low'],init_params['high'], size= size_mat)
  elif dist_type == 'sparse':
    if 'k' not in init_params.keys():
      raise KeyError('Initialization did not work since k was not set')

    k=init_params['k']
    b1 = [random.sample(list(np.arange(size_mat[0])),np.random.randint(1,np.min([size_mat[0],k]))) for i in range(size_mat[1])]
    b2 = [[i]*len(el) for i,el in enumerate(b1)]
    rand_mat = np.zeros((size_mat[0], size_mat[1]))
    rand_mat[np.hstack(b1), np.hstack(b2)] = 1
  elif dist_type == 'regional':
    if 'k' not in init_params.keys():
      raise KeyError('Initialization did not work since k was not set for regional initialization')

    k=init_params['k']
    splits = [len(split) for split in np.split(np.arange(size_mat[1]),k)]
    cur_repeats = [np.repeat(np.eye(size_mat[0]), int(np.ceil(split_len/size_mat[0])),axis = 1) for split_len in  splits]
    cur_repeats = np.hstack(cur_repeats)[:size_mat[1]]
    
    rand_mat = cur_repeats
  else:
    raise NameError('Unknown dist type!')
  if normalize:
    rand_mat = norm_mat(rand_mat)
  return rand_mat

  
def norm_mat(mat, type_norm = 'evals', to_norm = True):
  """
  This function comes to norm matrices.
  Inputs:
      mat       = the matrix to norm
      type_norm = what type of normalization to apply. Can be:
          - 'evals' - normalize by dividing by the max eigen-value
          - 'max'   - divide by the maximum abs value in the matrix
          - 'exp'   -  normalization using matrix exponential (matrix exponential) 
      to_norm   = whether to norm or not to.
  Output:  
      the normalized matrix
  """    
  if to_norm:
    if type_norm == 'evals':
      eigenvalues, _ =  linalg.eig(mat)
      mat = mat / np.max(np.abs(eigenvalues))
    elif type_norm == 'max':
      mat = mat / np.max(np.abs(mat))
    elif type_norm  == 'exp':
      mat = np.exp(-np.trace(mat))*expm(mat)
  return mat


def update_c(F, latent_dyn, 
             params_update_c = {'update_c_type':'inv','reg_term':0,'smooth_term':0, 'to_norm_fx' : False},clear_dyn = [], 
             direction = 'c2n',other_params = {'warm_start':False},random_state=0 , skip_error = False, cofficients = []):
  """  
  The function comes to update the coefficients of the sub-dynamics, {c_i}, by solving the inverse or solving lasso.
  Inputs:
      F               = list of sub-dynamics. Should be a list of k X k arrays. 
      latent_dyn      = latent_dynamics (dynamics dimensions X time)
      params_update_c = dictionary with keys:
          update_c_type  = options:
               - 'inv' (least squares)
               - 'lasso' (sklearn lasso)
               - 'fista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.FISTA.html)
               - 'omp' (https://pylops.readthedocs.io/en/latest/gallery/plot_ista.html#sphx-glr-gallery-plot-ista-py)
               - 'ista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.ISTA.html)       
               - 'IRLS' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.IRLS.html)
               - 'spgl1' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.SPGL1.html)
               
               
               - . Refers to the way the coefficients should be claculated (inv -> no l1 regularization)
          reg_term       = scalar between 0 to 1, describe the reg. term on the cofficients
          smooth_term    = scalar between 0 to 1, describe the smooth term on the cofficients (c_t - c_(t-1))
      direction      = can be c2n (clean to noise) OR  n2c (noise to clean)
      other_params   = additional parameters for the lasso solver (optional)
      random_state   = random state for reproducability (optional)
      skip_error     = whether to skip an error when solving the inverse for c (optional)
      cofficients    = needed only if smooth_term > 0. This is the reference coefficients matrix to apply the constraint (c_hat_t - c_(t-1)) on. 
      
  Outputs: 
      coefficients matrix (k X T), type = np.array
      
  example:
  coeffs = update_c(np.random.rand(2,2), np.random.rand(2,15),{})
  """  

  if isinstance(latent_dyn,list):
    if len(latent_dyn) == 1: several_dyns = False
    else: several_dyns = True
  else:
    several_dyns = False
  if several_dyns: 
    n_times = latent_dyn[0].shape[1]-1
  else:
    n_times = latent_dyn.shape[1]-1
    
    
  params_update_c = {**{'update_c_type':'inv', 'smooth_term': 0, 'reg_term':0},**params_update_c}
  if len(clear_dyn) == 0:
    clear_dyn = latent_dyn
  if direction == 'n2c':
    latent_dyn, clear_dyn  =clear_dyn,  latent_dyn
  if isinstance(F,np.ndarray): F = [F]
  coeffs_list = []
  
  
  for time_point in np.arange(n_times):
    if not several_dyns:
      cur_dyn = clear_dyn[:,time_point]
      next_dyn = latent_dyn[:,time_point+1]
      total_next_dyn = next_dyn
      f_x_mat = []
      for f_i in F:
        f_x_mat.append(f_i @ cur_dyn)
      stacked_fx = np.vstack(f_x_mat).T 
      stacked_fx[stacked_fx> 10**8] = 10**8
    else: 
      total_next_dyn = []
      for dyn_num in range(len(latent_dyn)):
        cur_dyn = clear_dyn[dyn_num][:,time_point]
        next_dyn = latent_dyn[dyn_num][:,time_point+1]
        total_next_dyn.extend(next_dyn.flatten().tolist())
        f_x_mat = []
        for f_num,f_i in enumerate(F):
          f_x_mat.append(f_i @ cur_dyn)
        if dyn_num == 0:
          stacked_fx = np.vstack(f_x_mat).T 
        else:
          stacked_fx = np.vstack([stacked_fx, np.vstack(f_x_mat).T ])
        stacked_fx[stacked_fx> 10**8] = 10**8
    
      total_next_dyn = np.reshape(np.array(total_next_dyn), (-1,1))
    if len(F) == 1: stacked_fx = np.reshape(stacked_fx,[-1,1])
    if params_update_c['smooth_term'] > 0 and time_point > 0 :
        if len(cofficients) == 0:
            warnings.warn("Warning: you called the smoothing option without defining coefficients")
    if params_update_c['smooth_term'] > 0 and time_point > 0 and len(cofficients) > 0 :
        c_former = cofficients[:,time_point-1].reshape((-1,1))
        total_next_dyn_full = np.hstack([total_next_dyn, np.sqrt(params_update_c['smooth_term'])*c_former])
        stacked_fx_full = np.hstack([stacked_fx, np.sqrt(params_update_c['smooth_term'])*np.eye(len(stacked_fx))])
    else:
        total_next_dyn_full = total_next_dyn
        stacked_fx_full = stacked_fx   

    if params_update_c['update_c_type'] == 'inv' or (params_update_c['reg_term'] == 0 and params_update_c['smooth_term'] == 0):
      try:
          coeffs =linalg.pinv(stacked_fx_full) @ total_next_dyn_full.reshape((-1,1))
      except:
          if not skip_error:
              raise NameError('A problem in taking the inverse of fx when looking for the model coefficients')
          else:
              return np.nan*np.ones((len(F), latent_dyn.shape[1]))
    elif params_update_c['update_c_type'] == 'lasso' :

      clf = linear_model.Lasso(alpha=params_update_c['reg_term'],random_state=random_state, **other_params)
      clf.fit(stacked_fx_full,total_next_dyn_full.T )     
      coeffs = np.array(clf.coef_)

    elif params_update_c['update_c_type'].lower() == 'fista' :
        Aop = pylops.MatrixMult(stacked_fx_full)
        #print('fista')
        if 'threshkind' not in params_update_c: params_update_c['threshkind'] ='soft'

        coeffs = pylops.optimization.sparsity.FISTA(Aop, total_next_dyn_full.flatten(), niter=params_update_c['num_iters'],eps = params_update_c['reg_term'] , threshkind =  params_update_c.get('threshkind') )[0]

    elif params_update_c['update_c_type'].lower() == 'ista' :
        #print('ista')

        if 'threshkind' not in params_update_c: params_update_c['threshkind'] ='soft'
        Aop = pylops.MatrixMult(stacked_fx_full)
        coeffs = pylops.optimization.sparsity.ISTA(Aop, total_next_dyn_full.flatten(), niter=params_update_c['num_iters'] , 
                                                   eps = params_update_c['reg_term'],threshkind =  params_update_c.get('threshkind'))[0]
   
        
        
    elif params_update_c['update_c_type'].lower() == 'omp' :
        #print('omp')
        Aop = pylops.MatrixMult(stacked_fx_full)
        coeffs  = pylops.optimization.sparsity.OMP(Aop, total_next_dyn_full.flatten(), niter_outer=params_update_c['num_iters'], sigma=params_update_c['reg_term'])[0]
        
        
    elif params_update_c['update_c_type'].lower() == 'spgl1' :
        #print('spgl1')
        Aop = pylops.MatrixMult(stacked_fx_full)
        coeffs = pylops.optimization.sparsity.SPGL1(Aop, total_next_dyn_full.flatten(),iter_lim = params_update_c['num_iters'],
                                                   tau = params_update_c['reg_term'])[0]
        
        
    elif params_update_c['update_c_type'].lower() == 'irls' :
        #print('irls')
        Aop = pylops.MatrixMult(stacked_fx_full)
        
        coeffs = pylops.optimization.sparsity.IRLS(Aop, total_next_dyn_full.flatten(),  nouter=50, espI = params_update_c['reg_term'])[0]

        
    else:
        
        
      raise NameError('Unknown update c type')
    coeffs_list.append(coeffs.flatten())
  coeffs_final = np.vstack(coeffs_list)

  return coeffs_final.T


def create_next(latent_dyn, coefficients, F,time_point):
  """
  This function evaluate the dynamics at t+1 given the value of the dynamics at time t, the sub-dynamics, and other model parameters
  Inputs:
      latent_dyn    = the latent dynamics (can be either ground truth or estimated). [k X T]
      coefficients  = the sub-dynamics coefficients (used by the model)
      F             = a list of np.arrays, each np.array is a sub-dynamic with size kXk
      time_point    = current time point
      order         = how many time points in the future we want to estimate
  Outputs:
      k X 1 np.array describing the dynamics at time_point+1

  """
  if isinstance(F[0],list):
    F = [np.array(f_i) for f_i in F]

  if latent_dyn.shape[1] > 1:
    cur_A  = np.dstack([coefficients[i,time_point]*f_i @ latent_dyn[:, time_point] for i,f_i in enumerate(F)]).sum(2).T   
  else:
    cur_A  = np.dstack([coefficients[i,time_point]*f_i @ latent_dyn for i,f_i in enumerate(F)]).sum(2).T 
  return cur_A

def create_ci_fi_xt(latent_dyn,F,coefficients, cumulative = False, mute_infs = 10**50, 
                    max_inf = 10**60):
    
  """
  An intermediate step for the reconstruction -
  Specifically - It calculated the error that should be taken in the GD step for updating f: 
  f - eta * output_of(create_ci_fi_xt)
  output: 
      3d array of the gradient step (unweighted): [k X k X time]
  """

  if max_inf <= mute_infs:
    raise ValueError('max_inf should be higher than mute-infs')
  curse_dynamics = latent_dyn

  all_grads = []
  for time_point in np.arange(latent_dyn.shape[1]-1):
    if cumulative:
      if time_point > 0:
        previous_A = cur_A
      else:
        previous_A = curse_dynamics[:,0]
      cur_A = create_next(np.reshape(previous_A,[-1,1]), coefficients, F,time_point)
    else:
      cur_A = create_next(curse_dynamics, coefficients, F,time_point)
    next_A = latent_dyn[:,time_point+1]
    
    """
    The actual step
    """

    if cumulative:
      gradient_val = (next_A - cur_A) @ previous_A.T
    else:
      gradient_val = (next_A - cur_A) @ curse_dynamics[:, time_point].T
    all_grads.append(gradient_val)
  return np.dstack(all_grads)


def update_f_all(latent_dyn,F,coefficients,step_f, normalize = False, acumulated_error = False,
                 action_along_time = 'mean', weights_power = 1.2, normalize_eig = True):
    
  """
  Update all the sub-dynamics {f_i} using GD
  """
      
  if action_along_time == 'mean':
    
    all_grads = create_ci_fi_xt(latent_dyn,F,coefficients)
    new_f_s = [norm_mat(f_i+2*step_f*norm_mat(np.mean(all_grads[:,:,:]*np.reshape(coefficients[i,:], [1,1,-1]), 2),to_norm = normalize),to_norm = normalize_eig ) for i,f_i in enumerate(F)] 
  elif action_along_time == 'median':
    all_grads = create_ci_fi_xt(latent_dyn,F,coefficients)
            
    new_f_s = [norm_mat(f_i+2*step_f*norm_mat(np.median(all_grads[:,:,:]*np.reshape(coefficients[i,:], [1,1,-1]), 2),to_norm = normalize),to_norm = normalize_eig ) for i,f_i in enumerate(F)] 
    
  else:
    raise NameError('Unknown action along time. Should be mean or median')
  for f_num in range(len(new_f_s)):
      rand_mat = np.random.rand(new_f_s[f_num].shape[0],new_f_s[f_num].shape[1])
      new_f_s[f_num][np.isnan(new_f_s[f_num])] = rand_mat[np.isnan(new_f_s[f_num])] .flatten()
      
  return new_f_s




#%% Plotting functions
def add_bar_dynamics(coefficients_n, ax_all_all = [],min_max_points = [10,100,200,300,400,500], 
                     colors = np.array(['r','g','b','yellow']), centralize = False):
    if isinstance(ax_all_all, list) and len(ax_all_all) == 0:
        fig, ax_all_all  = plt.subplots(1,len(min_max_points), figsize = (8*len(min_max_points), 7))

    max_bar = np.max(np.abs(coefficients_n[:,min_max_points]))
    for pair_num,val in enumerate(min_max_points):
        ax_all = ax_all_all[pair_num]

        
        ax_all.bar(np.arange(coefficients_n.shape[0]),coefficients_n[:,val], 
                   color = np.array(colors)[:coefficients_n.shape[0]],
                   alpha = 0.3)
        # ax_all.set_title('t = %s'%str(val), fontsize = 40, fontweight = 'bold')
        ax_all.get_xaxis().set_ticks([]) #for ax in ax_all]
        ax_all.get_yaxis().set_ticks([]) #for ax in ax_all]
        ax_all.spines['top'].set_visible(False)
        
        ax_all.spines['right'].set_visible(False)
        ax_all.spines['bottom'].set_visible(False)
        ax_all.spines['left'].set_visible(False)  
        ax_all.axhline(0, ls = '-',alpha = 0.5, color = 'black', lw = 6)
        ax_all.set_ylim([-max_bar,max_bar])
        
def plot_sub_effect(sub_dyn, rec_rad_all = 5, colors = ['r','g','b','m'], alpha = 0.8, ax = [], 
                    n_points = 100, figsize = (10,10), params_labels = {'title':'sub-dyn effect'}, lw = 4):
    params_labels = {**{'zlabel':None}, **params_labels}
    if isinstance(ax,list) and len(ax) == 0:
        fig, ax = plt.subplots(figsize = figsize)
    if len(colors) == 1: colors = [colors]*4
    if not isinstance(rec_rad_all,list): rec_rad_all = [rec_rad_all]
    ax.axhline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    ax.axvline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    for rec_rad in rec_rad_all:        
        ax.plot([-rec_rad, rec_rad],[rec_rad,rec_rad],alpha = alpha**2, color = colors[0], ls ='--',lw=lw)
        ax.plot([-rec_rad, rec_rad],[-rec_rad,-rec_rad],alpha = alpha**2, color = colors[1], ls = '--',lw=lw)
        ax.plot([rec_rad,  rec_rad],[-rec_rad,rec_rad],alpha = alpha**2, color = colors[2], ls = '--',lw=lw)
        ax.plot([-rec_rad,-rec_rad], [ -rec_rad,  rec_rad],alpha = alpha**2, color = colors[3], ls = '--',lw=lw)
    

        if not (sub_dyn == 0).all():
            sub_dyn = norm_mat(sub_dyn, type_norm = 'evals')
        effect_up = sub_dyn @ np.vstack([np.linspace(-rec_rad, rec_rad, n_points), [rec_rad]*n_points])
        effect_down = sub_dyn @ np.vstack([np.linspace(-rec_rad, rec_rad, n_points), [-rec_rad]*n_points])
        effect_right = sub_dyn @ np.vstack([[rec_rad]*n_points,np.linspace(-rec_rad, rec_rad, n_points)])
        effect_left = sub_dyn @ np.vstack([[-rec_rad]*n_points,np.linspace(-rec_rad, rec_rad, n_points)])
        ax.plot(effect_up[0,:],effect_up[1,:],alpha = alpha, color = colors[0],lw=lw)
        ax.plot(effect_down[0,:],effect_down[1,:],alpha = alpha, color = colors[1],lw=lw)
        ax.plot(effect_right[0,:],effect_right[1,:],alpha = alpha, color = colors[2],lw=lw)
        ax.plot(effect_left[0,:],effect_left[1,:],alpha = alpha, color = colors[3],lw=lw)
        # Up
        add_arrow(ax, [0,rec_rad], [np.mean(effect_up[0,:]),np.mean(effect_up[1,:])],arrowprops = {'facecolor' :colors[0]})
        add_arrow(ax, [0,-rec_rad], [np.mean(effect_down[0,:]),np.mean(effect_down[1,:])],arrowprops = {'facecolor' :colors[1]})
        add_arrow(ax, [rec_rad,0], [np.mean(effect_right[0,:]),np.mean(effect_right[1,:])],arrowprops = {'facecolor' :colors[2]})
        add_arrow(ax, [-rec_rad,0], [np.mean(effect_left[0,:]),np.mean(effect_left[1,:])],arrowprops = {'facecolor' :colors[3]})
    add_labels(ax, **params_labels)


def add_dummy_sub_legend(ax, colors,lenf, label_base = 'f'):
    dummy_lines = []
    for i,color in enumerate(colors[:lenf]):
        dummy_lines.append(ax.plot([],[],c = color, label = '%s %s'%(label_base, str(i)))[0])
    ax.set_title('Dynamics colored by mix of colors of the dominant dynamics')
    legend = ax.legend([dummy_lines[i] for i in range(len(dummy_lines))], ['f %s'%str(i) for i in range(len(colors))], loc = 'upper left')
    ax.legend()
        
def plot_subs_effects_2d(F, colors =[['r','maroon','darkred','coral'],['forestgreen','limegreen','darkgreen','springgreen']] , alpha = 0.7 , rec_rad_all = 5, 
                         n_points = 100,  params_labels = {'title':'sub-dyn effect'}, lw = 4, evec_colors = ['r','g'], include_dyn = False, loc_leg = 'upper left', 
                         axs = [], fig = []):
    if include_dyn:
        fig, axs = plt.subplots(len(F), 3, figsize = (35,8*len(F)),sharey='col', sharex = 'col')
    else:
        if isinstance(axs,list) and len(axs) == 0:
            fig, axs = plt.subplots(len(F), 2, figsize = (30,8*len(F)),sharey='col', sharex = 'col')
            
    if isinstance(colors[0], list):
        [plot_sub_effect(f_i, rec_rad_all , colors[i] , alpha, axs[i,1], n_points, params_labels = {'title':'f %s effect'%str(i+1)}, lw = lw) for i,f_i in enumerate(F)]

    else:
        [plot_sub_effect(f_i, rec_rad_all , colors , alpha, axs[i,1], n_points, params_labels = {'title':'f %s effect'%str(i+1)}, lw = lw) for i,f_i in enumerate(F)]
        
    [plot_evals_evecs(axs[i,0], f_i, evec_colors[i] , alpha) for i,f_i in enumerate(F)]
    dummy_lines = []
    dummy_lines.append(axs[0,0].plot([],[], c="black", ls = '--', lw = lw)[0])
    dummy_lines.append(axs[0,0].plot([],[], c="black", ls = '-', lw = lw)[0])

    legend = axs[0,1].legend([dummy_lines[i] for i in [0,1]], ['Original', 'after sub-dynamic transform'], loc = loc_leg )
    axs[0,1].add_artist(legend)
    if include_dyn:
        [quiver_plot(sub_dyn = f, ax = axs[i,2], chosen_color = evec_colors[i], type_plot='streamplot',cons_color =True, xlabel = 'dv',ylabel = 'dw') for i,f in enumerate(F)]
        [axs[i,2].set_title('f %s'%str(i+1), fontsize = 18) for i in range(len(F))]
        
    fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
    
    
def plot_subs(F, axs = [],params_F_plot = {'cmap':'PiYG'}, include_sup = True,annot = True):
  """
  This function plots heatmaps of the sub-dynamics
  """
  params_F_plot = {**{'cmap':'PiYG'},**params_F_plot}
  if isinstance(axs,list):
    if len(axs) == 0:
      fig, axs = plt.subplots(1,len(F), sharex = True,sharey = True)

  [sns.heatmap(f_i, ax = axs[i],annot=annot, **params_F_plot) for i,f_i in enumerate(F)]
  [ax.set_title('f#%g'%i) for i,ax in enumerate(axs)]
  if include_sup: plt.suptitle('Sub-Dynamics')
  plt.subplots_adjust(hspace = 0.5,wspace = 0.5)
    

def plot_evals_evecs(ax, sub_dyn, colors =['r','g','b','m'] , alpha = 0.7, title ='evals'):
    eigenvalues, eigenvectors =  linalg.eig(sub_dyn)
    for eval_num, eigenval in enumerate(eigenvalues):
        #add_arrow(ax, [0,0], [eigenvectors[0,eval_num], eigenvectors[1,eval_num]], arrowprops = {'facecolor' :colors[eval_num]})
        ax.scatter( np.real(eigenval),np.imag(eigenval), alpha = alpha, color = colors, s = 300) #[eval_num])
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.axhline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    ax.axvline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    ax.set_title('evals')
    

def plot_3d_color_scatter(latent_dyn,coefficients, ax = [], figsize = (15,10), delta = 0.4, colors = []):
    
    if latent_dyn.shape[0] != 3:
        print('Dynamics is not 3d')
        pass
    else:
        if len(colors) == 0:
            colors = ['r','g','b']
        if isinstance(ax,list) and len(ax) == 0:
            fig, ax = plt.subplots(figsize = figsize, subplot_kw={'projection':'3d'})  
        for row in range(coefficients.shape[0]):
            coefficients_row = coefficients[row]
            coefficients_row[coefficients_row == 0]  = 0.01
            
            ax.scatter(latent_dyn[0,:]+delta*row,latent_dyn[1,:]+delta*row,latent_dyn[2,:]+delta, s = coefficients_row**0.3, c = colors[row])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
        
        
def add_arrow(ax, start, end,arrowprops = {'facecolor' : 'black', 'width':1, 'alpha' :0.2} ):
    arrowprops = {**{'facecolor' : 'black', 'width':1.5, 'alpha' :0.2, 'edgecolor':'none'}, **arrowprops}
    ax.annotate('',ha = 'center', va = 'bottom',  xytext = start,xy =end,
                arrowprops = arrowprops)
    

def rgb_to_hex(rgb_vec):
  r = rgb_vec[0]; g = rgb_vec[1]; b = rgb_vec[2]
  return rgb2hex(int(255*r), int(255*g), int(255*b))

def remove_edges(ax):
    ax.spines['top'].set_visible(False)    
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
def quiver_plot(sub_dyn = [], xmin = -5, xmax = 5, ymin = -5, ymax = 5, ax = [], chosen_color = 'red',
                alpha = 0.4, w = 0.02, type_plot = 'quiver', zmin = -5, zmax = 5, cons_color = False,
                return_artist = False,xlabel = 'x',ylabel = 'y',quiver_3d = False,inter=2):
    """
    type_plot - can be either quiver or streamplot
    """
    
    if len(sub_dyn) == 0:
        sub_dyn =  np.array([[0,-1],[1,0]])

    
    if ymin >= ymax:
        raise ValueError('ymin should be < ymax')
    elif xmin >=xmax:            
        raise ValueError('xmin should be < xmax')
    else:

        if not quiver_3d:
            if isinstance(ax,list) and len(ax) == 0:
                fig, ax = plt.subplots()
            X, Y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin,ymax))

            new_mat = sub_dyn - np.eye(len(sub_dyn))

            U = new_mat[0,:] @ np.vstack([X.flatten(), Y.flatten()])
            V = new_mat[1,:] @ np.vstack([X.flatten(), Y.flatten()])

            if type_plot == 'quiver':
                h = ax.quiver(X,Y,U,V, color = chosen_color, alpha = alpha, width = w)
            elif type_plot == 'streamplot':

                
                x = np.linspace(xmin,xmax,100)
                y = np.linspace(ymin,ymax,100)
                X, Y = np.meshgrid(x, y)
                new_mat = sub_dyn - np.eye(len(sub_dyn))
                U = new_mat[0,:] @ np.vstack([X.flatten(), Y.flatten()])
                V = new_mat[1,:] @ np.vstack([X.flatten(), Y.flatten()])
                

                if cons_color:

                    if len(chosen_color[:]) == 3 and isinstance(chosen_color, (list,np.ndarray)): 
                        color_stream = rgb_to_hex(chosen_color)
                    elif isinstance(chosen_color, str) and chosen_color[0] != '#':
                        color_stream = list(name_to_rgb(chosen_color))
                    else:
                        color_stream = chosen_color

                else:
                    new_mat_color = np.abs(new_mat  @ np.vstack([x.flatten(), y.flatten()]))
                    color_stream = new_mat_color.T @ new_mat_color
                try:
                    h = ax.streamplot(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100),U.reshape(X.shape),V.reshape(Y.shape), color = color_stream) #chosen_color
                except:
                    h = ax.streamplot(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100),U.reshape(X.shape),V.reshape(Y.shape), color = chosen_color) #chosen_color
            else:
                raise NameError('Wrong plot name')
        else:
            if isinstance(ax,list) and len(ax) == 0:
                fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
            X, Y , Z = np.meshgrid(np.arange(xmin, xmax,inter), np.arange(ymin,ymax,inter), np.arange(zmin,zmax,inter))

            new_mat = sub_dyn - np.eye(len(sub_dyn))
            U = np.zeros(X.shape); V = np.zeros(X.shape); W = np.zeros(X.shape); 

            for xloc in np.arange(X.shape[0]):
                for yloc in np.arange(X.shape[1]):
                    for zloc in np.arange(X.shape[2]):
                        U[xloc,yloc,zloc] = new_mat[0,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))
                        V[xloc,yloc,zloc] = new_mat[1,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))
                        W[xloc,yloc,zloc] = new_mat[2,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))

            if type_plot == 'quiver':                    
                h = ax.quiver(X,Y,Z,U,V,W, color = chosen_color, alpha = alpha,lw = 1.5, length=0.8, normalize=True,arrow_length_ratio=0.5)#, width = w
                ax.grid(False)
            elif type_plot == 'streamplot':
                raise NameError('streamplot is not accepted for the 3d case')
         
            else:
                raise NameError('Wront plot name')
    if quiver_3d: zlabel ='z'
    else: zlabel = None
 
    add_labels(ax, zlabel = zlabel, xlabel = xlabel, ylabel = ylabel) 
    if return_artist: return h


def visualize_dyn(dyn,ax = [], params_plot = {},turn_off_back = False, marker_size = 10, include_line = False, 
                  color_sig = [],cmap = 'cool', return_fig = False, color_by_dominant = False, coefficients =[],
                  figsize = (5,5),colorbar = False, colors = [],vmin = None,vmax = None, color_mix = False, alpha = 0.4,
                  colors_dyns = np.array(['r','g','b','yellow']), add_text = 't ', text_points = [],fontsize_times = 18, 
                  marker = "o",delta_text = 0.5, color_for_0 =None, legend = [],fig = [],return_mappable = False):
   """
   Plot the multi-dimensional dynamics
   Inputs:
       dyn          = dynamics to plot. Should be a np.array with size k X T
       ax           = the subplot to plot in (optional)
       params_plot  = additional parameters for the plotting (optional). Can include plotting-related keys like xlabel, ylabel, title, etc.
       turn_off_back= disable backgroud of the plot? (optional). Boolean
       marker_size  = marker size of the plot (optional). Integer
       include_line = add a curve to the plot (in addition to the scatter plot). Boolean
       color_sig    = the color signal. if empty and color_by_dominant - color by the dominant dynamics. If empty and not color_by_dominant - color by time.
       cmap         = cmap
       colors       = if not empty -> pre-defined colors for the different sub-dynamics. Otherwise - colors are according to the cmap.
       color_mix    = relevant only if  color_by_dominant. In this case the colors need to be in the form of [r,g,b]
   Output:
       (only if return_fig) -> returns the figure      
      
   """
   if not isinstance(color_sig,list) and not isinstance(color_sig,np.ndarray): color_sig = [color_sig]


   if isinstance(ax,list) and len(ax) == 0:
       if dyn.shape[0] == 3:
           fig, ax = plt.subplots(figsize = figsize, subplot_kw={'projection':'3d'})  
       else:
           fig, ax = plt.subplots(figsize = figsize)  
           
       

   if include_line:
       if dyn.shape[0] == 3:
           ax.plot(dyn[0,:], dyn[1,:], dyn[2,:],alpha = 0.2)
       else:
           ax.plot(dyn[0,:], dyn[1,:], alpha = 0.2)
   if len(legend) > 0:
       [ax.scatter([],[], c = colors_dyns[i], label = legend[i], s = 10) for i in np.arange(len(legend))]
       ax.legend()
   # Create color sig        
   if len(color_sig) == 0: 
       color_sig = np.arange(dyn.shape[1])      
   if color_by_dominant and (coefficients.shape[1] == dyn.shape[1]-1 or coefficients.shape[1] == dyn.shape[1]): 
       if color_mix:
           if len(colors) == 0 or not np.shape(colors)[0] == 3: raise ValueError('colors mat should have 3 rows')
           else:

               color_sig = ((np.array(colors)[:,:coefficients.shape[0]] @ np.abs(coefficients))  / np.max(np.abs(coefficients).sum(0).reshape((1,-1)))).T
               color_sig[np.isnan(color_sig) ] = 0.1
               dyn = dyn[:,:-1]
       else:
           
           color_sig_tmp = find_dominant_dyn(coefficients)
           if len(colors_dyns) > 0: 
               color_sig = colors_dyns[color_sig_tmp]
           elif len(color_sig) == 0:  
               color_sig=color_sig_tmp 
           else:        
               color_sig=np.array(color_sig)[color_sig_tmp] 
           if len(color_sig.flatten()) < dyn.shape[1]: dyn = dyn[:,:len(color_sig.flatten())]
           if color_for_0:

               color_sig[np.sum(coefficients,0) == 0] = color_for_0


   if dyn.shape[0] > 2:
       if len(colors) == 0:
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker = marker, s = marker_size,c= color_sig,cmap = cmap, alpha = alpha,
                          vmin = vmin, vmax = vmax)
       else:
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker =marker, s = marker_size,c= color_sig, alpha = alpha)
   else:
       dyn = np.array(dyn)
       
       if len(colors) == 0:
           h = ax.scatter(dyn[0,:], dyn[1,:],  marker = marker, s = marker_size,c= color_sig,cmap = cmap, alpha = alpha,
                          vmin = vmin, vmax = vmax)
       else:
           h = ax.scatter(dyn[0,:], dyn[1,:],  marker = marker, s = marker_size,c= color_sig, alpha = alpha)
  
           params_plot['zlabel'] = None
   if len(params_plot) > 0:
     if dyn.shape[0] == 3:
         if 'xlabel' in params_plot.keys():
           add_labels(ax, xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
         elif 'zlabel' in params_plot.keys():
               add_labels(ax,  zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
         else:
           add_labels(ax,   title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
     else:
         if 'xlabel' in params_plot.keys():
           add_labels(ax, xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=None, title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None)
         elif 'zlabel' in params_plot.keys():
               add_labels(ax,  zlabel=None, title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None)
         else:
           add_labels(ax,   title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None,zlabel = None);
   if len(text_points) > 0:
       
       if dyn.shape[0] == 3:
           [ax.text(dyn[0,t]+delta_text,dyn[1,t]+delta_text,dyn[2,t]+delta_text, '%s = %s'%(add_text, str(t)),  fontsize =fontsize_times, fontweight = 'bold') for t in text_points]
       else:
           [ax.text(dyn[0,t]+delta_text,dyn[1,t]+delta_text, '%s = %s'%(add_text, str(t)),  fontsize =fontsize_times, fontweight = 'bold') for t in text_points]

   remove_edges(ax)
   ax.set_axis_off()
   if colorbar:
       fig.colorbar(h, cax=ax, position = 'top')
   if return_mappable:
       return h
 
            
            
#%% Helper Functions and Post-Analysis Functions
def str2bool(str_to_change):
    """
    Transform 'true' or 'yes' to True boolean variable 
    Example:
        str2bool('true') - > True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes')
    return str_to_change

def norm_over_time(coefficients, type_norm = 'normal'):
    if type_norm == 'normal':
        coefficients_norm = (coefficients - np.mean(coefficients,1).reshape((-1,1)))/np.std(coefficients, 1).reshape((-1,1))
    return coefficients_norm

def norm_coeffs(coefficients, type_norm, same_width = True,width_des = 0.7,factor_power = 0.9, min_width = 0.01):
    """
    type_norm can be:      'sum_abs', 'norm','abs'
    """
    if type_norm == 'norm':
        coefficients_n =      norm_over_time(np.abs(coefficients), type_norm = 'normal')   
        coefficients_n =      coefficients_n - np.min(coefficients_n,1).reshape((-1,1))

    elif type_norm == 'sum_abs':
        coefficients[np.abs(coefficients) < min_width] = min_width
        coefficients_n = np.abs(coefficients) / np.sum(np.abs(coefficients),1).reshape((-1,1))
    elif type_norm == 'abs':
        coefficients[np.abs(coefficients) < min_width] = min_width
        coefficients_n = np.abs(coefficients) 
    elif type_norm == 'no_norm':
        coefficients_n = coefficients
    else:
        raise NameError('Invalid type_norm value')


    coefficients_n[coefficients_n < min_width]  = min_width
    if same_width:        coefficients_n = width_des*(np.abs(coefficients_n)**factor_power) / np.sum(np.abs(coefficients_n)**factor_power,axis = 0)   
    else:                 coefficients_n = np.abs(coefficients_n) / np.sum(np.abs(coefficients_n),axis = 0)  
    coefficients_n[coefficients_n < min_width]  = min_width
    return coefficients_n

    
def movmfunc(func, mat, window = 3, direction = 0):
  """
  moving window with applying the function func on the matrix 'mat' towrads the direction 'direction'
  """
  if len(mat.shape) == 1: 
      mat = mat.reshape((-1,1))
      direction = 0
  addition = int(np.ceil((window-1)/2))
  if direction == 0:
    mat_wrap = np.vstack([np.nan*np.ones((addition,np.shape(mat)[1])), mat, np.nan*np.ones((addition,np.shape(mat)[1]))])
    movefunc_res = np.vstack([func(mat_wrap[i-addition:i+addition,:],axis = direction) for i in range(addition, np.shape(mat_wrap)[0]-addition)])
  elif direction == 1:
    mat_wrap = np.hstack([np.nan*np.ones((np.shape(mat)[0],addition)), mat, np.nan*np.ones((np.shape(mat)[0],addition))])
    movefunc_res = np.vstack([func(mat_wrap[:,i-addition:i+addition],axis = direction) for i in range(addition, np.shape(mat_wrap)[1]-addition)]).T
  return movefunc_res

def create_reco(latent_dyn,coefficients, F, type_find = 'median',min_far =10, smooth_coeffs = False,
                smoothing_params = {'wind':5}):
  """
  This function creates the reconstruction 
  Inputs:
      latent_dyn   = the ground truth latent dynamics
      coefficients = the operators coefficients (c(t)_i)
      F            = a list of transport operators (a list with M transport operators, 
                                                    each is a square matrix, kXk, where k is the latent dynamics
                                                    dimension )
      type_find    = 'median'
      min_far      = 10
      smooth_coeffs= False
      smoothing_params = {'wind':5}
      
  Outputs:
      cur_reco    = dLDS reconstruction of the latent dynamics
      
  """
  if smooth_coeffs:
    coefficients = movmfunc(np.nanmedian, coefficients, window = smoothing_params['wind'], direction = 1)
  

  cur_reco = np.hstack([create_next(latent_dyn, coefficients, F,time_point) for time_point in range(latent_dyn.shape[1]-1)])
  cur_reco = np.hstack([latent_dyn[:,0].reshape((-1,1)),cur_reco])
    

  return cur_reco


def add_labels(ax, xlabel='X', ylabel='Y', zlabel='Z', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], ylabel_params = {},zlabel_params = {}, xlabel_params = {},  title_params = {}):
  """
  This function add labels, titles, limits, etc. to figures;
  Inputs:
      ax      = the subplot to edit
      xlabel  = xlabel
      ylabel  = ylabel
      zlabel  = zlabel (if the figure is 2d please define zlabel = None)
      etc.
  """
  if xlabel != '' and xlabel != None: ax.set_xlabel(xlabel, **xlabel_params)
  if ylabel != '' and ylabel != None:ax.set_ylabel(ylabel, **ylabel_params)
  if zlabel != '' and zlabel != None:ax.set_zlabel(zlabel,**ylabel_params)
  if title != '' and title != None: ax.set_title(title, **title_params)
  if xlim != None: ax.set_xlim(xlim)
  if ylim != None: ax.set_ylim(ylim)
  if zlim != None: ax.set_zlim(zlim)
  
  if (np.array(xticklabels) != None).any(): 
      if len(xticks) == 0: xticks = np.arange(len(xticklabels))
      ax.set_xticks(xticks);
      ax.set_xticklabels(xticklabels);
  if (np.array(yticklabels) != None).any(): 
      if len(yticks) == 0: yticks = np.arange(len(yticklabels)) +0.5
      ax.set_yticks(yticks);
      ax.set_yticklabels(yticklabels);
  if len(legend)       > 0:  ax.legend(legend)



def find_dominant_dyn(coefficients):
    """
    This function finds the # of the most dominant dynamics in each time point. It should be used when comparing to rsLDS
    Input:
        coefficients: np.array of kXT
    Output:
        an array with len T, containing the index of the most dominant sub-dynamic at each time point
    """
    domi = np.argmax(np.abs(coefficients),0)
    return domi  




#%% Saving
  
def check_save_name(save_name, invalid_signs = '!@#$%^&*.,:;', addi_path = [], sep=sep)  :
    """
    Check if the name is valid
    """
    for invalid_sign in invalid_signs:   save_name = save_name.replace(invalid_sign,'_')
    if len(addi_path) == 0:    return save_name
    else:   
        path_name = sep.join(addi_path)
        return path_name +sep +  save_name

def save_file_dynamics(save_name, folders_names,to_save =[], invalid_signs = '!@#$%^&*.,:;', sep  = sep , type_save = '.npy'):
    """
    Save dynamics & model results
    """                  
    save_name = check_save_name(save_name, invalid_signs)
    path_name = sep.join(folders_names)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    if type_save == '.npy':
        if not save_name.endswith('.npy'): save_name = save_name + '.npy'
        np.save(path_name +sep +  save_name, to_save)
    elif type_save == '.pkl':
        if not save_name.endswith('.pkl'): save_name = save_name + '.pkl'
        dill.dump_session(path_name +sep +  save_name)

def saveLoad(opt,filename):
    global calc
    if opt == "save":
        f = open(filename, 'wb')
        pickle.dump(calc, f, 2)
        f.close
     
    elif opt == "load":
        f = open(filename, 'rb')
        calc = pickle.load(f)
    else:
        print('Invalid saveLoad option')
        
def load_vars(folders_names ,  save_name ,sep=sep , ending = '.pkl',full_name = False):
    """
    Load results previously saved
    Example:
        load_vars('' ,  'save_c.pkl' ,sep=sep , ending = '.pkl',full_name = False)
    """
    if full_name: 
        dill.load_session(save_name)    
    else:
        if len(folders_names) > 0: path_name = sep.join(folders_names)
        else: path_name = ''
      
        if not save_name.endswith(ending): save_name = '%s%s'%(save_name , ending)
        dill.load_session(path_name +sep +save_name)

    


def create_colors(len_colors, perm = [0,1,2]):
    """
    Create a set of discrete colors with a one-directional order
    Input: 
        len_colors = number of different colors needed
    Output:
        3 X len_colors matrix decpiting the colors in the cols
    """
    colors = np.vstack([np.linspace(0,1,len_colors),(1-np.linspace(0,1,len_colors))**2,1-np.linspace(0,1,len_colors)])
    colors = colors[perm, :]
    return colors



#%% Plot tricolor curve for Lorenz
    
def min_dist(dotA1, dotA2, dotB1, dotB2, num_sects = 500):
    x_lin = np.linspace(dotA1[0], dotA2[0])
    y_lin = np.linspace(dotA1[1], dotA2[1])
    x_lin_or = np.linspace(dotB1[0], dotB2[0])
    y_lin_or = np.linspace(dotB1[1], dotB2[1])
    dist_list = []
    for pairA_num, pairAx in enumerate(x_lin):
        pairAy = y_lin[pairA_num]
        for pairB_num, pairBx in enumerate(x_lin_or):
            pairBy = y_lin_or[pairB_num]
            dist = (pairAx - pairBx)**2 + (pairAy - pairBy)**2
            dist_list.append(dist)
    return dist_list
    

def find_perpendicular(d1, d2, perp_length = 1, prev_v = [], next_v = [], ref_point = [],choose_meth = 'intersection',initial_point = 'mid',  
                       direction_initial = 'low', return_unchose = False, layer_num = 0):
    """
    This function find the 2 point of the orthogonal vector to a vector defined by points d1,d2
    d1 =                first data point
    d2 =                second data point
    perp_length =       desired width
    prev_v =            previous value of v. Needed only if choose_meth == 'prev'
    next_v =            next value of v. Needed only if choose_meth == 'prev'
    ref_point =         reference point for the 'smooth' case, or for 2nd+ layers
    choose_meth =       'intersection' (eliminate intersections) OR 'smooth' (smoothing with previous prediction) OR 'prev' (eliminate convexity)
    direction_initial = to which direction take the first perp point  
    return_unchose =    whether to return unchosen directions   
    
    """       
    # Check Input    
    if d2[0] == d1[0] and d2[1] == d1[1]:
        raise ValueError('d1 and d2 are the same point')
    
    # Define start point for un-perp curve
    if initial_point == 'mid':
        perp_begin = (np.array(d1) + np.array(d2))/2
        d1_perp = perp_begin
    elif initial_point == 'end':        d1_perp = d2
    elif initial_point == 'start':        d1_perp = d1
    else:        raise NameError('Unknown intial point')       
    
    # If perpendicular direction is according to 'intersection' elimination
    if choose_meth == 'intersection':
        if len(prev_v) > 0:        intersected_curve1 = prev_v
        else:                      intersected_curve1 = d1
        if len(next_v) > 0:        intersected_curve2 = next_v
        else:                      intersected_curve2 = d2
        
    # If a horizontal line       
    if d2[0] == d1[0]:        d2_perp = np.array([d1_perp[0]+perp_length, d1_perp[1]])
    # If a vertical line
    elif d2[1] == d1[1]:        d2_perp = np.array([d1_perp[0], d1_perp[1]+perp_length])
    else:
        m = (d2[1]-d1[1])/(d2[0]-d1[0]) 
        m_per = -1/m                                                   # Slope of perp curve        
        theta1 = np.arctan(m_per)
        theta2 = theta1 + np.pi
        
        # if smoothing
        if choose_meth == 'smooth' or choose_meth == 'intersection':
            if len(ref_point) == 0: 
                smooth_val =[]
            else:                smooth_val = np.array(ref_point)
        
        # if by convexity
        if choose_meth == 'prev':
            if len(prev_v) > 0 and len(next_v) > 0:                     # both sides are provided
                prev_mid_or = (np.array(prev_v) + np.array(next_v))/2
            elif len(prev_v) > 0 and len(next_v) == 0:                  # only the previous side is provided
                prev_mid_or = (np.array(prev_v) + np.array(d2))/2
            elif len(next_v) > 0 and len(prev_v) == 0:                  # only the next side is provided               
                prev_mid_or = (np.array(d1) + np.array(next_v))/2
            else:
                raise ValueError('prev or next should be defined (to detect convexity)!')        

        if choose_meth == 'prev':
            prev_mid = prev_mid_or
        elif choose_meth == 'smooth':
            prev_mid = smooth_val
        elif choose_meth == 'intersection':
            prev_mid = smooth_val
            
        x_shift = perp_length * np.cos(theta1)
        y_shift = perp_length * np.sin(theta1)
        d2_perp1 = np.array([d1_perp[0] + x_shift, d1_perp[1]+ y_shift])            
        
        x_shift2 = perp_length * np.cos(theta2)
        y_shift2 = perp_length * np.sin(theta2)
        d2_perp2 = np.array([d1_perp[0] + x_shift2, d1_perp[1]+ y_shift2])
        options_last = [d2_perp1, d2_perp2]
        
        # Choose the option that goes outside
        if len(prev_mid) > 0:
            
          
            if len(ref_point) > 0 and layer_num > 0:                               # here ref point is a point of a different dynamics layer from which we want to take distance
                dist1 = np.sum((smooth_val - d2_perp1)**2)
                dist2 = np.sum((smooth_val - d2_perp2)**2)
                max_opt = np.argmax([dist1, dist2])

            elif choose_meth == 'intersection':
                dist1 = np.min(min_dist(prev_mid, d2_perp1, intersected_curve1, intersected_curve2))
                dist2 = np.min(min_dist(prev_mid, d2_perp2, intersected_curve1, intersected_curve2))
                max_opt = np.argmax([dist1,dist2]) 
         
            else:
                dist1 = np.sum((prev_mid - d2_perp1)**2)
                dist2 = np.sum((prev_mid - d2_perp2)**2)
                max_opt = np.argmin([dist1,dist2])                  
        else:
        
            if len(ref_point) > 0 and layer_num >0:                               # here ref point is a point of a different dynamics layer from which we want to take distance
                dist1 = np.sum((ref_point - d2_perp1)**2)
                dist2 = np.sum((ref_point - d2_perp2)**2)
                max_opt = np.argmax([dist1, dist2])
             
            elif direction_initial == 'low':
                max_opt = np.argmin([d2_perp1[1], d2_perp2[1]])
            elif direction_initial == 'high':
                max_opt = np.argmax([d2_perp1[1], d2_perp2[1]])
            elif direction_initial == 'right' :
                max_opt = np.argmax([d2_perp1[0], d2_perp2[0]])
            elif direction_initial == 'left':
                max_opt = np.argmin([d2_perp1[0], d2_perp2[0]])

                
            else:
                raise NameError('Invalid direction initial value') 
    
    d2_perp = options_last[max_opt] # take the desired direction
    if return_unchose:
        d2_perp_unchose = options_last[np.abs(1 - max_opt)] 
        return d1_perp, d2_perp, d2_perp_unchose
    return d1_perp, d2_perp


def find_lows_high(coeff_row, latent_dyn,   choose_meth ='intersection',factor_power = 0.9, initial_point = 'start',
                   direction_initial = 'low', return_unchose = False, ref_point = [], layer_num = 0):
    """
    Calculates the coordinates of the 'high' values of a specific kayer
    """
    
    if return_unchose: unchosen_highs = []
    ### Initialize
    x_highs_y_highs = []; x_lows_y_lows = []
    if isinstance(ref_point, np.ndarray):
        if len(ref_point.shape) > 1:
            ref_shape_all = ref_point
        else:
            ref_shape_all = np.array([])
    else:
        ref_shape_all = np.array([])
    # Iterate over time
    for t_num in range(0,latent_dyn.shape[1]-2):   
        d1_coeff = latent_dyn[:,t_num]
        d2_coeff = latent_dyn[:,t_num+1]
        prev_v = latent_dyn[:,t_num-1] 
        next_v = latent_dyn[:,t_num+2]
        c_len = (coeff_row[t_num] + coeff_row[t_num+1])/2

        if len(ref_shape_all) > 0 and ref_shape_all.shape[0] > t_num and layer_num > 0: # and ref_shape_all.shape[1] >1
            ref_point = ref_shape_all[t_num,:]

          
            if len(ref_point) >  0 and layer_num > 0 :  #and t_num  < 3
                 pass

        
        elif t_num > 2 and (choose_meth == 'smooth' or choose_meth == 'intersection'):   
            ref_point  = d2_perp          
        else:              
            ref_point = []       

        
        if return_unchose:  d1_perp, d2_perp, d2_perp_unchosen = find_perpendicular(d1_coeff, d2_coeff,c_len**factor_power, prev_v = prev_v, next_v=next_v,ref_point  = ref_point , choose_meth = choose_meth, initial_point=initial_point, direction_initial =direction_initial, return_unchose = return_unchose,layer_num=layer_num)# c_len
        else:               d1_perp, d2_perp = find_perpendicular(d1_coeff, d2_coeff,c_len**factor_power, prev_v = prev_v, next_v=next_v,ref_point  = ref_point , choose_meth = choose_meth, initial_point=initial_point, direction_initial= direction_initial, return_unchose = return_unchose,layer_num=layer_num)# c_len
        # Add results to results lists
        x_lows_y_lows.append([d1_perp[0],d1_perp[1]])
        x_highs_y_highs.append([d2_perp[0],d2_perp[1]])
        if return_unchose: unchosen_highs.append([d2_perp_unchosen[0],d2_perp_unchosen[1]])
    # return
    if return_unchose: 
        return x_lows_y_lows, x_highs_y_highs, unchosen_highs
    return x_lows_y_lows, x_highs_y_highs        

def spec_corr(v1,v2):
  """
  absolute value of correlation
  """
  corr = np.corrcoef(v1[:],v2[:])
  return np.abs(corr[0,1])


def plot_multi_colors(store_dict,min_time_plot = 0,max_time_plot = -100,  colors = ['green','red','blue'], ax = [],
                      fig = [], alpha = 0.99, smooth_window = 3, factor_power = 0.9, coefficients_n = [], to_scatter = False, 
                      to_scatter_only_one = False ,choose_meth = 'intersection', title = ''):
    """
    store_dict is a dictionary with the high estimation results. 
    example:        
        store_dict , coefficients_n = calculate_high_for_all(coefficients,choose_meth = 'intersection',width_des = width_des, latent_dyn = latent_dyn, direction_initial = direction_initial,factor_power = factor_power, return_unchose=True)
    
    """
    if len(colors) < len(store_dict):                raise ValueError('Not enough colors were provided')
    if isinstance(ax, list) and len(ax) == 0:        fig, ax = plt.subplots(figsize = (20,20))
    for key_counter, (key,set_plot) in enumerate(store_dict.items()):
        if key_counter == 0:
            x_lows_y_lows = store_dict[key][0]
            x_highs_y_highs = store_dict[key][1]
            low_ref =[]
            high_ref = []
        else:
            low_ref = [np.array(x_highs_y_highs)[min_time_plot,0],   np.array(x_highs_y_highs)[min_time_plot,1]]
            high_ref = [np.array(x_highs_y_highs)[max_time_plot,0],np.array(x_highs_y_highs)[max_time_plot,1]]
        if len(coefficients_n) > 0:
            # Define the length of the last perp. 
            c_len = (coefficients_n[key,max_time_plot-1] + coefficients_n[key,max_time_plot])/2
            # Create perp. in the last point            
            d1_p, d2_p =find_perpendicular([np.array(x_lows_y_lows)[max_time_plot-2,0],np.array(x_lows_y_lows)[max_time_plot-2,1]], 
                                           [np.array(x_lows_y_lows)[max_time_plot-1,0],np.array(x_lows_y_lows)[max_time_plot-1,1]], 
                                           perp_length = c_len**factor_power, 
                                           ref_point = high_ref, 
                                           choose_meth = 'intersection',initial_point = 'end')
            # Define the length of the first perp. 
            c_len_start = (coefficients_n[key,min_time_plot-1] + coefficients_n[key,min_time_plot])/2
            # Create perp. in the first point   
            d1_p_start =[np.array(x_highs_y_highs)[min_time_plot,0],np.array(x_highs_y_highs)[min_time_plot,1]]
                                                       
            d2_p_start=  [np.array(x_highs_y_highs)[min_time_plot+1,0],np.array(x_highs_y_highs)[min_time_plot+1,1]]                                                        

            x_lows_y_lows = store_dict[key][0]
            x_highs_y_highs = store_dict[key][1] 

            stack_x = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array([d2_p[0]]), np.array(x_highs_y_highs)[max_time_plot-1:min_time_plot+1:-1,0].flatten(),np.array([d2_p_start[0]])])
            stack_y = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), np.array([d2_p[1]]),np.array(x_highs_y_highs)[max_time_plot-1:min_time_plot+1:-1,1].flatten(),np.array([d2_p_start[1]])])
            
        else:
            stack_x = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array(x_highs_y_highs)[max_time_plot:min_time_plot:,0].flatten()])
            stack_y = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), np.array(x_highs_y_highs)[max_time_plot:min_time_plot:,1].flatten()])
        stack_x_smooth = stack_x
        stack_y_smooth = stack_y
        if key_counter !=0:
            ax.fill(stack_x_smooth, stack_y_smooth, alpha=0.3, facecolor=colors[key_counter], edgecolor=None, zorder=1, snap = True)#
        else:
            ax.fill(stack_x_smooth, stack_y_smooth, alpha=alpha, facecolor=colors[key_counter], edgecolor=None, zorder=1, snap = True)#

    if to_scatter or (to_scatter_only_one and key == np.max(list(store_dict.keys()))):
        

          ax.scatter(np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), c = 'black', alpha = alpha, s = 45)

    remove_edges(ax)
    if not title  == '':
        ax.set_title(title, fontsize = 20)
    
  
def calculate_high_for_all(coefficients, choose_meth = 'both', same_width = True,factor_power = 0.9, width_des = 0.7, 
                           initial_point = 'start', latent_dyn = [],
                          direction_initial = 'low', return_unchose = False, type_norm = 'norm',min_width =0.01):
    """
    Create the dictionary to store results
    """
    if len(latent_dyn) == 0: raise ValueError('Empty latent dyn was provided')
    
    # Coeffs normalization
    coefficients_n = norm_coeffs(coefficients, type_norm, same_width = same_width, width_des = width_des,factor_power =factor_power,min_width=min_width )
    
    # Initialization
    store_dict      = {}
    dyn_use         = latent_dyn
    ref_point       = []
    
    for row in range(coefficients_n.shape[0]):

        coeff_row = coefficients_n[row,:]
        # Store the results for each layer
        if return_unchose:
            x_lows_y_lows, x_highs_y_highs,x_highs_y_highs2 = find_lows_high(coeff_row,dyn_use, choose_meth = choose_meth, factor_power=factor_power, 
                                                                             initial_point = initial_point,direction_initial = direction_initial,
                                                                             return_unchose = return_unchose, ref_point = ref_point,layer_num = row )             
            store_dict[row] = [x_lows_y_lows, x_highs_y_highs,x_highs_y_highs2]
        else:
            x_lows_y_lows, x_highs_y_highs = find_lows_high(coeff_row,dyn_use, choose_meth = choose_meth, factor_power=factor_power, 
                                                            initial_point = initial_point, direction_initial = direction_initial ,
                                                            return_unchose = return_unchose,ref_point = ref_point ,layer_num=row)             
            store_dict[row] = [x_lows_y_lows, x_highs_y_highs]
        # Update the reference points    
        if initial_point == 'mid':
            dyn_use = np.array(x_highs_y_highs).T
            dyn_use = (dyn_use[:,1:] + dyn_use[:,:-1])/2
            dyn_use = np.hstack([latent_dyn[:,:2], dyn_use, latent_dyn[:,-2:]])
        else:
            dyn_use = np.array(x_highs_y_highs).T

        ref_point = np.array(x_lows_y_lows)
    return store_dict, coefficients_n    


    
#%% Plot 2d axis of coeffs for fig 2

def plot_3d_dyn_basis(F, coefficients, projection = [0,-1], ax = [],  fig = [], time_emph = [], n_times = 5,
                      type_plot = 'quiver',range_p = 10,s=200, w = 0.05/3, alpha0 = 0.3, 
                      time_emph_text  = [10, 20, 30, 50,80, 100,200,300,400,500], turn_off_back = True, lim1 = np.nan, 
                      ax_qui = [], 
                      ax_base = [], to_title = True, loc_title = 'title', include_bar =True, axs_basis_colored = [],
                      colors_dyns = np.array(['r','g','b','yellow']) , plot_dyn_by_colorbase = False, remove_edges_ax = False, include_dynamics = False,
                      latent_dyn = [],fontsize_times = 16,delta_text = 0.1, delta_text_y = 0,delta_text_z = 0, 
                      new_colors = True, include_quiver = True, base_narrow = True,colors = [],color_by_dom = False,
                      quiver_3d = False, s_all = 10,to_remove_edge = True, to_grid = False, cons_color = False):   
    """
    ax = subplot to plot coefficients over time
    colors = should be a mat of k X 3
    """
    if not F[0].shape[0] ==3: quiver_3d = False
    if len(colors) ==0:    
        if color_by_dom:
            color_sig_tmp = find_dominant_dyn(np.abs(coefficients))
            colors = colors_dyns[color_sig_tmp]
            colors_base = np.zeros(coefficients.shape[1])
            
        else:
            colors_base = np.linspace(0,1,coefficients.shape[1]).reshape((-1,1))    
            colors = np.hstack([colors_base, 1-colors_base, colors_base**2])    

    if isinstance(ax,list) and len(ax) == 0:
        if len(F) == 3:        fig, ax = plt.subplots(subplot_kw={'projection':'3d'}, figsize= (10,10))
        elif len(F) == 2:      fig, ax = plt.subplots(figsize= (10,10))
        else: raise ValueError('Invalid dim for F')
    if len(time_emph) == 0: 
        time_emph =np.linspace(0,coefficients.shape[1]-2, n_times+1)[1:].astype(int)

    if include_dynamics:
        
        if len(latent_dyn) == 0: raise ValueError('You should provide latent dyn as input if "include dynamics" it True')
        if len(F[0]) == 3:
            fig_dyn,ax_dyn = plt.subplots(figsize = (15,15),subplot_kw={'projection':'3d'})
            if new_colors:

                ax_dyn.scatter(latent_dyn[0,:len(colors_base)], latent_dyn[1,:len(colors_base)],latent_dyn[2,:len(colors_base)], color = colors,alpha = 0.3)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph],latent_dyn[2,time_emph], c = 'black', s = 300)
            else:
                c_sig = np.arange(latent_dyn.shape[1])
                ax_dyn.scatter(latent_dyn[0,:], latent_dyn[1,:],latent_dyn[2,:], c = c_sig,alpha = 0.3, cmap = 'viridis', s = 100)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph],latent_dyn[2,time_emph], c = c_sig[time_emph], s = 300, cmap = 'viridis')
            [ax_dyn.text(latent_dyn[0,t] + delta_text,latent_dyn[1,t]+delta_text_y,latent_dyn[2,t]+delta_text_z, 't = %s'%str(t), fontsize =fontsize_times, fontweight = 'bold') for t in time_emph]
            ax_dyn.set_axis_off()
            
        else:
            fig_dyn,ax_dyn = plt.subplots(figsize = (10,10))
            if new_colors:
                
                
                ax_dyn.scatter(latent_dyn[0,:len(colors_base)], latent_dyn[1,:len(colors_base)], color = colors,alpha = 0.3, s = 50)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph], c = 'black', s = 200)
            else:
                c_sig = np.arange(latent_dyn.shape[1])
                ax_dyn.scatter(latent_dyn[0,:], latent_dyn[1,:], c = c_sig,alpha = 0.3, cmap = 'viridis', s = 100)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph], c = c_sig[time_emph], s = 300, cmap = 'viridis')
            [ax_dyn.text(latent_dyn[0,t] + delta_text,latent_dyn[1,t]+delta_text_y, 't = %s'%str(t), fontsize =fontsize_times, fontweight = 'bold') for t in time_emph]
            remove_edges(ax_dyn)

    if len(F[0]) == 3: 
        if quiver_3d:
            if type_plot == 'streamplot': 
                type_plot = 'quiver'
                print('If quiver_3d then type_plot need to be quiver (currently is streamplot)')
            if  include_quiver:
                if isinstance(ax_qui, list) and len(ax_qui)== 0:         
                    fig_qui, ax_qui = plt.subplots(1,len(time_emph), figsize= (7*len(time_emph),5) ,subplot_kw={'projection':'3d'})
            if isinstance(ax_base, list) and len(ax_base)==0:         
                if base_narrow:
                    fig_base, ax_base = plt.subplots(len(F),1, figsize= (5,7*len(F)) ,subplot_kw={'projection':'3d'})
                else:
                    fig_base, ax_base = plt.subplots(1,len(F),figsize= (7*len(F),5 ),subplot_kw={'projection':'3d'})
        else:
          
            F = [f[:,projection] for f in F]
            F = [f[projection, :] for f in F]
            if  include_quiver:
                if isinstance(ax_qui, list) and len(ax_qui)== 0:         fig_qui, ax_qui = plt.subplots(1,len(time_emph), figsize= (7*len(time_emph),5) )
            if isinstance(ax_base, list) and len(ax_base)==0:         
                if base_narrow:
                    fig_base, ax_base = plt.subplots(len(F),1, figsize= (5,7*len(F)) )
                else:
                    fig_base, ax_base = plt.subplots(1,len(F),figsize= (7*len(F),5 ))

    elif len(F[0]) == 2:  
        if isinstance(ax_qui, list) and len(ax_qui)==0:     fig_qui, ax_qui = plt.subplots(1,len(time_emph), figsize= (7*len(time_emph),5))
        if isinstance(ax_base, list) and len(ax_base)==0:   
            if base_narrow:
                fig_base, ax_base = plt.subplots(len(F), 1,figsize= (5,7*len(F) ))
            else:
                fig_base, ax_base = plt.subplots(1,len(F),figsize= (7*len(F),5 ))
    if len(F[0]) == 3:

        cmap = matplotlib.cm.get_cmap('viridis')
        if new_colors:

            ax.scatter(coefficients[0,:],coefficients[1,:],coefficients[2,:], c = colors, alpha = alpha0, s = s_all)

            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph],coefficients[2,time_emph], c = 'black',
                       s = s)

            [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p, color =colors[time_point] ,
                       w = w, ax = ax_qui[i], quiver_3d = quiver_3d, cons_color = cons_color) for i, time_point in enumerate(time_emph)]
        else:
            
            cmap = matplotlib.cm.get_cmap('viridis')
            colors_base = np.arange(coefficients.shape[1])
            ax.scatter(coefficients[0,:],coefficients[1,:],coefficients[2,:], c = colors_base, alpha = alpha0, s = s_all)
            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph],coefficients[2,time_emph], c = 'black', s = s, alpha = np.min([alpha0*2, 1]))
            if include_quiver:
                [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p,  
                           color = cmap(time_point/colors.shape[0])  ,
                       w = w, ax = ax_qui[i], quiver_3d = quiver_3d, cons_color = cons_color) for i, time_point in enumerate(time_emph)]
            
        if to_title and include_quiver:
            if loc_title == 'title':
                [ax_qui[i].set_title('t = ' + str(time_point), fontsize =fontsize_times*3 , fontweight = 'bold') for i, time_point in enumerate(time_emph)]
            else:
                [ax_qui[i].set_ylabel('t = ' + str(time_point), fontsize =fontsize_times, fontweight = 'bold') for i, time_point in enumerate(time_emph)]

        [ax.text(coefficients[0,time_point]+delta_text,coefficients[1,time_point]+delta_text_y,coefficients[2,time_point]+delta_text_z,'t = ' + str(time_point), fontsize =fontsize_times, fontweight = 'bold') for time_point in time_emph_text]
        ax.set_xlabel('f1');ax.set_ylabel('f2');ax.set_zlabel('f3');

    else:    


        cmap = matplotlib.cm.get_cmap('viridis')
        if new_colors:
            ax.scatter(coefficients[0,:],coefficients[1,:], c = colors, alpha = alpha0, s = s_all)
            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph], c = colors[time_emph], s = s)
            if include_quiver:
                [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p, color =colors[time_point] , w = w, ax = ax_qui[i],cons_color=cons_color ) for i, time_point in enumerate(time_emph)]
        else:
            colors_base = np.arange(coefficients.shape[1])
            ax.scatter(coefficients[0,:],coefficients[1,:], c = colors_base, alpha = alpha0, s = s_all)
            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph], c = colors_base[time_emph], s = s)
            if include_quiver:
                [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p, color = cmap(time_point/colors.shape[0]) , w = w, ax = ax_qui[i],cons_color= cons_color ) for i, time_point in enumerate(time_emph)]
        if latent_dyn.shape[0] == 3:
            [ax.text(coefficients[0,time_point]+delta_text,coefficients[1,time_point]+delta_text_y, coefficients[2,time_point]+delta_text_z,'t = ' + str(time_point),fontsize = fontsize_times ,fontweight = 'bold') for time_point in time_emph_text]
    
        else:
            [ax.text(coefficients[0,time_point]+delta_text,coefficients[1,time_point]+delta_text_y,'t = ' + str(time_point),fontsize = fontsize_times ,fontweight = 'bold') for time_point in time_emph_text]
        
        ax.set_xlabel('f1');ax.set_ylabel('f2');
        if  remove_edges_ax:        remove_edges(ax)

        if to_title and  include_quiver:
            if loc_title == 'title':
                [ax_qui[i].set_title('t = ' + str(time_point), fontsize = 30) for i, time_point in enumerate(time_emph)]
            else:
                [ax_qui[i].set_ylabel('t = ' + str(time_point), fontsize = 30) for i, time_point in enumerate(time_emph)]
    if to_remove_edge:  
        if  include_quiver:        [remove_edges(ax_spec) for ax_spec in ax_qui]
        [remove_edges(ax_spec) for ax_spec in ax_base]
        ax.set_xticks([])
        ax.set_yticks([])
        if quiver_3d:
            ax.set_zticks([])    
    [quiver_plot(f,-range_p, range_p, -range_p, range_p, ax = ax_base[f_num],chosen_color =  'black', w = w, type_plot = type_plot,cons_color =cons_color,quiver_3d = quiver_3d ) for f_num, f in enumerate(F)]
    [ax_base_spec.set_title('f %s'%str(i), fontsize = 16) for i, ax_base_spec in enumerate(ax_base)]
    

    if turn_off_back and  len(F) == 3:
      ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
      ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
      ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if not to_grid and  len(F) == 3:      
      ax.grid(False)
      ax.set_zticks([])      
      ax.xaxis._axinfo['juggled'] = (0,0,0)
      ax.yaxis._axinfo['juggled'] = (1,1,1)
      ax.zaxis._axinfo['juggled'] = (2,2,2)
    if not np.isnan(lim1):
        ax.set_xlim([-lim1,lim1])
        ax.set_ylim([-lim1,lim1])
        
        
    if include_bar:
        if base_narrow:
            fig_all_all, ax_all_all = plt.subplots(len(time_emph),1, figsize = (6,len(time_emph)*7))
        else:
            ax_all_all = []
        add_bar_dynamics(coefficients, ax_all_all = ax_all_all, min_max_points = time_emph, colors = colors_dyns, 
                         centralize = True)

        if isinstance( axs_basis_colored ,list) and len( axs_basis_colored ) == 0:
            if base_narrow:
                if quiver_3d: fig_basis_colored , axs_basis_colored = plt.subplots( len(F),1,figsize = (5,6*len(F)),subplot_kw={'projection':'3d'})
                else: fig_basis_colored , axs_basis_colored = plt.subplots( len(F),1,figsize = (5,6*len(F)))
                
            else:
                if quiver_3d: fig_basis_colored , axs_basis_colored = plt.subplots( 1, len(F), figsize = (6*len(F),5),subplot_kw={'projection':'3d'})
                else: fig_basis_colored , axs_basis_colored = plt.subplots( 1, len(F), figsize = (6*len(F),5))
        [quiver_plot(f,-range_p, range_p, -range_p, range_p, ax = axs_basis_colored[f_num],alpha = 0.7, chosen_color =  colors_dyns[f_num], w = w, type_plot = type_plot, cons_color = cons_color, quiver_3d=quiver_3d ) for f_num, f in enumerate(F)]
        [remove_edges(ax_spec) for ax_spec in axs_basis_colored]
        if quiver_3d:        [ax.set_zticks([]) for ax in axs_basis_colored]
       

            
def plot_reco_dyn(coefficients, F, time_point, type_plot = 'quiver', range_p = 10, color = 'black',
                  w = 0.05/3, ax = [], cons_color= False, to_remove_edges = False, projection = [0,1], 
                  return_artist = False,
                  xlabel = 'x',ylabel = 'y',quiver_3d = False):
    if isinstance(ax,list) and len(ax) == 0:
  
        fig, ax = plt.subplots()

    if len(F) == 3:
        merge_dyn_at_t_break = coefficients[0,time_point] * F[0]+coefficients[1,time_point] * F[1]+coefficients[2,time_point] * F[2]

        if not quiver_3d:      

            merge_dyn_at_t_break = merge_dyn_at_t_break[:, projection]
            merge_dyn_at_t_break = merge_dyn_at_t_break[projection,:]

    elif len(F) == 2:
        merge_dyn_at_t_break = coefficients[0,time_point] * F[0]+coefficients[1,time_point] * F[1]

    art = quiver_plot(sub_dyn = merge_dyn_at_t_break, chosen_color = color,  xmin = -range_p, 
                      xmax = range_p, ymin= -range_p,ymax= range_p, ax = ax, w = w, type_plot=type_plot,
                      cons_color= cons_color, return_artist = return_artist, xlabel = xlabel, ylabel = ylabel,
                      quiver_3d = quiver_3d)
    if to_remove_edges: remove_edges(ax)
    if return_artist:
        return art



def plot_c_space(coefficients,latent_dyn = [], axs = [], fig = [], xlim = [-50,50], ylim = [-50,50], add_midline = True, d3 = True, cmap = 'winter', color_sig = [], 
                 title = '', times_plot= [], cmap_f = []):
    if len(times_plot) > 0 and isinstance(cmap_f, list): cmap_f = plt.cm.get_cmap(cmap)
    if len(color_sig) == 0:    color_sig = latent_dyn[0,:-1]
    if isinstance(axs, list) and len(axs) == 0:
      
        if coefficients.shape[0] == 3:
            fig, axs =  plt.subplots(figsize = (15,15),subplot_kw={'projection':'3d'})
            d3 = True
            h = axs.scatter(coefficients[0,:], coefficients[1,:],coefficients[2,:], c = color_sig, cmap = cmap)
            if len(times_plot) > 0:
                axs.scatter(coefficients[0,times_plot], coefficients[1,times_plot], coefficients[2,times_plot], c =cmap_f(color_sig[times_plot]/np.max(color_sig)),s = 500 )
        elif coefficients.shape[0] == 2:
            if d3:       
                fig, axs =  plt.subplots(figsize = (15,15),subplot_kw={'projection':'3d'})
                h = axs.scatter(coefficients[0,:], coefficients[1,:], np.arange(coefficients.shape[1]), c = color_sig, cmap = cmap)
                if len(times_plot) > 0:
                    zax = np.arange(coefficients.shape[1])
                    axs.scatter(coefficients[0,times_plot], coefficients[1,times_plot], zax[times_plot], c =cmap_f(color_sig[times_plot]/np.max(color_sig)),s = 500 )
            else:
                fig, axs =  plt.subplots(figsize = (15,15))
                h = axs.scatter(coefficients[0,:], coefficients[1,:], c = color_sig, cmap = cmap)
                if len(times_plot) > 0:
                    axs.scatter(coefficients[0,times_plot], coefficients[1,times_plot],  c =cmap_f(color_sig[times_plot]/np.max(color_sig)),s = 500 )
        else:
            print('Invalid coefficients shape in axis 0')
    
    if len(xlim) > 0:    axs.set_xlim(xlim)
    if len(ylim) > 0:    axs.set_ylim(ylim)
    if not isinstance(fig, list):    fig.colorbar(h)

    if d3: 
        axs.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.grid(False)
        axs.set_axis_off()
    if add_midline:
        if d3:
            axs.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs.plot([0,0],[np.min(coefficients[1,:]),np.max(coefficients[1,:])],[0,0], color = 'black', ls = '--', alpha = 0.3)
            axs.plot([np.min(coefficients[0,:]),np.nanmax(coefficients[0,:])],[0,0],[0,0], color = 'black', ls = '--', alpha = 0.3)
            axs.plot([0,0],[0,0],[0,1.3*coefficients.shape[1]],color = 'black', alpha = 0.3, ls = '--')
            axs.view_init(elev=15, azim=30)
        else:
            axs.axhline(0, color = 'black', ls = '--', alpha = 0.3)
            axs.axvline(0, color = 'black', ls = '--', alpha = 0.3)
    if len(title) > 0:
        axs.set_title(title)

    

        
