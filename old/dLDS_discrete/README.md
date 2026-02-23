**The discrete model described in:** _Noga Mudrik*, Yenho Chen*, Eva Yezerets, Christopher Rozell, Adam Charles. "Decomposed Linear Dynamical Systems (dLDS) for learning the latent components of neural dynamics". 2022_


Learning interpretable representations of neural dynamics at a population level is
a crucial first step to understanding how neural activity patterns over time relate
to perception and behavior. Models of neural dynamics often focus on either
low-dimensional projections of neural activity, or on learning dynamical systems
that explicitly relate to the neural state over time. We discuss how these two
approaches are interrelated by considering dynamical systems as representative of
flows on a low-dimensional manifold. Building on this concept, we propose a new
decomposed dynamical system model that represents complex nonstationary and
nonlinear dynamics of time-series data as a sparse combination of simpler, more
interpretable components. The decomposed nature of the dynamics generalizes
over previous switched approaches and enables modeling of overlapping and
non-stationary drifts in the dynamics. We further present a dictionary learning-
driven approach to model fitting, where we leverage recent results in tracking sparse
vectors over time. We demonstrate that our model can learn efficient representations
and smoothly transition between dynamical modes in both continuous-time and
discrete-time examples. We show results on low-dimensional linear and nonlinear
attractors to demonstrate that decomposed systems can well approximate nonlinear
dynamics. Additionally, we apply our model to C. elegans data, illustrating a
diversity of dynamics that is obscured when classified into discrete states.

# Installation Instructions:
      1. (if itertools not installed): sudo pip3 install more-itertools [in the cmd]
      2. (if pickle not installed):    pip install pickle-mixin         [in the cmd]
      3. !pip install dLDS-discrete                                     [in the cmd]
      4. from dlds_discrete import main_functions                       [in Python console]
      5. from dlds_discrete.main_functions import *                     [in Python console]
      6. Use any function from the ones described below
      
      

## Main Useful Functions:

### 1. create_dynamics:
_create sample dynamics_



**create_dynamics**_(type_dyn = 'cyl', max_time = 1000, dt = 0.01, params_ex = {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2})_

#### Detailed Description:
      Create ground truth dynamics. 
      Inputs:
          type_dyn          = Can be 'cyl', 'lorenz','FHN', 'multi_cyl', 'torus', 'circ2d', 'spiral'
          max_time          = integer. Number of time points for the dynamics. Relevant only if data is empty;
          dt                = time interval for the dynamics.
          params_ex         = dictionary of parameters for the dynamics.  {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}):
    
      
      Outputs:
          dynamics: k X T matrix of the dynamics 




### 2. train_model_include_D:
_main function to train the model._

**train_model_include_D**_(max_time = 500, dt = 0.1, dynamics_type = 'cyl',num_subdyns = 3, 
                          error_reco = np.inf,  data = [], step_f = 30, GD_decay = 0.85, max_error = 1e-3, 
                          max_iter = 3000, F = [], coefficients = [], params= {'update_c_type':'inv','reg_term':0,'smooth_term':0}, 
                          epsilon_error_change = 10**(-5), D = [], x_former =[], latent_dim = None, include_D  = False,step_D = 30, reg1=0,reg_f =0 , 
                          max_data_reco = 1e-3,  sigma_mix_f = 0.1,  action_along_time = 'median', to_print = True, seed = 0, seed_f = 0, 
                          normalize_eig  = True,  params_ex = {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}, 
                          init_distant_F = False,max_corr = 0.1, decaying_reg = 1, other_params_c={}, include_last_up = False)_

#### Detailed Description:
      max_time         = Number of time points for the dynamics. Relevant only if data is empty;
      dt               =  time interval for the dynamics
      dynamics_type    = type of the dynamics. Can be 'cyl', 'lorenz','FHN', 'multi_cyl', 'torus', 'circ2d', 'spiral'
      num_subdyns      = number of sub-dynamics
      error_reco       = intial error for the reconstruction (do not touch)
      data             = if one wants to use a pre define groud-truth dynamics. If not empty - it overwrites max_time, dt, and dynamics_type
      step_f           = initial step size for GD on the sub-dynamics
      GD_decay         = Gradient descent decay rate
      max_error        = Threshold for the model error. If the model arrives at a lower reconstruction error - the training ends.
      max_iter         = # of max. iterations for training the model
      F                = pre-defined sub-dynamics. Keep empty if random.
      coefficients     = pre-defined coefficients. Keep empty if random.
      params           = dictionary that includes info about the regularization and coefficients solver. e.g. {'update_c_type':'inv','reg_term':0,'smooth_term':0}
      epsilon_error_change = check if the sub-dynamics do not change by at least epsilon_error_change, for at least 5 last iterations. Otherwise - add noise to f
      D                = pre-defined D matrix (keep empty if D = I)
      latent_dim       =  If D != I, it is the pre-defined latent dynamics.
      include_D        = If True -> D !=I; If False -> D = I
      step_D           = GD step for updating D, only if include_D is true
      reg1             = if include_D is true -> L1 regularization on D
      reg_f            = if include_D is true ->  Frobenius norm regularization on D
      max_data_reco    = if include_D is true -> threshold for the error on the reconstruction of the data (continue training if the error (y - Dx)^2 > max_data_reco)
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
      
* example call (for Lorenz, w. 3 operators):       train_model_include_D(10, 0.01, 'lorenz', 3, GD_decay = 0.99)



### 3. create_reco: 
_create the dynamics reconstruction using the operators and coefficients obtained by dLDS (F, c)._


**create_reco**_(latent_dyn,coefficients, F, type_find = 'median',min_far =10, smooth_coeffs = False,
          smoothing_params = {'wind':5})_
#### Detailed Description:                
                  This function creates the reconstruction 
                  Inputs:
                      latent_dyn   = the ground truth latent dynamics
                      coefficients = the operators coefficients ({$c(t)_i})
                      F            = a list of transport operators (a list with M transport operators, 
                                                                    each is a square matrix, kXk, where k is the latent dynamics
                                                                    dimension )
                      type_find    = 'median'
                      min_far      = 10
                      smooth_coeffs= False
                      smoothing_params = {'wind':5}

                  Outputs:
                      cur_reco    = dLDS reconstruction of the latent dynamics
                      


### 4. visualize_dyn:
_visualization of the dynamics, with various coloring options_ 
     

**visualize_dyn**_(dyn,ax = [], params_plot = {},turn_off_back = False, marker_size = 10, include_line = False, 
            color_sig = [],cmap = 'cool', return_fig = False, color_by_dominant = False, coefficients =[],
            figsize = (5,5),colorbar = False, colors = [],vmin = None,vmax = None, color_mix = False, alpha = 0.4,
            colors_dyns = np.array(['r','g','b','yellow']), add_text = 't ', text_points = [],fontsize_times = 18, 
            marker = "o",delta_text = 0.5, color_for_0 =None, legend = [],fig = [],return_mappable = False)_
#### Detailed Description:                
              Inputs:
                   dyn          = dynamics to plot. Should be a np.array with size k X T
                   ax           = the subplot to plot in. (optional). If empty list -> the function will create a subplot
                   params_plot  = additional parameters for the plotting (optional). Can include plotting-related keys like xlabel, ylabel, title, etc.
                   turn_off_back= disable backgroud of the plot? (optional). Boolean
                   marker_size  = marker size of the plot (optional). Integer
                   include_line = add a curve to the plot (in addition to the scatter plot). Boolean
                   color_sig    = the color signal. 
                                      If empty and color_by_dominant is true - color by the dominant dynamics. 
                                      If empty and not color_by_dominant - color by time.
                   cmap         = color map
                   colors       = if not empty -> pre-defined colors for the different sub-dynamics. 
                                  If empty -> colors are according to the cmap.
                   color_mix    = relevant only if  color_by_dominant is True. In this case the colors need to be in the form of [r,g,b]
               Output:   
                   h (only if return_fig) -> returns the figure   
                
