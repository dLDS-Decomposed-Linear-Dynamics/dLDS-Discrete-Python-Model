"""
Decomposed Linear Dynamical Systems (dLDS) for learning the latent components of neural dynamics
@code author: noga mudrik
"""


#%% Call Main Function


from main_functions import * 

update_c_type = 'inv' # can be also lasso or smooth
step_f = 30
step_f_original = step_f
num_subdyns = 3;
dynamic_type ='cyl'
max_time = 50
dt = 0.1
noise = 0
speed_change = 0
is_D_I = True
reg_term = 0.01
max_iter = 6000
num_iter = max_iter
GD_decay = 0.99
max_error =  1e-8 
max_data_reco = max_error
acumulated_error = False
error_order_initial = 2
error_step_add = 120
error_step_max = 20
error_step_max_display = error_step_max -2 #+ 2
error_step_add = 120
epsilon_error_change= 10**(-8)
sigma_mix_f = 0.1
weights_orders = np.linspace(1,2**error_step_max,error_step_max)[::-1]
grad_vec = np.linspace(0.99,1.01,error_step_max)

action_along_time = 'median' #can be mean

latent_dyn            = create_dynamics(type_dyn = dynamic_type, max_time = max_time, dt = dt)
F                     = [init_mat((latent_dyn.shape[0], latent_dyn.shape[0]),normalize=True) for i in range(num_subdyns)]
coefficients          = init_mat((num_subdyns,latent_dyn.shape[1]-1))
include_D             = False

error_reco            = np.inf
error_reco_all        = np.inf*np.ones((1,error_step_max))
cur_reco              = create_reco(latent_dyn=latent_dyn, coefficients= coefficients, F=F)
error_reco_array      = np.inf*np.ones((1,max(error_step_max_display,error_step_max)))
error_reco_array_med  = np.inf*np.ones((1,max(error_step_max_display,error_step_max)))
error_order           = error_step_max 

seed_f                = 0
same_c                = False
data = []



dyn_radius = 5
dyn_num_cyls = 5
dyn_bias = 0
addition_save = []
to_print = True
name_auto = False
to_save_without_ask = False
ylim_small = [-15,15]

seed_f                = 0
data =[]

reg_term =0
smooth_term = 0 
noise_max = 1
max_error =  1e-8 
include_D             = False
exp_power = 0.1

start_from_c = False

start_sparse_c =   False
init_distant_F =  False
max_corr = 0.1
decaying_reg = 0.999
normalize_eig = True
params_ex = {'radius':dyn_radius , 'num_cyls': dyn_num_cyls, 'bias':dyn_bias,'exp_power':exp_power}

bias_term = False
center_dynamics = False
bias_out = False
noise_interval = 0.1



params_update_c = {'reg_term': reg_term, 'update_c_type':update_c_type,'smooth_term':smooth_term}




width_des = 0.6
t_break = 280
factor_power = 0.3
quarter_initial = 'low'
smooth_window = 3
colors = [[1,0.1,0],[0.1,0,1], [0,1,0]]
s_scatter = 200
start_run = 30
plot_movemean  = False
max_time_plot = 500
colors_dyn = ['r','g','b']
n_samples_range = np.arange(5,106,20)
num_noises = 5
max_dyn =5

init_distant_F                                       = True
step_f = 50
epsilon_error_change= 10**(-8) 


action_along_time = 'median'  #can also be 'mean'

error_reco            = np.inf

error_reco_array      = []