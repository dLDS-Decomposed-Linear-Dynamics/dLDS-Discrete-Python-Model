"""
Decomposed Linear Dynamical Systems (dLDS) for learning the latent components of neural dynamics
@code author: noga mudrik
"""

#%% Imports:

from importlib import reload  

import main_functions
main_functions = reload(main_functions)
from main_functions import * 
from datetime import date


"""
Parameters
"""
exec(open('create_params.py').read())

addi_save         = date.today().strftime('%d%m%y') # For saving
if 'addition_save' not in locals():    addition_save = []

update_c_types                                       = ['inv'] #['spgl1'] 
num_iters                                            = [10]
max_iter                                             = 6000
is_D_I                                               = True


"""
Parameters to choose
"""
dt                                                  = float(input('dt (rec for Lorenz 0.01, rec for FHN 0.2)'))
max_time                                            = float(input('max time (rec for Lorenz 10, rec for FHN 200)'))
dynamic_type                                        = input('dynamic type (e.g. lorenz, FHN)')
addi_name                                           = input('additional name id')
num_subdyns                                         = [int(input('num_dyns (m)'))]
include_last_up                                     = str2bool(input('include last up? (for FHN reg)'))
reg_vals_new                                        = [float(input('reg_val_input (tau)'))]
addition_save.append(addi_save)
latent_dyn                                          = create_dynamics(type_dyn = dynamic_type, max_time = max_time, dt = dt)
include_D                                           = False                    
to_load                                             = False



name_auto             = True
normalize_eig         = True
to_print              = False
seed_f                = 0
dt_range              = np.linspace(0.001, 1, 20)
exp_power             = 0.1

"""
Runnining over the parameters
"""
for num_iter in num_iters:
    for reg_term in reg_vals_new:     
        for update_c_type in update_c_types :
                for num_subs in num_subdyns:
                    to_save_without_ask   = True                                      
                    sigma_mix_f           = 0.1                
                    F                     = [init_mat((latent_dyn.shape[0], latent_dyn.shape[0]),normalize=True) for i in range(num_subs)]
                    coefficients          = init_mat((num_subs,latent_dyn.shape[1]-1))
                    save_name             = '%s_%gsub%greg%s_iters%s'%(dynamic_type,num_subs,reg_term, update_c_type, str(num_iter))
                    data                  = latent_dyn

                    params_update_c = {'reg_term': reg_term, 'update_c_type':update_c_type,'smooth_term' :smooth_term, 'num_iters': num_iter, 'threshkind':'soft'}
                
                    if to_save_without_ask: to_save = True
                    else: to_save = str2bool(input('To save?'))

                    coefficients, F, latent_dyn, error_reco_array, D = train_model_include_D(max_time , dt ,  dynamic_type, num_subdyns = num_subs, 
                                                                                       data = data, step_f = step_f, GD_decay =  GD_decay, 
                                                                                       max_error = max_error, max_iter = max_iter, 
                                                                                       include_D = include_D, seed_f = seed_f, 
                                                                                       normalize_eig = normalize_eig, 
                                                                                       to_print = to_print, params = params_update_c )                         
                    if to_save: 
                        if name_auto:                                pass
                        else:                                        save_name = input('save_name')
                        save_dict = {'F':F, 'coefficients':coefficients, 'latent_dyn': latent_dyn, 'max_time': max_time, 'dt':dt,'dyn_type':dynamic_type,
                                     'error_reco_array' :error_reco_array, 'D':D}          
                        save_file_dynamics(save_name, ['main_folder_results', dynamic_type, 'clean%s'%addi_name,update_c_type ]+addition_save, save_dict )
                        save_file_dynamics(save_name, ['main_folder_results' ,dynamic_type, 'clean%s'%addi_name,update_c_type ]+addition_save, [], type_save = '.pkl' )
    
