# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:44:19 2026

@author: noga mudrik
"""
# -*- coding: utf-8 -*-
"""
dLDS Visualization
==================

Plotting functions for dLDS results.

Usage:
    from dlds.viz import plot_f, plot_coefficients, plot_reconstruction, plot_error_history
    
    plot_f(result.F, path_save='figures/')
    plot_coefficients(result.dynamic_coefficients, path_save='figures/')
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')


def plot_3d(mat, params_fig = {}, fig = [], ax = [], params_plot = {}, type_plot = 'plot', to_return = False):
    """
    Plot 3D data.

    Parameters:
    - mat (numpy.ndarray): 3D data to be plotted.
    - params_fig (dict): Additional parameters for creating the figure.
    - fig (matplotlib.figure.Figure): Existing figure to use (optional).
    - ax (numpy.ndarray): Existing 3D subplot axes to use (optional).
    - params_plot (dict): Additional parameters for the plot.
    - type_plot (str): Type of 3D plot ('plot' for line plot, 'scatter' for scatter plot).

    Returns:
    - fig (matplotlib.figure.Figure): The created or existing figure.
    - ax (numpy.ndarray): The created or existing 3D subplot axes.
    """ 
    if checkEmptyList(ax):
        fig, ax = create_3d_ax(1,1, params_fig)
    if type_plot == 'plot':    
        scatter = ax.plot(mat[0], mat[1], mat[2], **params_plot)
    else:
        scatter = ax.scatter(mat[0], mat[1], mat[2], **params_plot)
    if to_return:
        return scatter
    

def checkEmptyList(obj):
    """
    Parameters
    ----------
    obj : any type

    Returns
    -------
    Boolean variable (whether obj is a list)
    """
    return isinstance(obj, list) and len(obj) == 0

    
def create_3d_ax(num_rows, num_cols, params = {}):
    """
    Create a 3D subplot grid.

    Parameters:
    - num_rows (int): Number of rows in the subplot grid.
    - num_cols (int): Number of columns in the subplot grid.
    - params (dict): Additional parameters to pass to plt.subplots.

    Returns:
    - fig (matplotlib.figure.Figure): The created figure.
    - ax (numpy.ndarray): The created 3D subplot axes.
    """
    fig, ax = plt.subplots(num_rows, num_cols, subplot_kw = {'projection': '3d'}, **params)
    return  fig, ax


def checkEmptyList(obj):
    """
    Parameters
    ----------
    obj : any type

    Returns
    -------
    Boolean variable (whether obj is a list)
    """
    return isinstance(obj, list) and len(obj) == 0





def plot_3d(mat, params_fig = {}, fig = [], ax = [], params_plot = {}, type_plot = 'plot', to_return = False):
    """
    Plot 3D data.

    Parameters:
    - mat (numpy.ndarray): 3D data to be plotted.
    - params_fig (dict): Additional parameters for creating the figure.
    - fig (matplotlib.figure.Figure): Existing figure to use (optional).
    - ax (numpy.ndarray): Existing 3D subplot axes to use (optional).
    - params_plot (dict): Additional parameters for the plot.
    - type_plot (str): Type of 3D plot ('plot' for line plot, 'scatter' for scatter plot).

    Returns:
    - fig (matplotlib.figure.Figure): The created or existing figure.
    - ax (numpy.ndarray): The created or existing 3D subplot axes.
    """ 
    if checkEmptyList(ax):
        fig, ax = create_3d_ax(1,1, params_fig)
    if type_plot == 'plot':    
        scatter = ax.plot(mat[0], mat[1], mat[2], **params_plot)
    else:
        scatter = ax.scatter(mat[0], mat[1], mat[2], **params_plot)
    if to_return:
        return scatter
    

def gaussian_array(length,sigma = 1  ):
    """
    Generate an array of Gaussian values with a given length and standard deviation.
    
    Args:
        length (int): The length of the array.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Default is 1.
    
    Returns:
        ndarray: The array of Gaussian values.
    """
    x = np.linspace(-3, 3, length)  # Adjust the range if needed
    gaussian = np.exp(-(x ** 2) / (2 * sigma ** 2))
    normalized_gaussian = gaussian / np.max(gaussian) # /sum()
    return normalized_gaussian

def gaussian_convolve(mat, wind = 10, direction = 1, sigma = 1, norm_sum = True, plot_gaussian = False, return_gaussian = False):
    """
    Convolve a 2D matrix with a Gaussian kernel along the specified direction.

    Parameters:
        mat (numpy.ndarray): The 2D input matrix to be convolved with the Gaussian kernel.
        wind (int, optional): The half-size of the Gaussian kernel window. Default is 10.
        direction (int, optional): The direction of convolution. 
            1 for horizontal (along columns), 0 for vertical (along rows). Default is 1.
        sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.

    Returns:
        numpy.ndarray: The convolved 2D matrix with the same shape as the input 'mat'.

    Raises:
        ValueError: If 'direction' is not 0 or 1.
    """
    if direction == 1:
        gaussian = gaussian_array(2*wind,sigma)
        if norm_sum:
            gaussian = gaussian / np.sum(gaussian)
        if plot_gaussian:
            plt.figure(); plt.plot(gaussian)
        mat_shape = mat.shape[1]
        T_or = mat.shape[1]
        mat = pad_mat(mat, np.nan, wind)
        mat_smooth = np.vstack( [[ np.nansum(mat[row, t:t+2*wind]*gaussian)                    
                     for t in range(T_or)] 
                   for row in range(mat.shape[0])])
        if return_gaussian:
            return mat_smooth, gaussian
            
        return mat_smooth
    elif direction == 0:
        return gaussian_convolve(mat.T, wind, direction = 1, sigma = sigma).T
    else:
        raise ValueError('invalid direction')    
        
def pad_mat(mat, pad_val, size_each = 1, axis = 1):
    if axis == 1:
        each_pad = np.ones((mat.shape[0], size_each))*pad_val
        mat = np.hstack([each_pad, mat, each_pad])
    else:
        each_pad = np.ones((size_each, mat.shape[1]))*pad_val
        mat = np.vstack([each_pad, mat, each_pad])        
    return mat
        
        
                     
def create_3d_ax(num_rows, num_cols, params = {}):
    """
    Create a 3D subplot grid.
    
    Parameters:
    - num_rows (int): Number of rows in the subplot grid.
    - num_cols (int): Number of columns in the subplot grid.
    - params (dict): Additional parameters to pass to plt.subplots.
    
    Returns:
    - fig (matplotlib.figure.Figure): The created figure.
    - ax (numpy.ndarray): The created 3D subplot axes.
    """
    fig, ax = plt.subplots(num_rows, num_cols, subplot_kw = {'projection': '3d'}, **params)
    return  fig, ax

def plot_scatter_by_dominant_subdynamics(x, coefficients, dims=[], trial_idx=0, 
                                          to_save=False, path_save='', fig_title='scatter_dominant_subdyn',
                                          cmap = 'tab10',
                                          scatter_kwargs={}):
    """
    Scatter plot of trajectory colored by dominant subdynamics.
    """
    # === Input validation ===
    assert isinstance(x, np.ndarray) and x.ndim == 2, "x must be 2D np.ndarray, got %s" % type(x).__name__
    p, T = x.shape
    assert p >= 2, "x must have at least 2 dimensions, got %d" % p
    assert isinstance(to_save, bool), "to_save must be bool, got %s" % type(to_save).__name__
    assert isinstance(path_save, str), "path_save must be str, got %s" % type(path_save).__name__
    assert isinstance(scatter_kwargs, dict), "scatter_kwargs must be dict, got %s" % type(scatter_kwargs).__name__
    
    # === Get coefficients ===
    if isinstance(coefficients, list):
        assert trial_idx < len(coefficients), "trial_idx=%d but only %d trials" % (trial_idx, len(coefficients))
        c = coefficients[trial_idx]
    else:
        c = coefficients
    
    assert isinstance(c, np.ndarray) and c.ndim == 2, "coefficients must be 2D np.ndarray"
    M, T_c = c.shape
    assert T_c == T - 1, "coefficients has %d columns but x has %d columns (expected T-1=%d)" % (T_c, T, T - 1)
    
    # === Scatter kwargs ===
    default_scatter_kwargs = {'s': 200, 'alpha': 0.7, 'cmap': 'viridis', 'edgecolor': 'none', 'linewidth': 0}
    scatter_kwargs = {**default_scatter_kwargs, **scatter_kwargs}
    
    # === Dominant subdynamics ===
    dominant_idx = np.argmax(np.abs(c), axis=0)
    assert dominant_idx.shape[0] == T - 1, "dominant_idx length mismatch"
    
    # === Setup dims ===
    if len(dims) == 0:
        dims = [0, 1, 2] if p >= 3 else [0, 1]
    assert len(dims) in [2, 3], "dims must have 2 or 3 elements, got %d" % len(dims)
    assert all(d < p for d in dims), "dims contain index >= p (%d)" % p
    
    # === Colormap ===
    if M > 10 and cmap == 'tab10':
        if M <= 20:
            cmap = 'tab20'
        else:
            print('changing cmaps due to too many dynamics')
            cmap = 'hsv'
    if cmap == 'tab10':
        num_parts = 10
    elif cmap == 'tab20':
        num_parts = 20
    else:
        num_parts = M
    color_values_unique = create_colors(num_parts, style = 'cmap', cmap = cmap)[:M]
    color_values = [color_values_unique[idx] for idx in dominant_idx]
    
    # === Plot ===
    if len(dims) == 3:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[dims[0], :-1], x[dims[1], :-1], x[dims[2], :-1], c=color_values, **scatter_kwargs)
        ax.set_xlabel('$x_{%d}$' % (dims[0] + 1))
        ax.set_ylabel('$x_{%d}$' % (dims[1] + 1))
        ax.set_zlabel('$x_{%d}$' % (dims[2] + 1))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(x[dims[0], :-1], x[dims[1], :-1], c=color_values, **scatter_kwargs)
        ax.set_xlabel('$x_{%d}$' % (dims[0] + 1))
        ax.set_ylabel('$x_{%d}$' % (dims[1] + 1))
    
    ax.set_title('Trajectory colored by dominant $c_m$')
    
    # === Legend ===
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_values_unique[m], markersize=10, 
                          label='$c_{%d}$' % (m + 1)) for m in range(M)]
    ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    if to_save:
        save_fig(fig_title, fig, path_save if path_save else '.')
    
    return fig, ax
def create_colors(len_colors, perm = [0,1,2], style = 'random', cmap  = 'viridis', shuffle_colors = False, shuffle_seed = 0):
    """
    Create a set of discrete colors with a one-directional order
    Input: 
        len_colors = number of different colors needed
    Output:
        3 X len_colors matrix decpiting the colors in the cols
    """
    if style == 'random':
        colors = np.vstack([np.linspace(0,1,len_colors),(1-np.linspace(0,1,len_colors))**2,1-np.linspace(0,1,len_colors)])
        colors = colors[perm, :]
        assert not shuffle_colors, "TB done"
    else:
        
        # Define the colormap you want to use
        #cmap = plt.get_cmap()  # Replace 'viridis' with the desired colormap name

        cmap = plt.get_cmap(cmap) 
        # Create an array of values ranging from 0 to 1 to represent positions in the colormap
        positions = np.linspace(0, 1, len_colors)
        
        # Generate the list of colors by applying the colormap to the positions
        colors = [cmap(pos) for pos in positions]
        
        # You can now use the 'colors' list as a list of colors in your application
        if shuffle_colors:
            random.seed(shuffle_seed)
            random.shuffle(colors)
            
    return colors

def plot_f(F_list, path_save=None, fig_title='dynamics_operators'):
    """
    Plot all F matrices as heatmaps.
    """
    M = len(F_list)
    
    fig, axes = plt.subplots(1, M, figsize=(4 * M, 4))
    
    if M == 1:
        axes = [axes]
    
    for i, f in enumerate(F_list):
        sns.heatmap(f, ax=axes[i], cmap='coolwarm', center=0, lw = 1, edgecolor = 'black',
                    linecolor = 'black',
                    square=True, cbar_kws={'shrink': 0.8})
        axes[i].set_title('$F_{%d}$' % (i + 1))
        axes[i].set_xlabel('Dimension')
        axes[i].set_ylabel('Dimension')
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, axes


def plot_single_F(F, idx=0, path_save=None, fig_title='single_F'):
    """
    Plot a single F matrix as heatmap.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    sns.heatmap(F, ax=ax, cmap='coolwarm', center=0, 
                square=True, annot=F.shape[0] <= 6)
    ax.set_title('$F_{%d}$' % (idx + 1))
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Dimension')
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, ax


def plot_coefficients(coefficients, trial_idx=0, path_save=None, fig_title='coefficients'):
    """
    Plot dynamics coefficients over time.
    """
    if isinstance(coefficients, list):
        c = coefficients[trial_idx]
    else:
        c = coefficients
    
    M, T = c.shape
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    for m in range(M):
        ax.plot(c[m, :], label='$c_{%d}$' % (m + 1), linewidth=1.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Coefficient value')
    ax.set_title('Dynamics Coefficients $c_m(t)$')
    ax.legend(loc='upper right')
    ax.set_xlim([0, T - 1])
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, ax


def plot_coefficients_heatmap(coefficients, trial_idx=0,
                              path_save=None, 
                              fig_title='coefficients_heatmap', cmap = 'PiYG'):
    """
    Plot dynamics coefficients as heatmap.
    """
    if isinstance(coefficients, list):
        c = coefficients[trial_idx]
    else:
        c = coefficients
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    
    sns.heatmap(c, ax=ax, cmap=cmap, cbar_kws={'shrink': 0.8}, center = 0 )
    ax.set_xlabel('Time')
    ax.set_ylabel('$c_m$')
    ax.set_title('Dynamics Coefficients')
    yticks = ['$c_{%d}$' % (m + 1) for m in range(c.shape[0])]
    ax.set_yticklabels(yticks, rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, ax


def plot_reconstruction(x_original, x_reconstructed, dims=None, path_save=None, fig_title='reconstruction',
                        fig = [], axs = []):
    """
    Plot original vs reconstructed trajectory.
    """
    p, T = x_original.shape
    
    if dims is None:
        dims = list(range(min(3, p)))
    
    n_dims = len(dims)
    
    fig, axs = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), sharex=True)
    
    if n_dims == 1:
        axs = [axs]
    
    for i, d in enumerate(dims):
        axs[i].plot(x_original[d, :], label='Original', linewidth=1.5)
        axs[i].plot(x_reconstructed[d, :], '--', label='Reconstructed', linewidth=1.5)
        axs[i].set_ylabel('$x_{%d}$' % (d + 1))
        axs[i].legend(loc='upper right')
    
    axs[-1].set_xlabel('Time')
    axs[0].set_title('Original vs Reconstructed')
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, axs


def plot_trajectory(x, dims=None, path_save=None, fig_title='trajectory'):
    """
    Plot trajectory over time.
    """
    p, T = x.shape
    
    if dims is None:
        dims = list(range(min(3, p)))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    for d in dims:
        ax.plot(x[d, :], label='$x_{%d}$' % (d + 1), linewidth=1.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Trajectory')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, ax


def plot_trajectory_3d(x, dims=None, path_save=None, fig_title='trajectory_3d'):
    """
    Plot 3D trajectory.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    if dims is None:
        dims = [0, 1, 2]
    
    assert len(dims) == 3, "dims must have exactly 3 elements for 3D plot"
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x[dims[0], :], x[dims[1], :], x[dims[2], :], linewidth=1.5)
    ax.scatter(x[dims[0], 0], x[dims[1], 0], x[dims[2], 0], c='g', s=50, label='Start')
    ax.scatter(x[dims[0], -1], x[dims[1], -1], x[dims[2], -1], c='r', s=50, label='End')
    
    ax.set_xlabel('$x_{%d}$' % (dims[0] + 1))
    ax.set_ylabel('$x_{%d}$' % (dims[1] + 1))
    ax.set_zlabel('$x_{%d}$' % (dims[2] + 1))
    ax.set_title('3D Trajectory')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, ax


def plot_error_history(error_history, path_save=None, fig_title='error_history', log_scale=True):
    """
    Plot training error over iterations.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    ax.plot(error_history, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.set_title('Training Error History')
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, ax


def plot_combined_F(F_combined, time_idx=0, path_save=None, fig_title='combined_F'):
    """
    Plot combined dynamics matrix F_t = sum_m c_mt * f_m at a specific time.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    F_t = F_combined[:, :, time_idx]
    
    sns.heatmap(F_t, ax=ax, cmap='coolwarm', center=0, square=True)
    ax.set_title('$F_t$ at $t=%d$' % time_idx)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Dimension')
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, ax

import os

def save_fig(name_fig, fig, save_path = '', formats = ['png','svg'], save_params = {}, verbose = True) :
    if len(save_path) == 0:
        save_path = os.getcwd()
    if 'transparent' not in save_params:
        save_params['transparent'] = True
    [fig.savefig(save_path + os.sep + '%s.%s'%(name_fig, format_i), **save_params) for format_i in formats]
    if verbose:
        print('saved figure: %s'%(save_path + os.sep + '%s.%s'%(name_fig, 'png')))

def plot_f_evolution(F_evolution, operator_idx=0, path_save=None, fig_title='F_evolution'):
    """
    Plot how a single F matrix evolves during training (as Frobenius norm).
    """
    norms = [np.linalg.norm(F_iter[operator_idx], 'fro') for F_iter in F_evolution]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    ax.plot(norms, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Frobenius norm')
    ax.set_title('$||F_{%d}||_F$ Evolution' % (operator_idx + 1))
    
    plt.tight_layout()
    plt.show()
    
    if path_save is not None:
        save_fig(fig_title, fig, path_save)
    
    return fig, ax


