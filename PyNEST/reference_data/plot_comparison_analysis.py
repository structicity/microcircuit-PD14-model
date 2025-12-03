from microcircuit import helpers

from plot_reference_analysis import compute_data_dist

from params import params as ref_dict

import matplotlib.pyplot as plt
import matplotlib

import numpy as np


from microcircuit.network_params import default_net_dict as net_dict
populations = net_dict['populations'] # list of populations
seeds = ref_dict['RNG_seeds'] # list of seeds

def plot_data_dists( 
        observable_name: str, 
        x_label: str, 
        observable_hist_mat: dict, 
        observable_best_bins: np.ndarray, 
        observable_ks_distances: dict, 
        observable_limits: tuple[float,float]=None,
        fig_hist: matplotlib.figure=None,
        axes_hist: matplotlib.axes=None,
        fig_ks: matplotlib.figure=None,
        axes_ks:  matplotlib.axes=None,
        ) -> None:
    '''
    Plots histograms and KS-distance distributions for different populations.
    -------------------------------------------------------------------------
    Parameters:
    - observable_name : str
        The name of the observable being plotted (used for figure file name).
    - x_label : str
        The label for the x-axis of the histogram plots.
    - observable_hist_mat : dict
        A dictionary containing the histograms for each population, where keys are population indices.
    - observable_best_bins : ndarray
        An array containing the best bin edges for each population.
    - observable_ks_distances : dict
        A dictionary containing KS-distance values for each population, where keys are population names.
    - observable_limits : tuple[float,float], optional
        The limits for the x-axis of the histogram plots. If None, limits are determined from the data.
    -------------------------------------------------------------------------
    Returns:
    - None
    '''

    from matplotlib import rcParams
    rcParams['figure.figsize']    = (ref_dict['max_fig_width'] / 3., (4. / 9.) * ref_dict['max_fig_width'])
    rcParams['figure.dpi']        = 300
    rcParams['font.family']       = 'sans-serif'
    rcParams['font.size']         = 8
    rcParams['legend.fontsize']   = 8
    rcParams['axes.titlesize']    = 10
    rcParams['axes.labelsize']    = 8
    rcParams['ytick.labelsize']   = 8
    rcParams['xtick.labelsize']   = 8
    rcParams['ytick.major.size']  = 0   ## remove y ticks  seeds = ref_dict['RNG_seeds'] # list of seeds    
    rcParams['text.usetex']       = True 
    rcParams['legend.framealpha'] = 1.0
    rcParams['legend.edgecolor']  = 'k'

    x_max_hist = 0
    x_min_hist = 0
    for cpop, pop in enumerate( populations ):
        bin_edges = observable_best_bins[cpop][3]
        x_max_hist = max( x_max_hist, np.max( bin_edges ) )
        x_min_hist = min( x_min_hist, np.min( bin_edges ) )

    ks_max = 0
    for cpop, pop in enumerate( populations ):
        ks_values = observable_ks_distances[pop]["list"]
        ks_max = max( ks_max, np.max( ks_values ) )

    for cpop, pop in enumerate( populations ):
        ax_hist = axes_hist[cpop // 2, cpop % 2]     # axes for each population
        ax_ks = axes_ks[cpop // 2, cpop % 2]

        bin_edges = observable_best_bins[cpop][3]
        bin_centers = bin_edges[:-1]

        pop_rel_hists = observable_hist_mat[cpop]

        # calculate population mean and std histograms across seeds
        pop_mean_hist = np.mean( pop_rel_hists, axis=0 )
        pop_std_hist  = np.std( pop_rel_hists, axis=0 )

        n_seeds = len( seeds )
        grayscale = np.linspace( 0.2, 0.8, n_seeds )
        colors = [(g, g, g) for g in grayscale]

        for cseed in range( n_seeds ):
            ax_hist.plot( bin_centers, pop_rel_hists[cseed], '-', color=colors[-1], label=f'Seed {cseed}' )

        ax_hist.plot( bin_centers, pop_mean_hist, 'k--', label='Mean' )
        ax_hist.fill_between( bin_centers, pop_mean_hist - pop_std_hist, pop_mean_hist + pop_std_hist, alpha=0.3 )

        # set x and y limits
        if observable_limits is not None:
            ax_hist.set_xlim( observable_limits[0], observable_limits[1] )
            ax_hist.set_xticks( [observable_limits[0], (observable_limits[0] + observable_limits[1]) / 2],
                              [r'$%.1f$' % observable_limits[0], r'$%.1f$' % ((observable_limits[0] + observable_limits[1]) / 2)] )
            if observable_name == 'spike_ccs':
                ax_hist.set_xticks( [observable_limits[0]/2, 0, observable_limits[1]/2],
                                  [r'$%.2f$' % (observable_limits[0]/2), r'$0$', r'$%.2f$' % (observable_limits[1]/2)] )
            
        else:
            ax_hist.set_xlim( 0, x_max_hist )
            ax_hist.set_xticks([0, x_max_hist/2], [r'$0$', r'$%.0f$' % (x_max_hist/2)] )
            if observable_name == 'spike_ccs':
                ax_hist.set_xlim(x_min_hist, x_max_hist)
                ax_hist.set_xticks([x_min_hist/2, 0, x_max_hist/2], [r'$%.2f$' % (x_min_hist/2), r'$0$', r'$%.2f$' % (x_max_hist/2)] )

        ax_hist.set_ylim( 0, np.max( pop_mean_hist ) * 1.2 )
        
        ks_values = observable_ks_distances[pop]["list"]
        mean = std = None
        if len( ks_values ) > 0:
            mean = np.mean( ks_values )
            std = np.std( ks_values )

        textbox = r'%s' % pop
        if mean is not None:
            textbox += r'\\{\tiny $D_\mathsf{KS} = %.2f$}' % mean
        
            ax_ks.hist( ks_values, bins=n_seeds, color='gray', alpha=0.5 )
            ax_ks.axvline( mean, color='red', linestyle='--', label='Mean KS-distance' )
            ax_ks.axvline( mean + std, color='blue', linestyle='--', label='Mean + Std' )
            ax_ks.axvline( mean - std, color='blue', linestyle='--', label='Mean - Std' )

        ax_hist.text( 0.95, 0.95, textbox, transform=ax_hist.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right' )
        ax_ks.text( 0.95, 0.95, r'%s' % pop, transform=ax_ks.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right' )
            
        if cpop % 2 == 0:
            ax_hist.set_ylabel( r'rel. freq.' )
            ax_hist.set_yticks( [] )
            ax_ks.set_ylabel( r'rel. freq.' )
            ax_ks.set_yticks( [] )
        else:
            ax_hist.set_yticks( [] )
    
    ax_ks.set_xlim( 0, ks_max * 1.1)
    ax_ks.set_xticks( [0, ks_max / 2], [r'$0$', r'$%.2f$' % (ks_max / 2)] )

    fig_hist.text(0.5, -0.01, x_label, ha="center", va="center")
    #fig_hist.savefig(f'{data_path}{observable_name}_distributions.pdf',
    #             bbox_inches="tight", pad_inches=0.02)
    fig_ks.text(0.5, 0.0, r'KS-distance', ha="center", va="center")
    """fig_ks.savefig(f'{data_path}{observable_name}_KS_distances.pdf',
               bbox_inches="tight", pad_inches=0.02)
    
    # save figures for README.md
    fig_hist.savefig(f'figures/{observable_name}_distributions_T' + str( int( ref_dict['t_sim'] * 1.0e-3 ) ) + 's.png',
                 bbox_inches="tight", pad_inches=0.02)
    fig_ks.savefig(f'figures/{observable_name}_KS_distances_T' + str( int( ref_dict['t_sim'] * 1.0e-3 ) ) + 's.png',
               bbox_inches="tight", pad_inches=0.02)"""

def create_fig():
    # plot of histograms for all populations
    fig_hist, axes_hist = plt.subplots( 4, 2, sharex=True, gridspec_kw={'hspace': 0.0, 'wspace': 0.0} )
    fig_ks, axes_ks = plt.subplots( 4, 2, sharex=True, gridspec_kw={'hspace': 0.0, 'wspace': 0.0} )
    return {
        "fig_hist": fig_hist,
        "axes_hist": axes_hist,
        "fig_ks": fig_ks,
        "axes_ks": axes_ks
    }

def main(data_paths, ref_dicts):
    # rates
    rates_fig_dict = create_fig()
    # spike_cvs
    spkcsv_fig_dict = create_fig()
    # spike_ccs
    spkccs_fig_dict = create_fig()
   
    for data_path, ref_dict in zip(data_paths, ref_dicts):
        # Read in the data from json files
        rates = helpers.json2dict( f'{data_path}rates.json' )
        spike_cvs = helpers.json2dict( f'{data_path}spike_cvs.json' )
        spike_ccs = helpers.json2dict( f'{data_path}spike_ccs.json' )

        rate_ks_distances = helpers.json2dict( f'{data_path}rate_ks_distances.json' )
        spike_cvs_ks_distances = helpers.json2dict( f'{data_path}spike_cvs_ks_distances.json' )
        spike_ccs_ks_distances = helpers.json2dict( f'{data_path}spike_ccs_ks_distances.json' )

        # Compute distributions and statistics
        rate_hist_mat, rate_best_bins, rate_stats = compute_data_dist( observable=rates, observable_name='rate', observable_limits=ref_dict['rate_lim'], units='1/s', bin_size=ref_dict['rate_binsize'] )
        spike_cvs_hist_mat, spike_cvs_best_bins, spike_cvs_stats = compute_data_dist( observable=spike_cvs, observable_name='spike_cvs', observable_limits=ref_dict['cv_lim'], bin_size=ref_dict['cv_binsize'] )
        spike_ccs_hist_mat, spike_ccs_best_bins, spike_ccs_stats = compute_data_dist( observable=spike_ccs, observable_name='spike_ccs', observable_limits=ref_dict['cc_lim'], bin_size=ref_dict['cc_binsize'] )

        # Plot distributions and KS distances
        plot_data_dists( 'rate', r'\begin{center} time averaged single neuron\\firing rate (s$^{-1}$) \end{center}', rate_hist_mat, rate_best_bins, rate_ks_distances, observable_limits=ref_dict['rate_lim'],
            **rates_fig_dict)
        plot_data_dists( 'spike_cvs', r'spike irregularity (ISI CV)', spike_cvs_hist_mat, spike_cvs_best_bins, spike_cvs_ks_distances, observable_limits=ref_dict['cv_lim'],
            **spkcsv_fig_dict )
        plot_data_dists( 'spike_ccs', r'\begin{center} spike correlation coefficient\\(bin size $%.1f$ ms) \end{center}' % ref_dict['binsize'], spike_ccs_hist_mat, spike_ccs_best_bins, spike_ccs_ks_distances, observable_limits=ref_dict['cc_lim'],
            **spkccs_fig_dict)
    plt.show()

    ## current memory consumption of the python process (in MB)
    import psutil
    mem = psutil.Process().memory_info().rss / ( 1024 * 1024 )
    print( f"Current memory consumption: {mem:.2f} MB" )

if __name__ == "__main__":
    data_paths = ["./data/data_T10s/", "./data/data_T10s/"]
    ref_dicts = [ref_dict, ref_dict]
    main(data_paths=data_paths, ref_dicts=ref_dicts)
    plt.show()