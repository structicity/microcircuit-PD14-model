# -*- coding: utf-8 -*-
#
# generate_reference_data.py
#
# This file is part of https://github.com/INM-6/microcircuit-PD14-model
#
# SPDX-License-Identifier: GPL-2.0-or-later

'''
Generation of reference data for the microcircuit model.
'''

#####################
import time
import nest
import numpy as np

## import model implementation
from microcircuit import network

## import (default) parameters (network, simulation, stimulus)
from microcircuit.network_params import default_net_dict as net_dict
from microcircuit.sim_params import default_sim_dict as sim_dict
from microcircuit.stimulus_params import default_stim_dict as stim_dict

## import analysis parameters
from params import params as ref_dict

from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--path", type=str, default="data")
args = parser.parse_args()

path = Path(args.path)
sim_dict.update(
        {
            "data_path": str(path) + "/",
            "rng_seed": args.seed,
        }
)

#####################
scaling_factor = ref_dict['scaling_factor']
duration_seq = 300
model_eprop = "eprop_iaf_psc_delta"

eprop_params = {
    "beta": 1.7,  # width scaling of the pseudo-derivative
    "C_m": 1.0,
    "c_reg": 2.0 / duration_seq,  # coefficient of firing rate regularization
    "E_L": 0.0,
    "eprop_isi_trace_cutoff": 100,
    "f_target": 10.0,  # spikes/s, target firing rate for firing rate regularization
    "gamma": 0.5,  # height scaling of the pseudo-derivative
    "I_e": 0.0,
    "kappa": 0.99,  # low-pass filter of the eligibility trace
    "kappa_reg": 0.99,  # low-pass filter of the firing rate for regularization
    "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
    "t_ref": 0.0,  # ms, duration of refractory period
    "tau_m": 30.0,
    "tau_syn": 0.5, #! Added from microcircuit and used for calculate PSC
    "V_m": 0.0,
    "V_th": 0.6,  # mV, spike threshold membrane voltage
    "V0_mean": {"original": 0.0, "optimized": [-68.28, -63.16, -63.33, -63.45, -63.11, -61.66, -66.72, -61.43]},
    # standard deviation of the average membrane potential (in mV)
    "V0_std": {"original": 0.8, "optimized": [5.36, 4.57, 4.74, 4.94, 4.94, 4.55, 5.46, 4.48]},
}

scale_factor = 1.0 - eprop_params["kappa"]  # factor for rescaling due to removal of irregular spike arrival
eprop_params["c_reg"] /= scale_factor**2

if model_eprop == "eprop_iaf_adapt":
    eprop_params["adapt_beta"] = 0.0  # adaptation scaling

if model_eprop in ["eprop_iaf_psc_delta", "eprop_iaf_psc_delta_adapt"]:
    eprop_params["V_reset"] = -0.5  # mV, reset membrane voltage
    eprop_params["c_reg"] = 2.0 / duration_seq / scale_factor**2
    eprop_params["V_th"] = 0.5

## set network scale
net_dict["N_scaling"] = scaling_factor
net_dict["K_scaling"] = scaling_factor
net_dict["neuron_model"] = model_eprop
net_dict["neuron_params"] = eprop_params
net_dict["V0_type"] = "original"
net_dict["PSP_exc_mean"] = 10
net_dict["weight_rel_std"] = 10

## set pre-simulation time to 0 and desired simulation time
sim_dict["t_presim"] = ref_dict["t_presim"]
sim_dict["t_sim"] = ref_dict["t_sim"] # simulate for 10.0s

## set number of local number of threads
sim_dict["local_num_threads"] = ref_dict['local_num_threads']

def main():

    ## start timer 
    time_start = time.time()

    ## create instance of the network
    net = network.Network(sim_dict, net_dict, stim_dict)
    time_network = time.time()

    ## create all nodes (neurons, devices)
    net.create()
    time_create = time.time()

    ## connect nework
    net.connect()
    time_connect = time.time()

    ## pre-simulation (warm-up phase)
    net.simulate(sim_dict["t_presim"])
    time_presimulate = time.time()

    ## simulation
    net.simulate(sim_dict["t_sim"])
    time_simulate = time.time()

    ## current memory consumption of the python process (in MB)
    import psutil
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    #TODO store benchmark data in store_metadata
    #net.benchmark_data = {}
    #net.benchmark_data['memory'] = mem

    #####################
    ## plot spikes and firing rate distribution
    print()
    print('##########################################')
    print()
    observation_interval = np.array([sim_dict["t_presim"], sim_dict["t_presim"] + sim_dict["t_sim"]])
    net.evaluate(observation_interval , observation_interval )
    print()
    print('Raster plot                  : see %s ' % (sim_dict['data_path'] + 'raster_plot.png') )
    print('Distributions of firing rates: see %s ' % (sim_dict['data_path'] + 'box_plot.png'   ) )
    time_evaluate = time.time()

    #####################
    ## print timers and memory consumption

    print()
    print('##########################################')
    print()
    print('Times of Rank %d:' % nest.Rank())
    print('    Total time:')
    print('    Time to initialize  : %.3fs' % (time_network - time_start))
    print('    Time to create      : %.3fs' % (time_create - time_network))
    print('    Time to connect     : %.3fs' % (time_connect - time_create))
    print('    Time to presimulate : %.3fs' % (time_presimulate - time_connect))
    print('    Time to simulate    : %.3fs' % (time_simulate - time_presimulate))
    print('    Time to evaluate    : %.3fs' % (time_evaluate - time_simulate))
    print()
    print("Memory consumption: %dMB" % mem)
    print()
    print('##########################################')
    print()

    net.store_metadata()
    
#####################

if __name__== '__main__':
    main()
