#!/bin/bash

N_scaling=0.2
K_scaling=1.0
model_name=iaf_psc_exp
# Python reference_data scripts expect t_sim in ms.
t_sim_ms=10000
t_sim_s=$((t_sim_ms / 1000))

data_id_prefix=data_T${t_sim_s}s
data_id=${model_name}_downscaled${N_scaling}${K_scaling}
full_data_id="${data_id_prefix}_${data_id}"

for seed in 123450 123451 123452 123453 123454 123455 123456 123457 123458 123459
do
    echo "Running seed $seed"
    python PyNEST/reference_data/generate_reference_data.py --seed="$seed" --path="data/$full_data_id/seed-$seed" --model_name="$model_name" --N_scaling="$N_scaling" --K_scaling="$K_scaling" --t_sim="$t_sim_ms"
done

for seed in 123450 123451 123452 123453 123454 123455 123456 123457 123458 123459
do
    echo "Running seed $seed"
    python PyNEST/reference_data/analyze_reference_data.py --seed="$seed" --path="data/$full_data_id/seed-$seed" --N_scaling="$N_scaling" --K_scaling="$K_scaling" --t_sim="$t_sim_ms"
done

python PyNEST/reference_data/compute_ensemble_statistics.py --data_id="$data_id" --N_scaling="$N_scaling" --K_scaling="$K_scaling" --t_sim="$t_sim_ms"

python PyNEST/reference_data/plot_reference_analysis.py --data_id="$data_id" --N_scaling="$N_scaling" --K_scaling="$K_scaling" --t_sim="$t_sim_ms"
