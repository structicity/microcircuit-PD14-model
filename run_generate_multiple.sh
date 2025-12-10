#!/bin/bash

for seed in 123410 123411 123412 123413 123414 123415 123416 123417 123418 123419
do
    echo "Running seed $seed"
    python PyNEST/reference_data/generate_reference_data.py --seed="$seed" --path="data/data_T10s/seed-$seed"
done
