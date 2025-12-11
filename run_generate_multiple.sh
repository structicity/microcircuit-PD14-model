#!/bin/bash

for seed in 123450 123451 123452 123453 123454 123455 123456 123457 123458 123459
do
    echo "Running seed $seed"
    python PyNEST/reference_data/generate_reference_data.py --seed="$seed" --path="data/data_T10s_epropiafpscdelta/seed-$seed"
done
