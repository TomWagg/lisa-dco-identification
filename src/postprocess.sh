#!/bin/bash

MAX_JOBS=3
BASE="/mnt/ceph/users/twagg/lisa-dcos/"

for model in fiducial alpha_low alpha_high no_ecsn maltsev qcflag2; do
    for dco_type in BHBH BHNS NSNS BHWD NSWD; do
        # wait until a slot frees up
        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
            wait -n
        done

        echo "Running postprocess_dcos.py for $model $dco_type"
        python -u postprocess_dcos.py \
            -f "${BASE}/${model}" \
            -d "$dco_type" > "logs/postprocess_${model}_${dco_type}.log" 2>&1 &
    done
done

# wait for the final batch
wait
echo "All done"