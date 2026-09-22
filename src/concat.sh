#!/bin/bash

MAX_JOBS=3
BASE="/mnt/ceph/users/twagg/lisa-dcos/fiducial"

for dco_type in BHBH BHNS NSNS BHWD NSWD BHBH_pessimistic BHNS_pessimistic NSNS_pessimistic BHWD_pessimistic NSWD_pessimistic; do
    # wait until a slot frees up
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n
    done

    echo "Running concat_detectable_pops.py for $dco_type"
    python -u concat_detectable_pops.py \
        -f "${BASE}/detection_files" \
        -d "$dco_type" \
        -o "${BASE}/" \
        -O > "logs/concat_${dco_type}.log" 2>&1 &
done

# wait for the final batch
wait
echo "All done with fiducial"

for variation in alpha_low alpha_high maltsev no_ecsn qcflag2; do
    for dco_type in BHBH BHNS NSNS BHWD NSWD; do
        # wait until a slot frees up
        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
            wait -n
        done

        echo "Running concat_detectable_pops.py for $dco_type $variation"
        python -u concat_detectable_pops.py \
            -f "/mnt/ceph/users/twagg/lisa-dcos/${variation}/detection_files" \
            -d "${dco_type}_pessimistic" \
            -o "/mnt/ceph/users/twagg/lisa-dcos/${variation}/" \
            -t \
            -O > "logs/concat_${dco_type}_${variation}.log" 2>&1 &
    done
done

# wait for the final batch
wait
echo "All done with variations"