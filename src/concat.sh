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
        -O > "${dco_type}.log" 2>&1 &
done

# wait for the final batch
wait
echo "All done"