for DCO in BHBH BHNS NSNS BHWD NSWD; do
    for VARIANT in fiducial alpha_low maltsev alpha_high qcflag2 no_ecsn; do
        echo "Checking ${DCO} ${VARIANT}"
        for i in $(seq 0 255); do
            [[ -f "/mnt/home/twagg/ceph/lisa-dcos/${VARIANT}/detection_files/${DCO}_pessimistic_f_detect_${i}.h5" ]] || echo $i
        done
    done
done