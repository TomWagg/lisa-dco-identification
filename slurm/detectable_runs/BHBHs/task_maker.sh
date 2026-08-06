for i in 0 1 2 3 4 5 6 7; do
    start=$((i * 32))
    sed "s/start 0/start ${start}/" _Tasks_template > Tasks_${i}
done