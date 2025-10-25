export JAX_NUM_CPU_DEVICES=2
num_processes=4

range=$(seq 0 $(($num_processes - 1)))

for i in $range; do
  uv run main.py $i $num_processes > /tmp/main_$i.out &
done

wait

for i in $range; do
  echo "=================== process $i output ==================="
  cat /tmp/main_$i.out
  echo
done