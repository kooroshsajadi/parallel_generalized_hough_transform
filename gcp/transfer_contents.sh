#!/bin/bash
for vm in 10.128.0.{6..12}; do
  scp GHTParallelBatch $vm:~/
  scp -r ~/resources $vm:~/
  scp hostfile.txt $vm:~/
done
mpiexec -n 16 --hostfile hostfile.txt ./GHTParallelBatch