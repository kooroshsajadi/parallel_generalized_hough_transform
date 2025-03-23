#!/bin/bash
# This bash can be executed in the master node for sending contents to slave nodes and then start the program.
for vm in 10.128.0.{6..12}; do
  scp GHTParallelBatch $vm:~/
  scp -r ~/resources $vm:~/
  scp hostfile.txt $vm:~/
done
mpiexec -n 16 --hostfile hostfile.txt ./GHTParallelBatch