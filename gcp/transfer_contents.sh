#!/bin/bash
for vm in 10.128.0.{6..12}; do
  scp GHTParallelBatch $vm:~/
  scp -r ~/resources $vm:~/
done