#!/bin/bash
PROJECT_ID=adcoar-452409  # Replace with your project ID
REGION=us-central1 # Replace with your desired region
ZONES=("$REGION-a" "$REGION-a" "$REGION-b" "$REGION-b" "$REGION-c" "$REGION-c" "$REGION-f" "$REGION-f")

for i in {1..8}; do
  echo "Setting up light-cluster-$i in ${ZONES[$i-1]}"
  gcloud compute ssh light-cluster-$i --zone="${ZONES[$i-1]}" --project="$PROJECT_ID" --command="
    sudo yum update -y &&
    sudo yum install -y epel-release gcc gcc-c++ make centos-release-scl &&
    sudo yum install -y openmpi openmpi-devel opencv opencv-devel zlib-devel &&
    echo 'module load mpi/openmpi-x86_64' >> ~/.bashrc
  "
  if [ $? -eq 0 ]; then
    echo "Successfully set up light-cluster-$i"
  else
    echo "Failed to set up light-cluster-$i"
  fi
done