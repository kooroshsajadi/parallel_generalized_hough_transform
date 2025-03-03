#!/bin/bash
VMS=("10.128.0.6" "10.128.0.7" "10.128.0.8" "10.128.0.9" "10.128.0.10" "10.128.0.11" "10.128.0.12")
for vm in "${VMS[@]}"; do
  echo "Setting up $vm"
  ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no $vm "
    echo 'seyedkourosh_sajjadi01 ALL=(ALL) NOPASSWD:ALL' | sudo tee -a /etc/sudoers.d/90-user &&
    echo 'module load mpi/openmpi-x86_64' >> ~/.bashrc &&
    source ~/.bashrc  # Activate immediately
  "
  if [ $? -eq 0 ]; then
    echo "Successfully set up $vm"
  else
    echo "Failed to set up $vm"
  fi
done