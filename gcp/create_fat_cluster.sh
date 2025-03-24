#!/bin/bash
PROJECT_ID=adcoar-452409  # Replace with your project ID
REGION=us-central1 # Replace with your desired region
ZONES=("$REGION-a" "$REGION-a")

for i in {1..2}; do
  gcloud compute instances create fat-cluster-$i \
    --project="$PROJECT_ID" \
    --zone="${ZONES[$i-1]}" \
    --machine-type=custom-8-16384 \
    --image-family=centos-stream-9 \
    --image-project=centos-cloud \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-standard \
    --tags=fat-cluster
  if [ $? -eq 0 ]; then
    echo "Successfully created fat-cluster-$i in ${ZONES[$i-1]}"
  else
    echo "Failed to create fat-cluster-$i in ${ZONES[$i-1]}"
  fi
done