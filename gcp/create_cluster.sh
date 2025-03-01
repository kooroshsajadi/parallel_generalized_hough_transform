#!/bin/bash
PROJECT_ID=my-light-cluster  # Replace with your PROJECT_ID
REGION=us-central1 # Replace with your preferred REGION
ZONES=("$REGION-a" "$REGION-a" "$REGION-b" "$REGION-b" "$REGION-c" "$REGION-c" "$REGION-f" "$REGION-f")

for i in {1..8}; do
  gcloud compute instances create light-cluster-$i \
    --project=$PROJECT_ID \
    --zone=${ZONES[$i-1]} \
    --machine-type=custom-2-4096 \
    --image-family=centos-7 \
    --image-project=centos-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard \
    --tags=light-cluster \
    --no-address  # No external IP, internal cluster (optional)
  echo "Created light-cluster-$i in ${ZONES[$i-1]}"
done