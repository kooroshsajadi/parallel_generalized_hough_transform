#!/bin/bash
PROJECT_ID=adcoar-452409  # Replace with your project ID
REGION=us-central1 # Replace with your desired region
ZONES=("$REGION-a" "$REGION-a" "$REGION-b" "$REGION-b" "$REGION-c" "$REGION-c" "$REGION-f" "$REGION-f")

for i in {1..8}; do
  gcloud compute instances create light-cluster-$i \
    --project="$PROJECT_ID" \
    --zone="${ZONES[$i-1]}" \
    --machine-type=custom-2-4096 \
    --image-family=centos-stream-9 \
    --image-project=centos-cloud \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-standard \
    --tags=light-cluster
  if [ $? -eq 0 ]; then
    echo "Successfully created light-cluster-$i in ${ZONES[$i-1]}"
  else
    echo "Failed to create light-cluster-$i in ${ZONES[$i-1]}"
  fi
done