for i in {1..8}; do
  gcloud compute scp id_rsa.pub light-cluster-$i:~/Documents/infossh \
    --zone=us-central1-$(printf "%c" $(($i % 4 + 97))) \
    --project=adcoar-452409
done