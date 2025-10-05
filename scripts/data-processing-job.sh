kubectl apply -f k8s-jobs/data-processing-job.yaml
kubectl logs job/data-processing-job -f