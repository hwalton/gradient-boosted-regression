eval $(minikube docker-env)
docker build -f Dockerfile.data -t gbr-data:latest .
docker build -f Dockerfile.train -t gbr-train:latest .
kubectl delete job data-processing-job
kubectl delete job training-job