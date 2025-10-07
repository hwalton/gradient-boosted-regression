#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-default}
SERVICES=("gbr-airflow-webserver" "gbr-mlflow-service" "gbr-serving-service")
TIMEOUT=${TIMEOUT:-300}   # seconds to wait per service
SLEEP=${SLEEP:-5}

echo "Namespace: $NAMESPACE"
for svc in "${SERVICES[@]}"; do
  echo
  echo "Processing service: $svc"

  # ensure service exists
  if ! kubectl get svc "$svc" -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "  Service $svc not found in namespace $NAMESPACE — skipping."
    continue
  fi

  cur_type=$(kubectl get svc "$svc" -n "$NAMESPACE" -o jsonpath='{.spec.type}')
  echo "  Current type: $cur_type"

  if [ "$cur_type" != "LoadBalancer" ]; then
    echo "  Patching $svc -> type=LoadBalancer"
    kubectl patch svc "$svc" -n "$NAMESPACE" -p '{"spec": {"type": "LoadBalancer"}}' || {
      echo "  Patch failed for $svc"
      continue
    }
  else
    echo "  Already type=LoadBalancer"
  fi

  # wait for external endpoint
  start=$(date +%s)
  endpoint=""
  while true; do
    # try hostname first, then ip
    endpoint=$(kubectl get svc "$svc" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)
    if [ -z "$endpoint" ]; then
      endpoint=$(kubectl get svc "$svc" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    fi
    if [ -n "$endpoint" ]; then
      break
    fi
    now=$(date +%s)
    elapsed=$((now - start))
    if [ $elapsed -ge $TIMEOUT ]; then
      echo "  Timeout waiting for external endpoint for $svc (waited ${elapsed}s)"
      break
    fi
    printf "  Waiting for EXTERNAL-IP/hostname for %s... (%ds elapsed)\r" "$svc" "$elapsed"
    sleep "$SLEEP"
  done

  # get published port (client port)
  PORT=$(kubectl get svc "$svc" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || true)
  if [ -n "$endpoint" ]; then
    echo
    echo "  Service $svc exposed at: http://${endpoint}:${PORT}"

    # quick reachability test using curl (HTTP). try HTTPS only if HTTP fails and common https port used
    if command -v curl >/dev/null 2>&1; then
      echo -n "  Testing HTTP reachability... "
      if curl -sS -I --connect-timeout 5 "http://${endpoint}:${PORT}/" >/dev/null 2>&1; then
        echo "OK (HTTP)"
      else
        echo "failed (HTTP)"
        # try https if port looks like HTTPS or as a second attempt
        if curl -sS -I --connect-timeout 5 "https://${endpoint}:${PORT}/" >/dev/null 2>&1; then
          echo "  HTTPS reachable"
        else
          echo "  Neither HTTP nor HTTPS responded quickly; check target health and security groups."
        fi
      fi
    else
      echo "  curl not available to test reachability."
    fi

    # if aws cli present, show ELB/target-group health for this LB
    if command -v aws >/dev/null 2>&1; then
      REGION=${AWS_REGION:-$(kubectl config view --minify --output 'jsonpath={.clusters[0].cluster.server}' | sed -n 's/.*amazonaws.com\/api\/.*/eu-west-2/p' || echo "eu-west-2")}
      LB_DNS="$endpoint"
      echo "  Inspecting load balancer ($LB_DNS) via AWS API (region: $REGION)..."
      # try elbv2
      LB_ARN=$(aws elbv2 describe-load-balancers --region "$REGION" --query "LoadBalancers[?DNSName=='${LB_DNS}'].LoadBalancerArn | [0]" --output text 2>/dev/null || true)
      if [ -n "$LB_ARN" ] && [ "$LB_ARN" != "None" ]; then
        echo "    Found ALB/NLB ARN: $LB_ARN"
        aws elbv2 describe-listeners --region "$REGION" --load-balancer-arn "$LB_ARN" --output json || true
        echo "    Target groups and health:"
        for tg in $(aws elbv2 describe-target-groups --region "$REGION" --load-balancer-arn "$LB_ARN" --query 'TargetGroups[].TargetGroupArn' --output text || true); do
          echo "      Target group: $tg"
          aws elbv2 describe-target-health --region "$REGION" --target-group-arn "$tg" --output json || true
        done
      else
        # try classic ELB
        LB_NAME=$(aws elb describe-load-balancers --region "$REGION" --query "LoadBalancerDescriptions[?DNSName=='${LB_DNS}'].LoadBalancerName | [0]" --output text 2>/dev/null || true)
        if [ -n "$LB_NAME" ] && [ "$LB_NAME" != "None" ]; then
          echo "    Found classic ELB: $LB_NAME"
          aws elb describe-instance-health --region "$REGION" --load-balancer-name "$LB_NAME" --output json || true
        else
          echo "    Could not find LB via AWS API. Ensure AWS CLI region/permissions are correct."
        fi
      fi
    fi

  else
    # fallback: if nodes have public IPs and service has nodePort(s), show NodePort access
    NODEPORT=$(kubectl get svc "$svc" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || true)
    if [ -n "$NODEPORT" ]; then
      NODE_IP=$(kubectl get nodes -o wide --no-headers | awk '{print $6; exit}')
      if [ -n "$NODE_IP" ]; then
        echo "  No LB assigned yet. NodePort access (if node has public IP): http://${NODE_IP}:${NODEPORT}"
      else
        echo "  No LB assigned and no node public IP found. Re-run this script later or check cloud provider."
      fi
    else
      echo "  No load balancer or nodePort available for $svc"
    fi
  fi
done

echo
echo "Done."