#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME=${CLUSTER_NAME:-gbr-cluster4}
REGION=${REGION:-eu-west-2}

command -v aws >/dev/null 2>&1 || { echo "aws CLI not found in PATH"; exit 1; }

echo "Cluster: ${CLUSTER_NAME}  Region: ${REGION}"

# get control-plane security group
CLUSTER_SG=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" \
  --query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' --output text)

if [ -z "${CLUSTER_SG}" ] || [ "${CLUSTER_SG}" = "None" ]; then
  echo "ERROR: control plane security group not found for cluster ${CLUSTER_NAME}"
  exit 1
fi
echo "Control plane SG: ${CLUSTER_SG}"

# show worker instances and their SGs (human friendly)
aws ec2 describe-instances --region "${REGION}" \
  --filters "Name=tag:eks:cluster-name,Values=${CLUSTER_NAME}" \
  --query 'Reservations[].Instances[].{ID:InstanceId,PrivateIp:PrivateIpAddress,PublicIp:PublicIpAddress,State:State.Name,SGs:SecurityGroups}' --output table

# collect unique node SG ids
NODE_SG_IDS=$(aws ec2 describe-instances --region "${REGION}" \
  --filters "Name=tag:eks:cluster-name,Values=${CLUSTER_NAME}" \
  --query "Reservations[].Instances[].SecurityGroups[].GroupId" --output text | tr '\t' '\n' | sort -u)

if [ -z "${NODE_SG_IDS}" ]; then
  echo "No node security groups found for cluster ${CLUSTER_NAME}. Nothing to do."
  exit 0
fi

echo "Node SGs:"
printf '%s\n' ${NODE_SG_IDS}

# ensure control-plane -> kubelet (10250) allowed on node SGs
for sg in ${NODE_SG_IDS}; do
  echo "Processing node SG: ${sg}"

  # check whether a rule allowing cluster SG to TCP/10250 already exists
  EXISTS=$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${sg}" \
    --query "SecurityGroups[0].IpPermissions[?IpProtocol=='tcp' && FromPort==\`10250\` && ToPort==\`10250\`].UserIdGroupPairs[?GroupId=='${CLUSTER_SG}']" \
    --output text)

  if [ -n "${EXISTS}" ]; then
    echo "  Ingress for ${CLUSTER_SG}:10250 already present on ${sg}"
    continue
  fi

  echo "  Authorizing ingress on ${sg} from ${CLUSTER_SG}:10250"
  if aws ec2 authorize-security-group-ingress --region "${REGION}" --group-id "${sg}" \
      --protocol tcp --port 10250 --source-group "${CLUSTER_SG}"; then
    echo "  Ingress added."
  else
    echo "  Warning: failed to add ingress to ${sg} (it may already exist or you lack permissions)."
  fi
done

echo "Done. Re-check node readiness with: kubectl get nodes -o wide"