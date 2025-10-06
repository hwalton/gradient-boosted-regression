#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME=${CLUSTER_NAME:-gbr-cluster}
REGION=${REGION:-eu-west-2}
NODE_ROLE_NAME=${NODE_ROLE_NAME:-gbr-eks-nodegroup-role}

echo "Deleting EKS cluster resources for cluster='$CLUSTER_NAME' region='$REGION'..."

# 1) Try eksctl delete cluster (deletes nodegroups & stacks it created)
if command -v eksctl >/dev/null 2>&1; then
  echo "Running: eksctl delete cluster --name ${CLUSTER_NAME} --region ${REGION}"
  eksctl delete cluster --name "${CLUSTER_NAME}" --region "${REGION}" || true
else
  echo "eksctl not installed, skipping eksctl delete cluster"
fi

# 2) Delete any CloudFormation stacks that match eksctl-<cluster> (leftovers)
STACKS=$(aws cloudformation list-stacks --region "${REGION}" \
  --query "StackSummaries[?contains(StackName, 'eksctl-${CLUSTER_NAME}')].StackName" --output text || true)

if [ -n "$STACKS" ] && [ "$STACKS" != "None" ]; then
  echo "Found CloudFormation stacks to delete:"
  for s in $STACKS; do
    echo " - deleting stack: $s"
    aws cloudformation delete-stack --region "${REGION}" --stack-name "$s" || true
    echo " - waiting for deletion of $s ..."
    aws cloudformation wait stack-delete-complete --region "${REGION}" --stack-name "$s" || true
  done
else
  echo "No leftover eksctl-* CloudFormation stacks found."
fi

# 3) Remove IAM instance profiles / role if it exists (nodegroup role)
if aws iam get-role --role-name "${NODE_ROLE_NAME}" >/dev/null 2>&1; then
  echo "Cleaning up IAM role and instance profiles for role: ${NODE_ROLE_NAME}"

  # detach managed policies
  POLICIES=$(aws iam list-attached-role-policies --role-name "${NODE_ROLE_NAME}" --query "AttachedPolicies[].PolicyArn" --output text || true)
  for p in $POLICIES; do
    [ -n "$p" ] && echo " Detaching $p" && aws iam detach-role-policy --role-name "${NODE_ROLE_NAME}" --policy-arn "$p" || true
  done

  # remove inline policies
  INLINES=$(aws iam list-role-policies --role-name "${NODE_ROLE_NAME}" --query "PolicyNames[]" --output text || true)
  for ip in $INLINES; do
    [ -n "$ip" ] && echo " Removing inline policy $ip" && aws iam delete-role-policy --role-name "${NODE_ROLE_NAME}" --policy-name "$ip" || true
  done

  # remove role from instance profiles and delete profiles
  IPROFILES=$(aws iam list-instance-profiles-for-role --role-name "${NODE_ROLE_NAME}" --query "InstanceProfiles[].InstanceProfileName" --output text || true)
  for ipn in $IPROFILES; do
    if [ -n "$ipn" ]; then
      echo " Removing role from instance profile $ipn"
      aws iam remove-role-from-instance-profile --instance-profile-name "$ipn" --role-name "${NODE_ROLE_NAME}" || true
      echo " Deleting instance profile $ipn"
      aws iam delete-instance-profile --instance-profile-name "$ipn" || true
    fi
  done

  # finally delete the role
  echo "Deleting role ${NODE_ROLE_NAME}"
  aws iam delete-role --role-name "${NODE_ROLE_NAME}" || true
else
  echo "IAM role ${NODE_ROLE_NAME} not found; skipping role cleanup."
fi

echo "Cleanup finished. Some async deletes (CloudFormation) may still be in progress on AWS."
echo "Verify with: eksctl get cluster --region ${REGION} || aws cloudformation list-stacks --region ${REGION} --query \"StackSummaries[?contains(StackName,'eksctl-${CLUSTER_NAME}')].StackName\" --output table"