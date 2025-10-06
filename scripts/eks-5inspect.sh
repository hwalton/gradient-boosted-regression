# ...existing code...
kubectl get pods -n default
kubectl describe deployment gbr-airflow -n default
kubectl logs -l app=gbr-airflow -n default -c airflow --tail=200 -f

eksctl get cluster --region=eu-west-2
eksctl get nodegroup --cluster gbr-cluster4 --region=eu-west-2

eksctl utils describe-stacks --cluster=gbr-cluster4 --region=eu-west-2

# ...existing code...
# Allow passing stack name as first arg, otherwise pick the first eksctl nodegroup stack
STACK_NAME=${1:-$(aws cloudformation list-stacks --region=eu-west-2 --query "StackSummaries[?contains(StackName,'eksctl-gbr-cluster4')].StackName | [0]" --output text)}

if [ -z "$STACK_NAME" ] || [ "$STACK_NAME" = "None" ]; then
  echo "No eksctl stack found. Pass stack name as first argument."
  exit 0
fi

echo "Describing CloudFormation events for: $STACK_NAME"
aws cloudformation describe-stack-events --region=eu-west-2 --stack-name "$STACK_NAME" --max-items 50 \
  --query 'StackEvents[?ResourceStatusReason!=null].[Timestamp,LogicalResourceId,ResourceType,ResourceStatus,ResourceStatusReason]' --output table

echo ""
echo "Describe stack resources for: $STACK_NAME"
aws cloudformation describe-stack-resources --region=eu-west-2 --stack-name "$STACK_NAME" --output table
# ...existing code...