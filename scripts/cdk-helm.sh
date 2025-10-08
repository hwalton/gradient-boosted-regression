# use a profile that can assume the cluster creation role
: "${AWS_PROFILE:=deploy-user}"

# role that CDK created (adjust name if yours differs)
ROLE_NAME="InfrastructureStack10-GBRClusterCreationRole588CD2A-QYP51mz3Wc01"
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text --profile "$AWS_PROFILE")

# get your ARN
ARN=$(aws sts get-caller-identity --query Arn --output text --profile "$AWS_PROFILE")

# add mapping (gives your principal system:masters) — safe to skip if done already
eksctl create iamidentitymapping \
  --cluster gbr-cluster10 --region eu-west-2 \
  --arn "$ARN" --username admin --group system:masters --profile "$AWS_PROFILE"

# update kubeconfig using the creation role
aws eks update-kubeconfig --name gbr-cluster10 --region eu-west-2 --role-arn "$ROLE_ARN" --profile "$AWS_PROFILE"

# install the local chart
helm upgrade --install gbr ./helm/gbr -n default --create-namespace