# load .env if needed
set -a; source .env; set +a

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
# update Lambda to the image you tested
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
aws lambda update-function-code --function-name gbr-trainer --image-uri "${ECR_URI}"

# poll until function is Active and update successful
aws lambda get-function-configuration --function-name gbr-trainer \
  --query '{State:State,LastUpdateStatus:LastUpdateStatus,LastUpdateStatusReason:LastUpdateStatusReason}' --output json
# repeat above until State == "Active" and LastUpdateStatus == "Successful"