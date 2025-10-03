# from repo root
set -a; source .env; set +a

# upload
aws s3 cp layer.zip "s3://${BUCKET}/layers/gbr-deps/layer.zip" --region "${REGION}"

# publish and capture ARN
arn=$(aws lambda publish-layer-version \
  --layer-name gbr-deps \
  --content S3Bucket="${BUCKET}",S3Key="layers/gbr-deps/layer.zip" \
  --compatible-runtimes python3.10 \
  --region "${REGION}" \
  --query LayerVersionArn --output text)

# attach to function
aws lambda update-function-configuration --function-name gbr-trainer --layers "${arn}" --region "${REGION}"