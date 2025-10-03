# Gradient Boosted Regression

# Setup

1. Ensure you have a Kaggle account and have set up your API token as per [Kaggle API documentation](https://www.kaggle.com/docs/api#authentication).

2. Set up and activate Conda environment:
   ```bash
   conda env create -f conda.yml
   conda activate gbr
   ```

3. Launch MLflow UI:
   ```bash
   mlflow ui --port 5000 --host 127.0.0.1
   ```
   Access the UI at `http://localhost:5000`.

4. Set up AWS
```bash
aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"

# create role
aws iam create-role --role-name gbr-lambda-role \
  --assume-role-policy-document file://training/lambda-trust.json

# attach basic Lambda logging managed policy
aws iam attach-role-policy --role-name gbr-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# attach S3 inline policy (replace BUCKET_NAME in file or create dynamically)
aws iam put-role-policy --role-name gbr-lambda-role --policy-name gbr-s3-access \
  --policy-document file://training/lambda-s3-policy.json

# get the role ARN
ROLE_ARN=$(aws iam get-role --role-name gbr-lambda-role --query 'Role.Arn' --output text)
echo $ROLE_ARN

# Check function status
aws lambda get-function-configuration --function-name gbr-trainer \
  --query '{State:State,StateReason:StateReason,LastUpdateStatus:LastUpdateStatus,LastUpdateStatusReason:LastUpdateStatusReason}' --output json

# Confirm image URI used by lambda:
aws lambda get-function --function-name gbr-trainer --query 'Code.ImageUri' --output text

# publish layer
aws lambda publish-layer-version --layer-name gbr-deps --description "gbr deps" \
  --zip-file fileb://layer.zip --compatible-runtimes python3.10 --region eu-west-2

  aws lambda update-function-configuration --function-name gbr-trainer \
  --layers arn:aws:lambda:eu-west-2:${ACCOUNT_ID}:layer:gbr-deps:<version>
```

lambda env vars:
`MODEL_BUCKET`
`PROCESSED_PREFIX`
`MODEL_KEY`
`GIT_REPO=https://github.com/<owner>/<repo>.git`
`GIT_REF=master` (optional)
`FORCE_PULL=true` (optional — force pull updates on existing clone)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ aws tutorial ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`aws sts get-caller-identity`
`npm --version`
`npm install -g aws-cdk@latest`
`cdk bootstrap aws://<account_id>/<region>`