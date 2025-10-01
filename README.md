# Gradient Boosted Regression

# Setup

1. Ensure you have a Kaggle account and have set up your API token as per [Kaggle API documentation](https://www.kaggle.com/docs/api#authentication).

2. Set up and activate Conda environment:
   ```bash
   conda env create -f conda.yml
   conda activate gradient-boosted-regressor
   ```

3. Launch MLflow UI:
   ```bash
   mlflow ui --port 5000 --host 127.0.0.1
   ```
   Access the UI at `http://localhost:5000`.

4. Set up AWS
```bash
aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"

aws iam attach-role-policy --role-name "$AWS_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

aws s3 cp data/processed s3://$BUCKET/processed/ --recursive

aws iam put-role-policy --role-name "SageMakerHW" \
  --policy-name "SageMakerS3Access" \
  --policy-document file://sagemaker-s3-policy.json
```