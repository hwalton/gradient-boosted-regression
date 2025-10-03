set -a; source .env; set +a

cat > /tmp/gbr-s3-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::$BUCKET"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject","s3:PutObject","s3:HeadObject"],
      "Resource": ["arn:aws:s3:::$BUCKET/*"]
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name gbr-lambda-role \
  --policy-name gbr-s3-access \
  --policy-document file:///tmp/gbr-s3-policy.json