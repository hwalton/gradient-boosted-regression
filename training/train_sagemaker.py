import os
import logging
import boto3
import sagemaker
from sagemaker.sklearn import SKLearn
import re
import sys
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pathlib import Path

def _get_aws_account_id():
    try:
        sts = boto3.client("sts")
        return sts.get_caller_identity()["Account"]
    except Exception as e:
        logger.debug("STS lookup failed: %s", e)
        return None

def resolve_role_arn():
    """
    Resolve SageMaker role ARN from env vars. Priority:
      1) SAGEMAKER_ROLE_ARN
      2) ROLE (full ARN)
      3) construct from AWS_ACCOUNT_ID or STS + SAGEMAKER_ROLE_NAME
    Raises RuntimeError if no role can be determined.
    """
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    if role_arn:
        return role_arn

    account = os.environ.get("AWS_ACCOUNT_ID") or _get_aws_account_id()
    role_name = os.environ.get("AWS_ROLE")  # change as needed
    if account and role_name:
        return f"arn:aws:iam::{account}:role/{role_name}"

    raise RuntimeError("SageMaker role not found. Set SAGEMAKER_ROLE_ARN or ROLE or AWS creds to derive account id.")

def preflight_check(role_arn: str, bucket: str, region: str | None = None) -> None:
    """Validate role ARN format, role existence & trust policy and S3 bucket reachability.
    Raises RuntimeError on failure (so you get immediate feedback before creating a training job)."""
    # quick ARN format check
    arn_re = re.compile(r"^arn:aws[a-z\-]*:iam::\d{12}:role\/[A-Za-z0-9+=,.@_/\-]+$")
    if not arn_re.match(role_arn):
        raise RuntimeError(f"Invalid role ARN format: {role_arn!r}")

    iam = boto3.client("iam")
    # extract role name from ARN
    role_name = role_arn.split(":")[-1].split("/", 1)[-1]
    try:
        resp = iam.get_role(RoleName=role_name)
    except ClientError as e:
        raise RuntimeError(f"IAM role not found or not accessible: {e}") from e

    # check trust policy includes SageMaker service principal
    trust_doc = resp.get("Role", {}).get("AssumeRolePolicyDocument", {})
    principals = []
    for stmt in trust_doc.get("Statement", []):
        p = stmt.get("Principal", {})
        service = p.get("Service")
        if isinstance(service, list):
            principals.extend(service)
        elif service:
            principals.append(service)
    if not any("sagemaker.amazonaws.com" in s for s in principals if isinstance(s, str)):
        raise RuntimeError("IAM role trust policy does not allow sagemaker.amazonaws.com to assume the role.")

    # check S3 bucket reachable
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        raise RuntimeError(f"S3 bucket '{bucket}' not accessible or does not exist: {e}") from e

    # quick quota hint (optional): check Service Quotas or call describe-account-limits via SageMaker if desired
    return

def main():
    load_dotenv(Path(".env"), override=True)
    sess = sagemaker.Session()

    try:
        role = resolve_role_arn()
    except RuntimeError as e:
        logger.error(str(e))
        raise

    bucket = os.environ.get("BUCKET")
    if not bucket:
        raise RuntimeError("Environment variable BUCKET is required (s3 bucket name).")

    # run fast preflight validations so user sees immediate errors locally
    try:
        preflight_check(role, bucket, os.getenv("REGION"))
    except Exception as e:
        logger.error("Preflight checks failed: %s", e)
        # exit early to avoid waiting on SageMaker API call
        sys.exit(1)

    sk = SKLearn(
        entry_point="training/sagemaker_entry.py",
        source_dir=".",                # repo root (uploads code)
        role=role,
        instance_type=os.environ.get("SM_INSTANCE_TYPE", "ml.m5.large"),
        instance_count=int(os.environ.get("SM_INSTANCE_COUNT", "1")),
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sess,
    )

    s3_input = f"s3://{bucket}/processed/"
    logger.info("Starting SageMaker SKLearn job with input: %s", s3_input)
    sk.fit({"training": s3_input})

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    main()