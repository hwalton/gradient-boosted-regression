#!/usr/bin/env python3
import os
import aws_cdk as cdk

from infrastructure.infrastructure_stack import InfrastructureStack

app = cdk.App()

# Prefer environment variables for region/account to avoid calling app.node.try_get_context
region = (
    os.getenv("CDK_DEFAULT_REGION")
    or os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "eu-west-2"
)
account = os.getenv("CDK_DEFAULT_ACCOUNT") or os.getenv("AWS_ACCOUNT_ID")
env = cdk.Environment(account=account, region=region) if account else None

# Prefer env var for cluster name (avoid app.node.try_get_context which triggers constructs typeguard issues)
cluster_name = os.getenv("CLUSTER_NAME") or "gbr-cluster5"

InfrastructureStack(app, "InfrastructureStack", env=env, cluster_name=cluster_name)

app.synth()
