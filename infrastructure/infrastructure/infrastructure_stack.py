import os
from typing import Optional

from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_eks as eks,
    CfnOutput,
)
from constructs import Construct

from aws_cdk.lambda_layer_kubectl_v32 import KubectlV32Layer

class InfrastructureStack10(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        cluster_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # prefer explicit parameter, then env var, then fallback default
        cluster_name = (
            cluster_name
            or os.getenv("CLUSTER_NAME")
            or os.getenv("CDK_CLUSTER_NAME")
            or "gbr-cluster10"
        )

        cluster = eks.Cluster(
            self,
            "GBRCluster",
            vpc=ec2.Vpc(self, "GBRVpc", max_azs=2, nat_gateways=1),
            default_capacity=0,
            version=eks.KubernetesVersion.V1_32,
            cluster_name=cluster_name,
            kubectl_layer=KubectlV32Layer(self, "kubectl")
        )




        # Managed nodegroup (adjust instance type/size to your needs)
        cluster.add_nodegroup_capacity(
            "gbr-nodegroup",
            desired_size=2,
            min_size=1,
            max_size=3,
            instance_types=[ec2.InstanceType("t3.small")],
        )

        # Useful outputs
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(
            self,
            "KubeconfigCommand",
            value=f"aws eks update-kubeconfig --name {cluster.cluster_name} --region {Stack.of(self).region}",
        )
