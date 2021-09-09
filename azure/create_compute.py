import os
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import InteractiveLoginAuthentication

# try:
TENANT_ID = os.getenv("TENANT_ID")
interactive_auth = InteractiveLoginAuthentication(tenant_id=TENANT_ID)
# except:
#     print("Need to export TENANT_ID, and get if from 'az account show --output table'!")


ws = Workspace.from_config()

# Choose a name for your GPU cluster
gpu_cluster_name = "gpu-nc12"

# Verify that the cluster does not exist already
try:
    gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC12',
                                                           idle_seconds_before_scaledown=1200,
                                                           min_nodes=0,
                                                           max_nodes=1)
    gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)

gpu_cluster.wait_for_completion(show_output=True)
