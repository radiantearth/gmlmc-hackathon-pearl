import os
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication

# get your TENANT_ID from "az account show --output table"
# get your "subscription_id" from "az account list --output table"



if __name__ == "__main__":
    try:
        TENANT_ID = os.getenv("TENANT_ID")
        interactive_auth = InteractiveLoginAuthentication(tenant_id=TENANT_ID)
        ws = Workspace.from_config()
    except:
        ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='experiment-train-md-combined')
    config = ScriptRunConfig(source_directory='./model',
                             script='train.py',
                             compute_target='gpu-nc12-va',
                             arguments=[
                                 '--input_fn', 'data/cc6_ne_train_combined.csv',
                                 '--input_fn_val', 'data/cc6_ne_val_combined.csv',
                                 '--output_dir',  './outputs',
                                 '--save_most_recent',
                                 '--num_epochs', 10,
                                 '--num_classes', 9,
                                 '--label_transform', 'naip'
                             ])

    # set up pytorch environment
    pytorch_env = Environment.from_conda_specification(
        name='lulc-pytorch-env',
        file_path='./.azureml/pytorch-env.yml'
    )

    # Specify a GPU base image
    pytorch_env.docker.enabled = True
    pytorch_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'


    config.run_config.environment = pytorch_env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)
