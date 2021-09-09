import os
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.authentication import InteractiveLoginAuthentication


if __name__ == "__main__":
    TENANT_ID = os.getenv("TENANT_ID")
    experiment_ID = os.getenv("experiment_ID")
    interactive_auth = InteractiveLoginAuthentication(tenant_id=TENANT_ID)
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='detroit-unet2-lowernodatathresh-7cls-eval')

    datastore = ws.get_default_datastore()
    print(datastore)
    # find the experiment Run ID through your Azure portal https://ml.azure.com/experiments/

    config = ScriptRunConfig(source_directory='./src',
                             script='eval.py',
                             compute_target='gpu-nc12',
                             arguments=[
                                 '--model_fn', 'data/9clas-combined-fcn.pt',
                                 '--input_fn', 'data/cc6_ne_test_combined.csv',
                                 '--output_dir', './outputs',
                                 '--num_classes', 9,
                                 '--label_transform', 'naip',
                                 '--model', 'fcn',
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
