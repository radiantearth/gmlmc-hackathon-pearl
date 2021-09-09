# tutorial/01-create-workspace.py
import os
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication

# get your TENANT_ID from "az account show --output table"
# get your "subscription_id" from "az account list --output table"

TENANT_ID = os.getenv("TENANT_ID")
interactive_auth = InteractiveLoginAuthentication(tenant_id=TENANT_ID)
subscription_id = os.getenv("subscription_id")


ws = Workspace.create(name='name', # provide a name for your workspace
                  subscription_id=subscription_id, # provide your subscription ID
                  resource_group='name', # provide a resource group name
                  create_resource_group=True,
                  location='location') # For example: 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.

# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')
