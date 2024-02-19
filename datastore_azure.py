from azureml.core import Workspace, Datastore

# Load your workspace
ws = Workspace.from_config()

# Register a datastore (adjust the parameters based on your storage)
blob_datastore = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name='your_blob_datastore',
    container_name='your_blob_container',
    account_name='your_storage_account_name',
    account_key='your_storage_account_key'
)
