import channelpedia_api as api
import os


class UnsupportedFileTypeError(BaseException):
    def __init__(self, message):
        super().__init__(message)


def read_file(filename):
    if not is_supported_file_type(filename):
        raise UnsupportedFileTypeError(f"Check supported file types: {filename}")
    return api.get_acell(filename)


def is_supported_file_type(filename):
    # As of writing, only json is supported
    supported_extensions = ['json']
    file_extension = filename.split('.')[-1].lower()
    return file_extension in supported_extensions


def get_all_cells(folder_path: str):
    # Use os.listdir to get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Check that there are files in folder_path
    if not file_list:
        raise FileNotFoundError("No data file found, check the path to data")
    else:
        print('File names loaded.')

    # Create a dictionary containing the data from all files present in folder_path.
    # Keys are the cell ids and values the content of the acell files.
    all_cells = {}
    for file_name in file_list:
        acell = {}
        try:
            acell = read_file(os.path.join(folder_path, file_name))
        except UnsupportedFileTypeError as e:
            print(f"Unsupported file type error: {e}")
            raise
        all_cells['aCell' + acell['aCell']['id']] = acell['aCell']

    return all_cells
