import os
import scipy.io
from read_cell import UnsupportedFileTypeError


def read_mat_rcell(path: str):
    mat = scipy.io.loadmat(path)

    rcell = mat['rCell']

    data = dict()
    data['cellInfo'] = {}
    protocol_struct = rcell[0, 0]['protocol']
    data['cellInfo']['CellName'] = 'cell' + rcell[0, 0]['cellInfo'][0, 0]['id'][0]
    data['cellInfo']['Species'] = rcell[0, 0]['cellInfo'][0, 0]['species'][0]
    celltype = rcell[0, 0]['cellInfo'][0, 0]['cellType'][0]
    brain_area = ['SNc' if celltype.startswith('SNc') else celltype]
    brain_area = ['DA' if brain_area[0].startswith('DA') else brain_area[0]]
    brain_area = ['CA' if brain_area[0].startswith('CA') else brain_area[0]]
    brain_area = ['Cortex' if (brain_area[0].startswith('L2') or brain_area[0].startswith('L5')) else brain_area[0]]
    data['cellInfo']['BrainArea'] = brain_area[0]
    celltypegroup = ['PC-L5' if celltype == 'L5PC' else 'PC-L2' if celltype == 'L2PC'
    else 'PC' if (celltype.endswith('PC') or celltype == 'Amygdala') else celltype]
    celltypegroup = ['IN' if celltypegroup[0].endswith('IN') or celltypegroup[0].endswith('FS') else celltypegroup[0]]
    data['cellInfo']['CellTypeGroup'] = celltypegroup[0]

    for protocol_idx in range(protocol_struct.size):
        protocol = protocol_struct[0, protocol_idx]['name'][0, 0][0]
        repetition = protocol_struct[0, protocol_idx]['repetition']
        traces = repetition[0, 0]['traceList']
        if protocol not in data:
            data[protocol] = {}
        for repetition_idx in range(traces.size):
            tracelist = traces[0, repetition_idx]
            repetition_key = 'repetition: ' + str(repetition_idx)
            if repetition_key not in data[protocol]:
                data[protocol][repetition_key] = {}
            for trace_idx in range(tracelist.size):
                sweep_key = 'sweep: ' + str(trace_idx)
                if sweep_key not in data[protocol][repetition_key]:
                    data[protocol][repetition_key][sweep_key] = {}  # Initialize the trace level
                data[protocol][repetition_key][sweep_key]['t'] = tracelist[0, trace_idx][0, 0]['t'].T
                data[protocol][repetition_key][sweep_key]['data'] = tracelist[0, trace_idx][0, 0]['data']

    return data


def read_mat_all_rcell(path: str):
    file_list = os.listdir(path)

    # Check that there are files in folder_path
    if not file_list:
        raise FileNotFoundError("No data file found, check the path to data")
    else:
        print('File names loaded.')

    # Create a dictionary containing the data from all files present in folder_path.
    # Keys are the cell ids and values the content of the rcell files.
    all_cells = {}
    for file_name in file_list:
        rcell = {}
        try:
            rcell = read_mat_rcell(os.path.join(path, file_name))
        except UnsupportedFileTypeError as e:
            print(f"Unsupported file type error: {e}")
            raise
        all_cells[rcell['cellInfo']['CellName']] = rcell

    return all_cells


if __name__ == "__main__":
    path = '/home/lucas/BBP/Data/rCells'
    all_cells = read_mat_all_rcell(path)
