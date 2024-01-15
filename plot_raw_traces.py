from read_mat_rcell import get_all_rcell_names, read_mat_rcell
import matplotlib.pyplot as plt
from read_cell import get_all_cells
import re
import time
import os
from read_cell import UnsupportedFileTypeError


def plot_raw_traces(path: str, protocol: str, acells: dict):
    all_cells = get_all_rcell_names(path)

    # unique combinations of brain area, cell type and species
    brain_areas = set()
    cell_type_groups = set()
    species = set()
    rcell = {}
    i = 0
    n = len(all_cells)
    for cell_name in all_cells:
        try:
            rcell = read_mat_rcell(os.path.join(path, cell_name + '.mat'))
        except UnsupportedFileTypeError as e:
            print(f"Unsupported file type error: {e}")
            continue
        except (AttributeError, IndexError):
            print('Error for cell: ' + cell_name)
            continue
        cell_info = rcell.get('cellInfo', {})
        brain_area = cell_info.get('BrainArea')
        cell_type_group = cell_info.get('CellTypeGroup')
        specie = cell_info.get('Species')
        brain_areas.add(brain_area)
        cell_type_groups.add(cell_type_group)
        species.add(specie)
        print(f'Loading {i/n*100:.2f}%')
        i = i + 1

    stims = get_stimuli(acells, protocol)
    i = 0
    n = len(all_cells)
    for cell_name in all_cells:
        print(f'Loading {i/n*100:.2f}%')
        i = i+1
        try:
            rcell = read_mat_rcell(os.path.join(path, cell_name + '.mat'))
        except UnsupportedFileTypeError as e:
            print(f"Unsupported file type error: {e}")
            continue
        except (AttributeError, IndexError):
            print('Error for cell: ' + cell_name)
            continue
        cell_name = rcell['cellInfo']['CellName']
        acell_name = 'aC' + cell_name[1:]
        try:
            acell = acells[acell_name]
        except KeyError:
            continue
        for stim in stims:
            for brain_area in brain_areas:
                for cell_type in cell_type_groups:
                    for specie in species:
                        plt.figure()
                        if (rcell['cellInfo']['Species'] == specie
                                and rcell['cellInfo']['BrainArea'] == brain_area
                                and rcell['cellInfo']['CellTypeGroup'] == cell_type):
                            acell_data = [element for element in acell['protocol']
                                          if element.get('name') == protocol][0]
                            for repetition_index, repetition in rcell[protocol].items():
                                for sweep_index, sweep in repetition.items():
                                    match = re.search(r'repetition: (\d+)', repetition_index)
                                    rep_idx = int(match.group(1))
                                    match = re.search(r'sweep: (\d+)', sweep_index)
                                    sweep_idx = int(match.group(1))
                                    spikecount = get_spikecount(acell=acell_data, repetition=rep_idx, sweep=sweep_idx)
                                    if spikecount:
                                        stim_data = acell_data['repetition'][rep_idx]['stim'][sweep_idx]
                                        if stim == stim_data:
                                            idx = get_second_ap_indices(acell=acell_data, repetition=rep_idx, sweep=sweep_idx)
                                            x = sweep['t'][0:idx]
                                            y = sweep['data'][0:idx]
                                            label = f'{cell_name}, {repetition_index}, {sweep_index}'
                                            plt.plot(x, y, label=label)

                        plt.xlabel('t[s]')
                        plt.ylabel('Voltage[mV]')
                        plt.title(brain_area + ', ' + cell_type + ', ' + specie + ', ' + str(stim) + 'mA')
                        plt.grid(True)
                        plt.legend(labelcolor='none')
                        plt.close()
                        # plt.show()


def get_second_ap_indices(acell, repetition, sweep):
    indices = acell['repetition'][repetition]['peak_indices'][sweep]
    if isinstance(indices, list) and len(indices) >= 2:
        idx = indices[1] - 200  # empirical value
    elif isinstance(indices, int):
        idx = indices + 1000
    else:
        idx = -1
    return idx


def get_spikecount(acell, repetition, sweep):
    spikecount = acell['repetition'][repetition]['spikecount'][sweep]
    return spikecount


def get_stimuli(acells, protocol):
    stims = set()
    for cell_name, acell in acells.items():
        acell_data = [element for element in acell['protocol'] if element.get('name') == protocol]
        if acell_data:
            acell_data = acell_data[0]
        else:
            continue
        if 'repetition' in acell_data:
            if isinstance(acell_data['repetition'], list):
                for repetition in acell_data['repetition']:
                    stim = repetition['stim']
                    stims.add(tuple(stim))
            else:
                stim = acell_data['repetition']['stim']
                stims.add(tuple(stim))
    return set().union(*stims)


if __name__ == "__main__":
    raw_path = 'C:\\Projects\\ASSProject\\Data\\matData'
    acells_path = 'C:\\Projects\\ASSProject\\Analysis\\Data\\jsonData'
    # ['FirePattern', 'IDRest', 'HyperDePol', 'IV', 'PosCheops', 'APWaveform', 'DeHyperPol', 'sAHP']
    protocol = 'IDRest'
    start_time = time.time()
    acells = get_all_cells(acells_path)
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the result
    print(f"Script execution time: {elapsed_time:.2f} seconds")
    start_time = time.time()
    plot_raw_traces(path=raw_path, protocol=protocol, acells=acells)
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the result
    print(f"Script execution time: {elapsed_time:.2f} seconds")
