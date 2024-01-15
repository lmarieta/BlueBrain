from read_mat_rcell import read_mat_all_rcell
import matplotlib.pyplot as plt
from read_cell import get_all_cells
import re


def plot_raw_traces(path: str, protocol: str, acells: dict):
    all_cells = read_mat_all_rcell(path)

    # unique combinations of brain area, cell type and species
    brain_areas = set()
    cell_type_groups = set()
    species = set()
    for _, cell in all_cells.items():
        cell_info = cell.get('cellInfo', {})
        brain_area = cell_info.get('BrainArea')
        cell_type_group = cell_info.get('CellTypeGroup')
        specie = cell_info.get('Species')
        brain_areas.add(brain_area)
        cell_type_groups.add(cell_type_group)
        species.add(specie)

    stims = get_stimuli(acells, protocol)

    for stim in stims:
        for brain_area in brain_areas:
            for cell_type in cell_type_groups:
                for specie in species:
                    plt.figure()
                    for _, cell in all_cells.items():
                        cell_name = cell['cellInfo']['CellName']
                        acell_name = 'aC' + cell_name[1:]
                        try:
                            acell = acells[acell_name]
                        except KeyError:
                            print(f"{acell_name} not found in acells. Skipping.")
                            break
                        if (cell['cellInfo']['Species'] == specie
                                and cell['cellInfo']['BrainArea'] == brain_area
                                and cell['cellInfo']['CellTypeGroup'] == cell_type):
                            acell_data = [element for element in acell['protocol']
                                          if element.get('name') == protocol][0]
                            for repetition_index, repetition in cell[protocol].items():
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
                    plt.legend()
                    plt.show()


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
    raw_path = '/home/lucas/BBP/Data/rCells'
    acells_path = '/home/lucas/BBP/Data/jsonData'
    # ['FirePattern', 'IDRest', 'HyperDePol', 'IV', 'PosCheops', 'APWaveform', 'DeHyperPol', 'sAHP']
    protocol = 'IDRest'
    acells = get_all_cells(acells_path)
    plot_raw_traces(path=raw_path, protocol=protocol, acells=acells)
