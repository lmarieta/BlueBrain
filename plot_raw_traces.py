import pandas as pd
from read_mat_rcell import get_all_rcell_names, read_mat_rcell
import matplotlib.pyplot as plt
from read_cell import get_all_cells
import re
import time
import os
from read_cell import UnsupportedFileTypeError
import plotly.graph_objects as go
from CellCount import load_data


def plot_raw_traces(rcell_path: str, acells_path: str, db_path:str, protocol: str):
    df = pd.DataFrame(columns=['cell_names', 'BrainArea', 'CellTypeGroup', 'Species'])
    df['cell_names'] = get_all_rcell_names(rcell_path)
    acells = get_all_cells(acells_path)
    db_cells = load_data(db_path)
    # unique combinations of brain area, cell type and species
    i = 0
    n = len(df['cell_names'])
    cell_names_to_plot = df['cell_names']
    # TODO create list of above threshold stim, brain area, cell type group and species without reading rcells
    pattern = r'cell(\d+_\d+)'
    for cell_name in cell_names_to_plot:
        match = re.search(pattern, cell_name)

        if match:
            cell_id = match.group(1)
        else:
            continue
        cell_info = db_cells[db_cells['CellID'] == cell_id]
        brain_area = cell_info['BrainArea'].iloc[0]
        df['BrainArea'][df['cell_names'] == cell_name] = brain_area
        cell_type_group = cell_info['CellTypeGroup'].iloc[0]
        df['CellTypeGroup'][df['cell_names'] == cell_name] = cell_type_group
        specie = cell_info['Species'].iloc[0]
        df['Species'][df['cell_names'] == cell_name] = specie

    brain_areas = df['BrainArea'].dropna().unique()
    cell_type_groups = df['CellTypeGroup'].dropna().unique()
    species = df['Species'].dropna().unique()
    stims = get_stimuli(acells, protocol)
    i = 0

    # TODO plot single stimulus
    for stim in stims:
        for brain_area in brain_areas:
            for cell_type in cell_type_groups:
                for specie in species:
                    # Filter cells based on conditions
                    cells_to_plot = [cell_name for cell_name in cell_names_to_plot if
                                     brain_area == df['BrainArea'].loc[df['cell_names'] == cell_name].values[0] and
                                     cell_type == df['CellTypeGroup'].loc[df['cell_names'] == cell_name].values[0] and
                                     specie == df['Species'].loc[df['cell_names'] == cell_name].values[0]]
                    if cells_to_plot:
                        # Create a single figure for each combination of stim, specie, brain area, and cell type
                        legend_data = []
                    for cell_name in cell_names_to_plot:
                        try:
                            print(f'Loading {i / n * 100:.2f}%')
                            i = i + 1
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

                        acell_data = [element for element in acell['protocol']
                                      if element.get('name') == protocol]
                        if not acell_data or 'repetition' not in acell_data:
                            continue
                        acell_data = acell_data[0]
                        for repetition_index, repetition in rcell[protocol].items():
                            for sweep_index, sweep in repetition.items():
                                match = re.search(r'repetition: (\d+)', repetition_index)
                                rep_idx = int(match.group(1))
                                match = re.search(r'sweep: (\d+)', sweep_index)
                                sweep_idx = int(match.group(1))
                                spikecount = get_spikecount(acell=acell_data, repetition=rep_idx, sweep=sweep_idx)
                                if spikecount:
                                    if isinstance(acell_data['repetition'], dict):
                                        stim_data = acell_data['repetition']['stim'][sweep_idx]
                                    else:
                                        stim_data = acell_data['repetition'][rep_idx]['stim'][sweep_idx]
                                    if stim == stim_data:
                                        start_idx = get_start_stimulus_indice(acell=acell_data, repetition=rep_idx)
                                        end_idx = get_second_ap_indice(acell=acell_data, repetition=rep_idx,
                                                                   sweep=sweep_idx)
                                        x = sweep['t'][start_idx:end_idx]
                                        y = sweep['data'][start_idx:end_idx]
                                        x = [item for sublist in x for item in sublist]
                                        y = [item for sublist in y for item in sublist]
                                        label = f'{cell_name}, {repetition_index}, {sweep_index}'
                                        hover_text = [f'{y_val:.2f} {label}' for y_val, label in zip(y, [label] * len(y))]
                                        # Create a scatter plot with hover text
                                        trace = go.Scatter(
                                            x=x,
                                            y=y,
                                            mode='lines',
                                            name=label,
                                            hoverinfo='text',  # Show hover text
                                            text=hover_text,  # Hover text for each point
                                        )

                                        legend_data.append(trace)

                    # Create layout
                    layout = go.Layout(
                        title=f'{brain_area}, {cell_type}, {specie}, {stim}mA',
                        xaxis=dict(title='t[s]'),
                        yaxis=dict(title='Voltage[mV]'),
                    )
                    # Create figure
                    fig = go.Figure(data=legend_data, layout=layout)

                    # Show the interactive plot
                    fig.show()


def get_start_stimulus_indice(acell, repetition):
    if isinstance(acell['repetition'], dict):
        idx = acell['repetition']['stim_ids'][1]
    else:
        idx = acell['repetition'][repetition]['stim_ids'][1]
    return idx


def get_second_ap_indice(acell, repetition, sweep):
    if isinstance(acell['repetition'], dict):
        indices = acell['repetition']['peak_indices'][sweep]
    else:
        indices = acell['repetition'][repetition]['peak_indices'][sweep]
    if isinstance(indices, list) and len(indices) >= 2:
        idx = min([indices[1] - 100, indices[0] + 500])  # empirical value
    elif isinstance(indices, int):
        idx = indices + 1000
    else:
        idx = -1
    return idx


def get_spikecount(acell, repetition, sweep):
    if isinstance(acell['repetition'], dict):
        spikecount = acell['repetition']['spikecount'][sweep]
    else:
        spikecount = acell['repetition'][repetition]['spikecount'][sweep]
    return spikecount


def get_stimuli(acells, protocol):
    stims = set()
    #for cell_name, acell in acells.items():
    cell_name = 'aCell189_2'
    acell = acells[cell_name]
    acell_data = [element for element in acell['protocol'] if element.get('name') == protocol]
    if acell_data:
        acell_data = acell_data[0]
    else:
        pass
        # continue
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
    rcell_path = 'C:\\Projects\\ASSProject\\Data\\matData'
    acells_path = 'C:\\Projects\\ASSProject\\Analysis\\Data\\jsonData'
    db_path = 'C:\\Projects\\ASSProject\\Analysis\\CellList30-May-2022.csv'
    # ['FirePattern', 'IDRest', 'HyperDePol', 'IV', 'PosCheops', 'APWaveform', 'DeHyperPol', 'sAHP']
    protocol = 'IDRest'

    start_time = time.time()
    plot_raw_traces(rcell_path=rcell_path, acells_path=acells_path, db_path=db_path, protocol=protocol)
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the result
    print(f"Script execution time: {elapsed_time:.2f} seconds")
