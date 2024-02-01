import numpy as np
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
from get_trace_indices import get_trace_indices
import plotly.express as px
import platform


def plot_raw_traces(rcell_path: str, acells_path: str, db_path: str, output_path: str, protocol: str,
                    threshold_key='spikecount', plot_type='plot_all_traces', repetition_index=[]):
    df = pd.DataFrame(columns=['cell_names', 'BrainArea', 'CellTypeGroup', 'Species', 'repetition',
                               'sweep', 'Threshold stimulus'])
    df['cell_names'] = get_all_rcell_names(rcell_path)
    acells = get_all_cells(acells_path)
    db_cells = load_data(db_path)
    # unique combinations of brain area, cell type and species
    cell_names_to_plot = df['cell_names']
    pattern = r'cell(\d+_\d+)'
    for cell_name in cell_names_to_plot:
        match = re.search(pattern, cell_name)

        if match:
            cell_id = match.group(1)
        else:
            continue
        cell_info = db_cells[db_cells['CellID'] == cell_id]
        if not cell_info.empty:
            brain_area = cell_info['BrainArea'].iloc[0]
            df['BrainArea'][df['cell_names'] == cell_name] = brain_area
            cell_type_group = cell_info['CellTypeGroup'].iloc[0]
            df['CellTypeGroup'][df['cell_names'] == cell_name] = cell_type_group
            specie = cell_info['Species'].iloc[0]
            df['Species'][df['cell_names'] == cell_name] = specie

    # remove cells where there is no acells or if they are not in the database to be analyzed
    brain_areas = df['BrainArea'].dropna().unique()
    cell_type_groups = df['CellTypeGroup'].dropna().unique()
    cell_type_groups = cell_type_groups[cell_type_groups != 'IN']
    df = df[~df['CellTypeGroup'].isin(['IN', np.nan])]
    species = df['Species'].dropna().unique()
    df = get_stimuli(acells=acells, df=df, threshold_key=threshold_key, protocol=protocol)
    df = df.dropna(subset=['repetition'])

    # Create color maps based on stimulus
    stim_values = np.unique(df['Threshold stimulus'])

    if plot_type in ['plot_all_traces', 'plot_mean']:
        # Create a color scale
        color_scale = px.colors.qualitative.Light24
        # Map each specie, brain area, cell type to a color
        color_map = {}
        idx = 0
        for specie in species:
            for brain_area in brain_areas:
                for cell_type_group in cell_type_groups:
                    color_map[(specie, brain_area, cell_type_group)] = color_scale[idx]
                    idx += 1
    else:
        # Create a color scale
        color_scale = px.colors.sequential.Plasma
        # Map each stim value to a color in the color scale
        color_map = {stim_val: color_scale[idx] for idx, stim_val in enumerate(stim_values)}

    cell_names_to_plot = df['cell_names']
    i = 0
    n = len(df['cell_names'].unique())
    legend_data = []
    for brain_area in brain_areas:
        for cell_type in cell_type_groups:
            for specie in species:
                # Filter cells based on conditions
                cells_to_plot = [cell_name for cell_name in cell_names_to_plot if
                                 brain_area == df['BrainArea'].loc[df['cell_names'] == cell_name].values[0] and
                                 cell_type == df['CellTypeGroup'].loc[df['cell_names'] == cell_name].values[0] and
                                 specie == df['Species'].loc[df['cell_names'] == cell_name].values[0]]
                cells_to_plot = set(cells_to_plot)
                if cells_to_plot:
                    label_key = (specie, brain_area, cell_type)
                    # Prepare structures for figures
                    if plot_type == 'plot_single_groups':
                        legend_data = []
                    if plot_type == 'plot_mean':
                        all_traces = []
                    for cell_name in cells_to_plot:
                        try:
                            print(f'Loading {i / n * 100:.2f}%')
                            i = i + 1
                            rcell = read_mat_rcell(os.path.join(rcell_path, cell_name + '.mat'))
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
                                      if element.get('name') == protocol][0]
                        repetitions = rcell[protocol]
                        if repetition_index:
                            keys = ['repetition: ' + str(idx) for idx in repetition_index]
                            repetitions = {key: repetitions[key] for key in keys}
                        for rep_idx, repetition in repetitions.items():
                            match = re.search(r'repetition: (\d+)', rep_idx)
                            rep_idx = int(match.group(1))
                            sweep_idx = df.loc[(df['cell_names'] == cell_name) & (df['repetition'] == rep_idx), 'sweep']
                            if not sweep_idx.empty:
                                sweep_idx = int(sweep_idx.iloc[0])
                                sweep = repetition['sweep: ' + str(sweep_idx)]
                                try:
                                    start_time = get_start_stimulus_time(acell=acell_data, repetition=rep_idx,
                                                                         sweep=sweep_idx)
                                except Exception as e:
                                    # Log the information to a text file
                                    error_message = (f"Error for Protocol: {protocol}, Cell: {cell_name}, "
                                                     f"Repetition: {rep_idx}, Sweep: {sweep_idx}\n")

                                    # Check if the error message is not already in the file
                                    with open('error_log.txt', 'r') as log_file:
                                        log_content = log_file.read()

                                    if error_message not in log_content:
                                        # If not present, append the error message to the file
                                        with open('error_log.txt', 'a+') as log_file:
                                            log_file.write(error_message)
                                    continue
                                # Find the index of the closest time in x
                                start_idx = np.argmin(np.abs(np.array(sweep['t']) - start_time))
                                end_idx = get_second_ap_indice(acell=acell_data, repetition=rep_idx,
                                                               sweep=sweep_idx)

                                x = sweep['t'][start_idx:end_idx] - sweep['t'][start_idx]
                                y = sweep['data'][start_idx:end_idx]
                                # Get indices where values in x are below the threshold
                                x = [item for sublist in x for item in sublist]
                                y = [item for sublist in y for item in sublist]
                                # only keep the first 5ms because an AP is shorter than that
                                time_threshold = 0.005
                                indices_to_truncate = [index for index, item in enumerate(x) if item < time_threshold]
                                x = x[:max(indices_to_truncate) + 1]
                                y = y[:max(indices_to_truncate) + 1]
                                stim = df.loc[(df['cell_names'] == cell_name) & (df['repetition'] == rep_idx),
                                'Threshold stimulus']
                                if plot_type == 'plot_mean':
                                    # Append each individual trace to the list
                                    all_traces.append(y)


                                unique_label = (f'{cell_name}, repetition {rep_idx}, sweep {sweep_idx}, stim '
                                                f'{stim.values[0]:.0f}pA')
                                if plot_type != 'plot_mean':
                                    hover_text = [f'{y_val:.2f} {label}' for y_val, label in
                                                  zip(y, [unique_label] * len(y))]
                                    # Create a scatter plot with hover text
                                    color = color_map[(specie, brain_area, cell_type)] if \
                                        plot_type == 'plot_all_traces' else color_map[stim.values[0]]
                                    trace = go.Scatter(
                                        x=x,
                                        y=y,
                                        mode='lines',
                                        name=unique_label,
                                        hoverinfo='text',  # Show hover text
                                        text=hover_text,  # Hover text for each point
                                        line=dict(
                                            color=color,
                                        ),
                                    )
                                    legend_data.append(trace)
                    if plot_type == 'plot_mean':
                        # Find the length of the longest list
                        max_length = max(map(len, all_traces))

                        # Put all traces in a single structure
                        all_padded_traces = np.full((len(all_traces), max_length), np.nan)

                        # Fill the array with the values from the time traces
                        for i, sublist in enumerate(all_traces):
                            all_padded_traces[i, :len(sublist)] = sublist

                        # Mean trace
                        y = np.mean(all_padded_traces, axis=0)

                        unique_label = f'{brain_area}_{cell_type}_{specie}'
                        hover_text = [f'{y_val:.2f} {unique_label}' for y_val in y]
                        color = color_map[(specie, brain_area, cell_type)]

                        trace = go.Scatter(
                            x=x,
                            y=y,
                            mode='lines',
                            name=unique_label,
                            hoverinfo='text',  # Show hover text
                            text=hover_text,  # Hover text for each point
                            line=dict(
                                color=color,  # Set color
                            ),
                        )

                        legend_data.extend([trace])
                    # Create layout
                    if plot_type in ['plot_all_traces', 'plot_mean']:
                        title = f'{protocol}'
                    else:
                        title = f'{protocol}, {brain_area}, {cell_type}, {specie}'
                    layout = go.Layout(
                        title=title,
                        xaxis=dict(title='t[s]'),
                        yaxis=dict(title='Voltage[mV]'),
                    )
                    # Create figure
                    fig = go.Figure(data=legend_data, layout=layout)

                    # Save the figure as an HTML file
                    if plot_type == 'plot_all_traces':
                        figure_name = f'{protocol}_raw_traces_first_AP'
                    elif plot_type == 'plot_mean':
                        figure_name = f'{protocol}_mean_raw_traces_first_AP'
                    else:
                        figure_name = f'{protocol}_{brain_area}_{cell_type}_{specie}_raw_traces_first_AP'
                    if repetition_index:
                        figure_name = f'{figure_name}_repetition'
                        for idx in repetition_index:
                            figure_name = f'{figure_name}_{idx}'
                    figure_name = figure_name + '.html'
                    figure_path = os.path.join(output_path, figure_name)
                    fig.write_html(figure_path)


def get_start_stimulus_time(acell, repetition, sweep):
    # time in ms
    if isinstance(acell['repetition'], dict):
        time = acell['repetition']['AP_begin_time'][0]
    elif isinstance(acell['repetition'][repetition]['AP_begin_time'][sweep], list):
        time = acell['repetition'][repetition]['AP_begin_time'][sweep][0]
    else:
        time = acell['repetition'][repetition]['AP_begin_time'][sweep]
    return time / 1000


def get_second_ap_indice(acell, repetition, sweep):
    if isinstance(acell['repetition'], dict):
        indices = acell['repetition']['peak_indices'][sweep]
    else:
        indices = acell['repetition'][repetition]['peak_indices'][sweep]
    if isinstance(indices, list) and len(indices) >= 2:
        idx = min([indices[1], indices[0] + 200])  # empirical value
    elif isinstance(indices, int):
        idx = indices + 200
    else:
        idx = -1
    return idx


def get_stimuli(acells, df, threshold_key, protocol):
    cell_names = []
    repetitions = []
    sweeps = []
    stimuli = []

    for cell_name, acell in acells.items():
        acell_data = [element for element in acell['protocol'] if element.get('name') == protocol]
        if acell_data:
            acell_data = acell_data[0]
        else:
            continue

        if 'repetition' in acell_data:
            if isinstance(acell_data['repetition'], list):
                for rep_idx, repetition in enumerate(acell_data['repetition']):
                    indices = get_trace_indices(protocol_to_plot=protocol, threshold_key=threshold_key,
                                                repetition=repetition)
                    if any(element is not None for element in indices):
                        stim = [repetition['stim'][index] for index in indices]
                        cell_names.append('c' + cell_name[2:])
                        repetitions.append(rep_idx)
                        sweeps.append(int(indices[0]))
                        if len(indices) > 1:
                            raise ValueError('protocol not implemented, check how the stimulus is extracted')
                        stimuli.append(stim[0])
            else:
                repetition = acell_data['repetition']
                indices = get_trace_indices(protocol_to_plot=protocol, threshold_key=threshold_key,
                                            repetition=repetition)
                if any(element is not None for element in indices):
                    stim = [repetition['stim'][index] for index in indices]
                    cell_names.append('c' + cell_name[2:])
                    repetitions.append(0)
                    sweeps.append(int(indices[0]))
                    if len(indices) > 1:
                        raise ValueError('protocol not implemented, check how the stimulus is extracted')
                    stimuli.append(stim[0])

    # Create a new DataFrame with the collected data
    result_df = pd.DataFrame({
        'cell_names': cell_names,
        'repetition': repetitions,
        'sweep': sweeps,
        'Threshold stimulus': stimuli
    })

    # Merge with the original DataFrame on 'cell_names'
    result_df = pd.merge(df[['cell_names', 'BrainArea', 'CellTypeGroup', 'Species']], result_df, on='cell_names',
                         how='left')

    return result_df


def check_inputs(protocol: str, plot_type: str):
    protocol_list = ['IDRest', 'APWaveform']
    if protocol not in protocol_list:
        raise ValueError(f'Select the protocol_list in the available list {protocol_list}')
    plot_list = ['plot_mean', 'plot_all_traces', 'plot_single_groups']
    if plot_type not in plot_list:
        raise ValueError(f'Select the plot_type in the available list {plot_list}')


def get_paths(current_os: str, plot_type: str):
    if current_os == 'Windows':
        rcell_path = 'C:\\Projects\\ASSProject\\Data\\matData'
        acells_path = 'C:\\Projects\\ASSProject\\Analysis\\Data\\jsonData'
        db_path = 'C:\\Projects\\ASSProject\\Analysis\\CellList30-May-2022.csv'
        if plot_type != 'plot_single_groups':
            output_path = 'C:\\Projects\\ASSProject\\Analysis\\Images\\raw_traces_first_AP_comparisons'
        else:
            output_path = 'C:\\Projects\\ASSProject\\Analysis\\Images\\raw_traces_first_AP'
    else:
        raise ValueError('to be implemented')

    return rcell_path, acells_path, db_path, output_path


if __name__ == "__main__":
    protocol = 'IDRest'
    plot_type = 'plot_single_groups'  # ['plot_mean', 'plot_all_traces', 'plot_single_groups']
    # get operating system
    current_os = platform.system()
    rcell_path, acells_path, db_path, output_path = get_paths(current_os, plot_type)

    # Change here if you want to plot a specific repetition, index starts at 0.
    repetition_index = [0]

    start_time = time.time()
    plot_raw_traces(rcell_path=rcell_path, acells_path=acells_path, db_path=db_path, output_path=output_path,
                    protocol=protocol, plot_type=plot_type, repetition_index=repetition_index)
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the result
    print(f"Script execution time: {elapsed_time:.2f} seconds")
