import pandas as pd
from CellCount import load_data
import numpy as np
from read_cell import get_all_cells
import os.path
from pandas import DataFrame
import warnings
from get_features import get_features
from get_trace_indices import get_trace_indices
from get_ap_index import get_ap_index


def condition_ap_index(values):
    length_values = len(values)
    return (length_values > 0) and any(values)


def extract_values_from_trace(repetition, protocol_name, feature_name, trace_index, ap_index):
    feature_values = []
    if protocol_name == 'FirePattern':
        # Firing rate is not (yet) an ecode feature and must be computed from spikecount and peak_time
        if feature_name == 'peak_time':
            warnings.warn('Peak time not implemented yet since not a single value per trace.')
            return None
        if (feature_name == 'firing_rate') and (len(repetition['ISI_values']) > 0):
            last_trace = repetition['ISI_values'][-1]
            value = np.median(last_trace)
            value = 1/value
            feature_values.append(value)
        if feature_name == 'ISI_values' and len(repetition[feature_name]) > 0:
            last_trace = repetition[feature_name][-1]
            value = np.median(last_trace)
            feature_values.append(value)

    elif protocol_name == 'IV':
        value = repetition[feature_name]
        feature_values.append(value)

    elif protocol_name == 'PosCheops':
        trace = []
        if len(repetition['cheops']) > 0:
            trace = repetition['cheops'][ap_index]
        if trace:
            value = trace[feature_name]
            feature_values.append(value)

    else:
        trace = []
        if len(repetition[feature_name]) > 0 and trace_index:
            trace = repetition[feature_name][trace_index]

        if trace:
            if isinstance(trace, list):
                value = repetition[feature_name][trace_index][ap_index]
            else:
                value = repetition[feature_name][trace_index]
            feature_values.append(value)

    return feature_values


def add_feature_value_to_data(value, feature_name, updated_row, ap_index=np.nan, trace_index=0):
    updated_row['trace_index'] = trace_index
    updated_row['AP_index'] = ap_index
    updated_row[feature_name] = value
    return updated_row


def data_preprocessing(path: str, all_cells: dict, feature_names: list, threshold_detector: str,
                       protocol_to_plot: str, rm_in_cells=False):
    count_df = load_data(path)
    cell_ids_to_keep = count_df['CellID'].tolist()

    # Some cells are in the CellList file but not in acell files
    keys_to_remove = [key for key, value in all_cells.items() if value['id'] not in cell_ids_to_keep]

    # Remove the keys from the dictionary
    for key in keys_to_remove:
        all_cells.pop(key)

    # Add brain area and cell type group in CellInfo to facilitate data analysis
    for index, key in enumerate(all_cells):
        all_cells[key]['cellInfo']['BrainArea'] = (
            count_df.loc[count_df['CellID'] == all_cells[key]['id'], 'BrainArea'].values[0])
        all_cells[key]['cellInfo']['CellTypeGroup'] = (
            count_df.loc[count_df['CellID'] == all_cells[key]['id'], 'CellTypeGroup'].values[0])

    # Create a list to store the data for plotting
    data_to_plot = pd.DataFrame()

    # Iterate over the cell names in all_cells
    for cell_name, cell_data in all_cells.items():
        protocol_index = [protocol for protocol in cell_data['protocol'] if protocol['name'] == protocol_to_plot]
        protocol = next(iter(protocol_index), None)
        if not protocol:
            continue
        ap_indices = get_ap_index(protocol_to_plot)
        # Iterate over all features for a given protocol and cell
        if 'repetition' in protocol:
            new_row = {
                'CellName': cell_name,
                'Species': cell_data['cellInfo']['species'],
                'BrainArea': cell_data['cellInfo']['BrainArea'],
                'CellTypeGroup': cell_data['cellInfo']['CellTypeGroup'],
                'protocol': protocol_to_plot
            }
            for feature_name in feature_names:
                new_row[feature_name] = np.nan

            if isinstance(protocol['repetition'], list):
                repetition_list = protocol['repetition']
                repetition_iterator = enumerate(repetition_list)
            else:  # if isinstance(protocol['repetition'], dict)
                repetition_iterator = [(1, protocol['repetition'])]

            for n_trace, repetition in repetition_iterator:
                trace_indices = get_trace_indices(protocol_to_plot=protocol_to_plot,
                                                  threshold_key=threshold_detector,
                                                  repetition=repetition)
                new_row['repetition'] = n_trace
                for trace_index in trace_indices:
                    for ap_index in ap_indices:
                        updated_row = pd.DataFrame(new_row, index=[0])
                        for feature_name in feature_names:
                            value = extract_values_from_trace(repetition=repetition,
                                                              protocol_name=protocol_to_plot,
                                                              feature_name=feature_name,
                                                              trace_index=trace_index,
                                                              ap_index=ap_index)
                            if condition_ap_index(value):
                                updated_row = add_feature_value_to_data(value=value,
                                                                        feature_name=feature_name,
                                                                        updated_row=updated_row,
                                                                        ap_index=ap_index,
                                                                        trace_index=trace_index)
                        if any(updated_row[feature_names].notna().any()):
                            data_to_plot = pd.concat([data_to_plot, updated_row],
                                                     ignore_index=True)

    # Create a "group" column that combines the three factors
    if 'AP_index' in data_to_plot.keys():
        data_to_plot['Group'] = (data_to_plot['Species'] + ' ' + data_to_plot['BrainArea'] + ' ' +
                                 data_to_plot['CellTypeGroup'] + ' AP index ' + data_to_plot['AP_index'].astype(str))
    else:
        data_to_plot['Group'] = (data_to_plot['Species'] + ' ' + data_to_plot['BrainArea'] + ' ' +
                                 data_to_plot['CellTypeGroup'])

    # Remove aberrant and placeholder values
    if 'AP_width' in feature_names:
        data_to_plot['AP_width'].replace(-32, np.nan, inplace=True)

        print('Replaced two outliers, NaNs and -32 values from AP_width by the median of AP_width within a given '
              'species, brain area, cell type and action potential combination.')

    data_to_plot = data_to_plot.set_index('Group')  # Set 'Group' as the index

    for feature in feature_names:
        grouped = group_median(df=data_to_plot, feature_name=feature)
        data_to_plot[feature] = data_to_plot[feature].fillna(grouped)

    print('Replaced NaNs from all columns by the median of the column within a given species, brain area and cell type '
          'combination.')
    if rm_in_cells:
        # Data preparation
        # Exclude IN cells
        data_to_plot = data_to_plot[data_to_plot['CellTypeGroup'] != 'IN']
        print('Remove IN cells for kruskal-wallis test\n')

    data_to_plot = data_to_plot.reset_index()

    return data_to_plot


def group_median(df: DataFrame, feature_name: str):
    # Filter only numeric columns
    df[feature_name] = pd.to_numeric(df[feature_name], errors='coerce')
    # Group the DataFrame by the 'Group' column
    grouped = df.groupby('Group')
    # Calculate the median for each group while ignoring NaN values
    group_medians = grouped[feature_name].median()
    return group_medians


if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')

    threshold_detector = 'spikecount'

    # ['FirePattern', 'IDRest', 'HyperDePol', 'IV', 'PosCheops', 'APWaveform', 'DeHyperPol', 'sAHP']
    protocol = 'FirePattern'
    feature_names = get_features(protocol)  # firing_rate

    data = data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names,
                              threshold_detector=threshold_detector, protocol_to_plot=protocol)
    pass
