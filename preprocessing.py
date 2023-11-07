import pandas as pd
from CellCount import load_data
import numpy as np
from read_cell import get_all_cells
import os.path


def extract_values_above_threshold(protocol, feature_name, threshold_key):
    repetition_data = protocol['repetition']
    feature_values = []
    if isinstance(repetition_data, list):
        for repetition_entry in repetition_data:
            spikecount = repetition_entry[threshold_key]
            value_above_threshold = repetition_entry[feature_name]
            if value_above_threshold:
                feature_values.append(value_above_threshold[next(i for i, x in enumerate(spikecount) if x > 0)])
        return feature_values
    elif isinstance(repetition_data, dict):
        spikecount = repetition_data[threshold_key]
        value_above_threshold = repetition_data[feature_name]
        feature_values.append(value_above_threshold[next(i for i, x in enumerate(spikecount) if x > 0)])
        return feature_values
    return None  # If no valid entry is found


def data_preprocessing(path: str, all_cells: dict, feature_names: list, threshold_detector: str, protocols_to_plot:list):
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
        # Iterate over all protocols
        protocols = [protocol for protocol in cell_data['protocol'] if protocol['name'] in protocols_to_plot]
        for protocol in protocols:
            # Iterate over all features for a given protocol and cell
            if 'repetition' in protocol:
                feature_values = []
                new_row = {
                    'CellName': cell_name,
                    'Species': cell_data['cellInfo']['species'],
                    'BrainArea': cell_data['cellInfo']['BrainArea'],
                    'CellTypeGroup': cell_data['cellInfo']['CellTypeGroup'],
                    'protocol': protocol['name']
                }
                for feature_name in feature_names:
                    values_above_threshold = extract_values_above_threshold(protocol, feature_name, threshold_detector)
                    # feature_values.extend(values_above_threshold)
                    # Flatten feature_values
                    feature_values = [item for sublist in values_above_threshold
                                      for item in (sublist if (isinstance(sublist, list)) else [sublist])]
                    new_row[feature_name] = feature_values

                # Find the maximum length of lists
                max_length = max(len(values) for key, values in new_row.items() if isinstance(values, list))

                # Pad lists with NaN to match the maximum length
                for key, values in new_row.items():
                    if key not in feature_names:
                        new_row[key] = [values] * max_length
                        print(values)
                    if isinstance(values, list) and len(values) < max_length:
                        values += [np.nan] * (max_length - len(values))

                data_to_plot = pd.concat([data_to_plot, pd.DataFrame(new_row)], ignore_index=True)

    # Remove aberrant and placeholder values
    if 'AP_width' in data_to_plot['protocol']:
        max_index = data_to_plot['AP_width'].idxmin()
        second_min_index = data_to_plot['AP_width'].nsmallest(2).index[-1]
        data_to_plot.at[max_index, 'AP_width'] = np.nan
        data_to_plot.at[second_min_index, 'AP_width'] = np.nan
        data_to_plot['AP_width'].replace(-32, np.nan, inplace=True)
        print('Removed outliers and -32 values from AP_width')

    # Create a "group" column that combines the three factors
    data_to_plot['Group'] = data_to_plot['Species'] + ' ' + data_to_plot['BrainArea'] + ' ' + data_to_plot[
        'CellTypeGroup']

    return data_to_plot

if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    feature_names = ['AP_begin_voltage', 'AP_amplitude', 'AP_width']
    threshold_detector = 'spikecount'
    protocols = ['APWaveform']
    data = data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names,
                                      threshold_detector=threshold_detector, protocols_to_plot=protocols)

    print(data)
