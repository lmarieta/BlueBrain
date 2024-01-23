from mann_whitney import mann_whitney, generate_heatmap
from read_cell import get_all_cells
import os
from get_features import get_features
from get_ap_index import get_ap_index
from preprocessing import data_preprocessing


def features_to_compare(protocol1, protocol2):
    pair_list = []
    match protocol1:
        case 'APWaveform':
            match protocol2:
                case 'PosCheops':
                    pair_list = [('AP_begin_voltage', 0, 'volt_for_depol_AP1', 0),
                                 ('AP_begin_voltage', 0, 'volt_for_depol_AP1', 1),
                                 ('AP_begin_voltage', 0, 'volt_for_depol_AP1', 2)]
                case 'IDRest':
                    pair_list = [('AP_width', 0, 'AP_width', 0),
                                 ('AP_amplitude', 0, 'AP_amplitude', 0)]

    return pair_list

if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    threshold_detector = 'spikecount'
    protocol_1 = 'APWaveform'
    protocol_2 = 'IDRest'
    feature_names_1 = get_features(protocol_1)
    feature_names_2 = get_features(protocol_2)

    cell_data_1 = data_preprocessing(path=path, all_cells=all_cells,
                                     feature_names=feature_names_1,
                                     threshold_detector=threshold_detector,
                                     protocol_to_plot=protocol_1,
                                     rm_in_cells=True)
    cell_data_2 = data_preprocessing(path=path, all_cells=all_cells,
                                     feature_names=feature_names_2,
                                     threshold_detector=threshold_detector,
                                     protocol_to_plot=protocol_2,
                                     rm_in_cells=True)

    cell_data_1 = cell_data_1[cell_data_1['protocol'] == protocol_1]
    cell_data_2 = cell_data_2[cell_data_2['protocol'] == protocol_2]

    ap_index_1 = get_ap_index(protocol_1)
    ap_index_2 = get_ap_index(protocol_2)

    # Get unique combinations of 'Species' and 'BrainArea'
    combinations_list_1 = (cell_data_1[['Species', 'BrainArea', 'CellTypeGroup']].drop_duplicates().values.
                           tolist())
    # Get unique combinations of 'Species' and 'BrainArea'
    combinations_list_2 = (cell_data_2[['Species', 'BrainArea', 'CellTypeGroup']].drop_duplicates().values.
                           tolist())

    for x in features_to_compare(protocol_1, protocol_2):
        data_dict_1 = {'combos': combinations_list_1, 'data': cell_data_1, 'feature': x[0], 'ap_index': x[1],
                       'protocol': protocol_1}
        data_dict_2 = {'combos': combinations_list_2, 'data': cell_data_2, 'feature': x[2], 'ap_index': x[3],
                       'protocol': protocol_2}

        mann_whitney_results = mann_whitney(data_dict_1=data_dict_1, data_dict_2=data_dict_2)

        ax = generate_heatmap(data=mann_whitney_results)
        # ax.show()
        ax.savefig('Images/p_values_' + protocol_1 + '_' + x[0] + '_AP_index_' + str(x[1]) + '_' +
                   protocol_2 + '_' + x[2] + '_AP_index_' + str(x[3]))
