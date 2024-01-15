# Mann-Whitney check if the means are different between species and between brain areas. It does not rely on the
# assumption of normality or homoscedasticity.
import numpy as np
from scipy.stats import mannwhitneyu
from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
from get_features import get_features
from get_ap_index import get_ap_index


def get_short_label(labels: tuple):
    short_label = ''
    for label in labels:
        match label:
            case 'Rat':
                s = 'R'
            case 'Mouse':
                s = 'M'
            case 'Cortex':
                s = 'Cx'
            case 'Amygdala':
                s = 'Amy'
            case _:
                s = label
        short_label = short_label + s
    return short_label


def mann_whitney(data_dict_1: dict, data_dict_2: dict, alpha=0.05) -> dict:
    feature_name_1 = data_dict_1['feature']
    feature_name_2 = data_dict_2['feature']
    ap_index_1 = data_dict_1['ap_index']
    ap_index_2 = data_dict_2['ap_index']
    data_1 = data_dict_1['data']
    data_2 = data_dict_2['data']
    protocol_1 = data_dict_1['protocol']
    protocol_2 = data_dict_2['protocol']

    combinations_list = list(product(data_dict_1['combos'], data_dict_2['combos']))

    # Dictionary to store Kruskal-Wallis results for each combination
    mann_whitney_results = {}

    # Compare each combination with every other combination
    for combo1, combo2 in combinations_list:
        species1, brain_area1, cell_type1 = combo1
        species2, brain_area2, cell_type2 = combo2

        # Extract data for each combination
        data1 = data_1[(data_1['Species'] == species1) &
                       (data_1['BrainArea'] == brain_area1) &
                       (data_1['CellTypeGroup'] == cell_type1) &
                       (data_1['AP_index'] == ap_index_1)][feature_name_1]
        data2 = data_2[(data_2['Species'] == species2) &
                       (data_2['BrainArea'] == brain_area2) &
                       (data_2['CellTypeGroup'] == cell_type2) &
                       (data_2['AP_index'] == ap_index_2)][feature_name_2]

        median_data1 = np.median(data1)
        median_data2 = np.median(data2)

        # Perform Mann-Whitney test on the two combinations
        mann_whitney_stat, p_value = mannwhitneyu(data1, data2)

        # Check if the p-value is significant
        # No correction here, see 'When to use the Bonferroni correction', Richard A. Armstrong,
        # https://doi.org/10.1111/opo.12131
        is_significant = p_value < alpha

        mann_whitney_results[((protocol_1, feature_name_1) + tuple(combo1) + (ap_index_1,),
                              (protocol_2, feature_name_2) + tuple(combo2) + (ap_index_2,))] = {
            'Mann-Whitney Statistic': mann_whitney_stat,
            'p-value': p_value,
            'Is Significant': is_significant,
            'Median 1st group': median_data1,
            'Median 2nd group': median_data2
        }

    # Display Mann-Whitney results for each combination comparison,
    # p smaller than alpha means difference between groups
    for (combo1, combo2), result in mann_whitney_results.items():
        print(
            f"Comparison: {combo1} vs {combo2}, Mann-Whitney Statistic: {result['Mann-Whitney Statistic']}, p-value: "
            f"{result['p-value']}, Significant: {result['Is Significant']}, Median {combo1}: "
            f"{result['Median 1st group']}, Median {combo2}: {result['Median 2nd group']}")

    return mann_whitney_results


def generate_heatmap(data: dict):
    ap_index = [combo[-1] for key in data.keys() for combo in key]
    ap_index1 = ap_index[0::2][0]
    ap_index2 = ap_index[1::2][0]

    # Extract combo labels and p-values
    combo_labels = [combo[2:5] for key in data.keys() for combo in key]
    id_labels = [combo[0:2] for key in data.keys() for combo in key]
    id_labels1 = id_labels[0::2][0]
    id_labels2 = id_labels[1::2][0]
    combo1_labels = combo_labels[::2]
    combo2_labels = combo_labels[1::2]
    unique_labels = list(set(combo_labels))

    # Initialize a 2D array to store the p-values
    p_values = np.full((len(unique_labels), len(unique_labels)), np.nan)
    combo1_map_labels = unique_labels
    combo2_map_labels = unique_labels

    for i, combo1 in enumerate(combo1_map_labels):
        for j, combo2 in enumerate(combo2_map_labels):
            key1 = id_labels1 + combo1 + (ap_index1,)
            key2 = id_labels2 + combo2 + (ap_index2,)
            if (key1, key2) in data.keys():
                p_values[j, i] = data[(key1, key2)]['p-value']

    combo1_map_labels = [get_short_label(labels) for labels in combo1_map_labels]
    combo2_map_labels = [get_short_label(labels) for labels in combo2_map_labels]

    sorted_idx1 = np.argsort(combo1_map_labels)
    sorted_idx2 = np.argsort(combo2_map_labels)

    p_values = p_values[sorted_idx2][:, sorted_idx1]

    threshold = 0.05

    # Create a custom colormap with white for values below the threshold
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    cmap.set_under(color='white', alpha=1)

    # Create a heatmap with the custom colormap
    ax = plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust the font size if needed

    heatmap = sns.heatmap(p_values, annot=True, fmt=".2f", cmap=cmap,
                          xticklabels=sorted(combo2_map_labels),
                          yticklabels=sorted(combo1_map_labels),
                          cbar_kws={'label': 'p-values'}, vmin=threshold)

    # Add labels and title
    plt.xlabel(id_labels1[0] + ' ' + id_labels1[1] + ' AP index: ' + str(ap_index1))
    plt.ylabel(id_labels2[0] + ' ' + id_labels2[1] + ' AP index: ' + str(ap_index2))
    plt.tight_layout()

    return ax


if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    threshold_detector = 'spikecount'
    mannwhitney_protocol = 'FirePattern'
    feature_names = get_features(mannwhitney_protocol)

    mannwhitney_data = data_preprocessing(path=path, all_cells=all_cells,
                                      feature_names=feature_names,
                                      threshold_detector=threshold_detector,
                                      protocol_to_plot=mannwhitney_protocol,
                                      rm_in_cells=True)

    ap_index = get_ap_index(mannwhitney_protocol)

    data = mannwhitney_data[mannwhitney_data['protocol'] == mannwhitney_protocol]

    # Get unique combinations of 'Species' and 'BrainArea'
    combinations_list = (data[['Species', 'BrainArea', 'CellTypeGroup']].drop_duplicates().values.
                         tolist())

    # Call Mann-Whitney
    for feature_name in feature_names:
        for i in ap_index:
            data_dict = {'combos': combinations_list, 'data': data, 'feature': feature_name, 'ap_index': i,
                         'protocol': mannwhitney_protocol}
            mann_whitney_results = mann_whitney(data_dict, data_dict)
            ax = generate_heatmap(data=mann_whitney_results)
            ax.show()
            # ax.savefig('Images/p_values_' + mannwhitney_protocol + '_' + feature_name + '_AP' + str(i))
