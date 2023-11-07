# Kruskal-Wallis check if the means are different between species and between brain areas. It does not rely on the
# assumption of normality or homoscedasticity.

from scipy.stats import kruskal
from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing
from itertools import combinations
from pandas import DataFrame



def kruskal_wallis(combinations_list: list, kruskal_data: DataFrame, feature_name: str) -> dict:
    # Dictionary to store Kruskal-Wallis results for each combination
    kruskal_results = {}

    # Compare each combination with every other combination
    for combo1, combo2 in combinations(combinations_list, 2):
        species1, brain_area1 = combo1
        species2, brain_area2 = combo2

        # Extract data for each combination
        data1 = kruskal_data[(kruskal_data['Species'] == species1) &
                             (kruskal_data['BrainArea'] == brain_area1)][feature_name]
        data2 = kruskal_data[(kruskal_data['Species'] == species2) &
                             (kruskal_data['BrainArea'] == brain_area2)][feature_name]

        # Perform Kruskal-Wallis test on the two combinations
        kruskal_stat, p_value = kruskal(data1, data2)

        kruskal_results[(tuple(combo1), tuple(combo2))] = {
            'Kruskal Statistic': kruskal_stat,
            'p-value': p_value
        }

    # Display Kruskal-Wallis results for each combination comparison
    for (combo1, combo2), result in kruskal_results.items():
        print(
            f"Comparison: {combo1} vs {combo2}, Kruskal Statistic: {result['Kruskal Statistic']}, p-value: "
            f"{result['p-value']}")

    return kruskal_results


if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    feature_names = ['AP_begin_voltage']  # [, 'AP_amplitude', 'AP_width']
    threshold_detector = 'spikecount'
    kruskal_protocols = ['APWaveform']
    kruskal_data = data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names,
                                    threshold_detector=threshold_detector, protocols_to_plot=kruskal_protocols)
    # Data preparation for ANOVA
    # Exclude IN cells and merge PC-L2 with PC-L5
    kruskal_data = kruskal_data[kruskal_data['CellTypeGroup'] != 'IN']
    print('Remove IN cells for kruskal-wallis test')
    print('Groupe PC cells together for kruskal-wallis test\n')

    # Get unique combinations of 'Species' and 'BrainArea'
    combinations_list = kruskal_data[['Species', 'BrainArea']].drop_duplicates().values.tolist()

    # Call Kruskal-Wallis
    for feature_name in feature_names:
        kruskal_wallis(combinations_list=combinations_list, kruskal_data=kruskal_data, feature_name=feature_name)

