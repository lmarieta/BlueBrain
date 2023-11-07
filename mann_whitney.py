# Mann-Whitney U test to check if the means are different between species and between brain areas. It does not rely on the
# assumption of normality or homoscedasticity.

from scipy.stats import mannwhitneyu
from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing
from itertools import combinations
from pandas import DataFrame


def mannwhitney(combinations_list: list, mannwhitney_data: DataFrame, feature_name: str) -> dict:
    # Dictionary to store Mann-Whitney results for each combination
    mannwhitney_results = {}

    # Compare each combination with every other combination
    for combo1, combo2 in combinations(combinations_list, 2):
        species1, brain_area1 = combo1
        species2, brain_area2 = combo2

        # Extract data for each combination
        data1 = mannwhitney_data[(mannwhitney_data['Species'] == species1) &
                             (mannwhitney_data['BrainArea'] == brain_area1)][feature_name]
        data2 = mannwhitney_data[(mannwhitney_data['Species'] == species2) &
                             (mannwhitney_data['BrainArea'] == brain_area2)][feature_name]

        # Perform Mann-Whitney test on the two combinations
        mann_whitney_stat, p_value = mannwhitneyu(data1, data2)

        mannwhitney_results[(tuple(combo1), tuple(combo2))] = {
            'Mann-Whitney-U Statistic': mann_whitney_stat,
            'p-value': p_value
        }

    # Display Mann-Whitney results for each combination comparison
    for (combo1, combo2), result in mannwhitney_results.items():
        print(
            f"Comparison: {combo1} vs {combo2}, Mann-Whitney-U Statistic: {result['Mann-Whitney-U Statistic']}, "
            f"p-value: {result['p-value']}")

    return mannwhitney_results


if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    feature_names = ['AP_begin_voltage']  # [, 'AP_amplitude', 'AP_width']
    threshold_detector = 'spikecount'
    mannwhitney_protocols = ['APWaveform']
    mannwhitney_data = data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names,
                                      threshold_detector=threshold_detector, protocols_to_plot=mannwhitney_protocols)

    # Data preparation for ANOVA
    # Exclude IN cells and merge PC-L2 with PC-L5
    mannwhitney_data = mannwhitney_data[mannwhitney_data['CellTypeGroup'] != 'IN']
    print('Remove IN cells for mann-whitney test')
    print('Groupe PC cells together for mann-whitney test\n')

    # Get unique combinations of 'Species' and 'BrainArea'
    combinations_list = mannwhitney_data[['Species', 'BrainArea']].drop_duplicates().values.tolist()

    # Perform Mann-Whitney U test
    for feature_name in feature_names:
        statistic = mannwhitney(combinations_list=combinations_list, mannwhitney_data=mannwhitney_data,
                                feature_name=feature_name)
