from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
from pandas.core.frame import DataFrame
from scipy.stats import shapiro
from scipy.stats import levene
from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing
import statsmodels.api as sm
import matplotlib.pyplot as plt


def check_normality(data):
    _, p = shapiro(data)
    if p > 0.05:
        print("Data appears to be normally distributed.")
    else:
        print("Data may not be normally distributed.")


def check_homoscedasticity(residuals, data, group):
    # Get unique levels of the 'group' factor
    levels = data[group].unique()

    # Initialize an empty list to store the data for each level
    group_data = []

    for level in levels:
        group_data.append(residuals[data[group] == level])

    # Perform Levene's test on the list of data for each level
    test_statistic, p = levene(*group_data)

    if p > 0.05:
        print(f"Equal variances are assumed (homoscedasticity) for {group}.")
    else:
        print(f"Equal variances are not assumed (heteroscedasticity) for {group}.")


def twoway_anova(dependent_variables: DataFrame, factors: DataFrame) -> DataFrame:
    # Replace 'dependent_variable', 'factor1', 'factor2', and 'factor3' with your actual column names.
    data = pd.concat([dependent_variables, factors], axis=1, ignore_index=True, join='outer')
    columns = dependent_variables.columns.tolist()
    columns.extend(factors.columns.tolist())
    data.columns = columns
    formula = (columns[0] + ' ~ C(' + columns[1] + ') + C(' + columns[2] + ') + C('
               + columns[1] + '):C(' + columns[2] + ')')

    # Perform the three-way ANOVA using statsmodels
    model = ols(formula, data=data).fit()
    anova_table = anova_lm(model, typ=3)  # Use type 3 in order to consider interactions

    # Create a Q-Q plot
    sm.qqplot(data[columns[0]], fit=True, line='45')

    # Add labels and title
    plt.title("Q-Q Plot")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")

    # Show the plot
    plt.show()

    # Residuals
    residuals = model.resid

    check_normality(residuals)
    check_homoscedasticity(residuals, data, columns[1])
    check_homoscedasticity(residuals, data, columns[2])

    return anova_table


if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    feature_names = ['AP_begin_voltage']  # [, 'AP_amplitude', 'AP_width']
    threshold_detector = 'spikecount'
    anova_protocols = ['APWaveform', 'DeHyperPol']
    anova_data = data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names,
                                    threshold_detector=threshold_detector, protocols_to_plot=anova_protocols)
    # Data preparation for ANOVA
    # Exclude IN cells and merge PC-L2 with PC-L5
    anova_data = anova_data[anova_data['CellTypeGroup'] != 'IN']
    print('Remove IN cells for anova study')
    print('Groupe PC cells together for anova study\n')

    # Call your twoway_anova function
    for protocol in anova_protocols:
        for feature in feature_names:
            data = anova_data[anova_data['protocol'] == protocol]

            result = twoway_anova(data[[feature]], data[['Species', 'BrainArea']])

            # Print the ANOVA results
            print(result)
