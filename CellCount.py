import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import os
import matplotlib as mpl


def merge_cells(cell_table, cell_group):
    cells_array = [np.asarray(c) for c in cell_group]
    h = np.array([cells_array[ii + 1][0] - cells_array[ii][0] for ii in range(len(cells_array) - 1)])
    v = np.array([cells_array[ii + 1][1] - cells_array[ii][1] for ii in range(len(cells_array) - 1)])

    # if it's a horizontal merge, all values for `h` are 0
    if not np.any(h):
        # sort by horizontal coord
        cells = np.array(sorted(list(cell_group), key=lambda v: v[1]))
        edges = ['BTL'] + ['BT' for ii in range(len(cells) - 2)] + ['BTR']
    elif not np.any(v):
        cells = np.array(sorted(list(cell_group), key=lambda h: h[0]))
        edges = ['TRL'] + ['RL' for ii in range(len(cells) - 2)] + ['BRL']
    else:
        raise ValueError("Only horizontal and vertical merges allowed")

    for cell, e in zip(cells, edges):
        cell_table[cell[0], cell[1]].visible_edges = e

    txts = [cell_table[cell[0], cell[1]].get_text() for cell in cells]
    tpos = [np.array(t.get_position()) for t in txts]

    # transpose the text of the left cell
    trans = (tpos[-1] - tpos[0]) / 2
    # didn't have to check for ha because I only want ha='center'
    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))
    for txt in txts[1:]:
        txt.set_visible(False)


def load_data(data_path: str):
    # Data pre-processing
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.replace(' ', '')

    df['Species'] = pd.Categorical(df['Species'])
    df['CellType'] = df['CellType'].str.replace(' ', '')
    df['CellType'] = pd.Categorical(df['CellType'])

    # add a column to group brain area
    df['BrainArea'] = df['CellType'].apply(lambda x: 'SNc' if x.startswith('SNc') else x)
    df['BrainArea'] = df['BrainArea'].apply(lambda x: 'DA' if x.startswith('DA') else x)
    df['BrainArea'] = df['BrainArea'].apply(lambda x: 'CA' if x.startswith('CA') else x)
    df['BrainArea'] = df['BrainArea'].apply(
        lambda x:
        'Cortex'
        if
        (x.startswith('L5') or
         x.startswith('L2'))
        else x)

    # add a column to group cell types
    # Fast-spiking are considered interneurons and amygdala are considered as pyramidal cells
    print('In this analysis, amygdala cells are all considered pyramidal cells, '
          'fast-spiking cells are considered inter-neurons. In cortex we count separately L2 and L5 pyramidal cells.')

    df['CellTypeGroup'] = df['CellType'].apply(
        lambda x:
        'PC-L5' if (x == 'L5PC')
        else 'PC-L2' if (x == 'L2PC')
        else 'PC' if (x.endswith('PC') or x == 'Amygdala')
        else x)
    df['CellTypeGroup'] = df['CellTypeGroup'].apply(
        lambda x: 'IN' if (x.endswith('IN') or x.endswith('FS')) else x)

    # Exclude from the analysis NO, DA and SNc cells
    # NO cells are excluded from the analysis because NO means no data or cell died during experiment
    # We are not interested in SNc and DA cells at time of writing
    print('NO, SNc and DA cells are excluded from the analysis')
    df = df[(df['CellTypeGroup'].str.startswith('PC')) | (df['CellTypeGroup'] == 'IN')]

    return df


if __name__ == "__main__":
    # Suppress FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Path to data file
    path = os.getcwd() + '/CellList30-May-2022.csv'

    count_df = load_data(path)
    # Group by 'Species' and 'BrainArea' and compute the count
    count_df = count_df.groupby(['Species', 'BrainArea', 'CellTypeGroup']).size().reset_index(name='Count')

    # Create a pivot table based on the 'BrainArea' column
    cell_type_table = count_df.pivot_table(index='Species',
                                           columns=['BrainArea', 'CellTypeGroup'],
                                           values='Count')

    # Remove empty columns)
    columns_to_keep = ~cell_type_table.eq(0).all()
    filtered_cell_type_table = cell_type_table.loc[:, columns_to_keep]

    # Calculate the column-wise sum
    column_totals = filtered_cell_type_table.sum()
    row_totals = filtered_cell_type_table.sum(axis=1)

    # Create a new DataFrame for the total row
    total_row = pd.DataFrame(column_totals).T.astype(int)
    total_sum = total_row.sum(axis=1).astype(int)
    total_row = ['Total'] + [''] + total_row.values.tolist()[0] + total_sum.values.tolist()
    total_column = pd.DataFrame(row_totals, columns=['TotalColumn']).astype(int)
    total_column = [['Total'], ['']] + [[val] for val in total_column['TotalColumn'].values.tolist()]

    # Prepare labels and column names
    index_name = filtered_cell_type_table.index.name
    num_cols = len(filtered_cell_type_table.columns.names)
    size = len(filtered_cell_type_table.index.values) + num_cols
    row_labels_name = ['' for _ in range(size)]
    row_labels_name[-2] = index_name

    row_labels = filtered_cell_type_table.index.to_list()
    for i in range(0, num_cols):
        row_labels.insert(0, filtered_cell_type_table.columns.names[num_cols - i - 1])

    row_labels_columns = [[row_labels_name[i]] + row_labels[i:i + 1] for i in range(len(row_labels_name))]

    header_rows = [
        [col[0] for i, col in enumerate(filtered_cell_type_table.columns)],
        [col[1] for i, col in enumerate(filtered_cell_type_table.columns)],
    ]

    # Display table
    fig, ax = plt.subplots(figsize=(5, 2), dpi=300)
    ax.axis('tight')
    ax.axis('off')
    celltext = np.vstack([np.hstack([row_labels_columns,
                                     np.vstack([header_rows, filtered_cell_type_table.values.astype(int)]),
                                     total_column]),
                          total_row])
    table = ax.table(cellText=celltext,
                     cellLoc='center',
                     loc='center'
                     )

    # Merge cells
    merge_cells(table, [(0, 0), (1, 0)])
    merge_cells(table, [(3, 0), (2, 0)])
    merge_cells(table, [(0, 4), (0, 5), (0, 6)])
    merge_cells(table, [(0, 7), (1, 7)])
    merge_cells(table, [(4, 0), (4, 1)])

    # plt.show()
    plt.savefig(os.getcwd() + '/Images/Cell_Count.jpg')
