from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing
import plotly.graph_objs as go


if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    threshold_detector = 'spikecount'
    protocol = 'DeHyperPol'
    feature_names = ['stim', 'hyper_sag_tau', 'hyper_sag_recovery_tau']

    data = data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names,
                              threshold_detector=threshold_detector, protocol_to_plot=protocol, rm_in_cells=True)
    # Data preparation

    trace_index = list(set(data['trace_index']))

    data = data[data['protocol'] == protocol]

    # Get unique trace_index values
    unique_trace_indices = data['trace_index'].unique()

    for feature_name in feature_names:
        if feature_name is not 'stim':
            # Create an empty list to store the box trace for each trace_index
            box_traces = []
            for i in unique_trace_indices:
                    trace = go.Box(
                        x=data[data['trace_index'] == i]['stim'],
                        y=data[data['trace_index'] == i][feature_name],
                        name=f'Trace Index {i}',
                        jitter=0.3,
                        hoverinfo='text',
                        text=data[data['trace_index'] == i]['CellName'] + ' sweep: ' + data[data['trace_index'] == i]['repetition'].astype(str)
                    )
                    box_traces.append(trace)

            layout = go.Layout(title=(f'Protocol : {protocol}, feature : {feature_name}'
                                      f', grouped by Species, Brain Area and Cell Type'),
                               yaxis=dict(title=feature_name, type='log'), xaxis=dict(title=feature_names[0]))
            fig = go.Figure(data=box_traces, layout=layout)
            fig.write_html('Images/dtau_dI_' + feature_name + '.html')
