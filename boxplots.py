import plotly.graph_objs as go
import plotly.offline as pyo
from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing


if __name__ == "__main__":
    all_cells = get_all_cells('/home/lucas/BBP/Data/jsonData')
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')
    feature_names = ['AP_begin_voltage', 'AP_amplitude', 'AP_width']
    threshold_detector = 'spikecount'
    protocols_to_plot = ['APWaveform', 'DeHyperPol']
    data_to_plot = data_preprocessing(path=path, all_cells=all_cells, feature_names=feature_names,
                                      threshold_detector=threshold_detector, protocols_to_plot=protocols_to_plot)

    for protocol_name in protocols_to_plot:
        for feature_name in feature_names:
            # data = data_to_plot[data_to_plot['protocol'] == protocol_name]
            data = data_to_plot[data_to_plot['protocol'] == protocol_name].sort_values(by='CellTypeGroup')

            if feature_name in data:
                # Create a box plot with seaborn
                # Create a box plot
                trace = go.Box(x=data[data['protocol'] == protocol_name]['Group'],
                               y=data[data['protocol'] == protocol_name][feature_name],
                               name='Box Plot',
                               jitter=0.3,
                               hoverinfo='text',
                               text=data[data['protocol'] == protocol_name]['CellName'])
                data = [trace]

                layout = go.Layout(title=(f'Protocol : {protocol_name}, feature : {feature_name}'
                                          f', grouped by Species, Brain Area and Cell Type'),
                                   yaxis=dict(title=feature_name))
                fig = go.Figure(data=data, layout=layout)

                pyo.plot(fig,
                         filename=os.path.join(os.getcwd(),
                                               'Images', protocol_name + '_' + feature_name + '_boxplot.html'),
                         auto_open=True)

