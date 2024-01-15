from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing
from get_features import get_features
from get_protocols import get_protocols


def merge_data(json_path='/home/lucas/BBP/Data/jsonData',
               celllist_path=os.path.join(os.getcwd(), 'CellList30-May-2022.csv'),
               threshold_detector='spikecount',
               protocols=get_protocols()):
    all_cells = get_all_cells(json_path)
    for protocol in protocols:
        feature_names = get_features(protocol)
        df = data_preprocessing(path=celllist_path, all_cells=all_cells, feature_names=feature_names,
                                threshold_detector=threshold_detector, protocol_to_plot=protocol)
        protocol_path = os.path.join(os.getcwd(), 'azure/df_' + protocol + '.csv')
        df.to_csv(protocol_path, index=False)


if __name__ == "__main__":
    merge_data()
