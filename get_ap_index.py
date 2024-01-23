def get_ap_index(protocol_name: str):
    ap_index = []
    match protocol_name:
        case 'APWaveform':
            ap_index = [0, 1]
        case 'PosCheops':
            ap_index = [0, 1, 2]
        case 'FirePattern':
            ap_index = [-1]
        case 'IV':
            ap_index = [0]
        case 'IDRest':
            ap_index = [0]
        case 'HyperDePol':
            ap_index = [0]
        case 'DeHyperPol':
            ap_index = [-1]
        case 'sAHP':
            ap_index = [-1]# [list(range(spikes)) for spikes in list_of_spikes]
    return ap_index
