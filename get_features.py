def get_features(protocol_name: str):
    feature_names = []
    match protocol_name:
        case 'APWaveform':
            feature_names = ['stim', 'AP_begin_voltage', 'AP_half_width', 'AP_amplitude', 'min_AHP_voltage',
                             'ISI_values']
        case 'DeHyperPol':
            feature_names = ['stim', 'max_sag_hyper', 'sag_steady_state', 'hyper_sag_tau', 'hyper_sag_recovery_tau']
        case 'FirePattern':
            feature_names = ['stim', 'ISI_values', 'firing_rate']
        case 'HyperDePol':
            feature_names = ['stim', 'AP_begin_voltage', 'AP_width', 'AP_amplitude', 'min_AHP_voltage']
        case 'IDRest':
            feature_names = ['stim', 'AP_begin_voltage', 'AP_half_width', 'AP_amplitude', 'min_AHP_voltage',
                             'ISI_values']
        case 'IV':
            feature_names = ['stim', 'IV_peak_m', 'IV_steady_m']
        case 'PosCheops':
            feature_names = ['AP_begin_voltage']
        case 'sAHP':
            feature_names = ['stim', 'min_AHP_voltage']
    return feature_names
