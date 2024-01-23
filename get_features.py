def get_features(protocol_name: str):
    feature_names = []
    match protocol_name:
        case 'APWaveform':
            feature_names = ['stim', 'AP_begin_voltage', 'AP_width', 'AP_amplitude', 'min_AHP_voltage']
        case 'DeHyperPol':
            feature_names = ['stim', 'max_sag_hyper', 'sag_steady_state', 'hyper_sag_tau', 'hyper_sag_recovery_tau']
        case 'FirePattern':
            feature_names = ['stim', 'ISI_values', 'firing_rate']
        case 'HyperDePol':
            feature_names = ['stim', 'AP_begin_voltage', 'AP_width', 'AP_amplitude', 'min_AHP_voltage']
        case 'IDRest':
            feature_names = ['stim', 'AP_begin_voltage', 'AP_width', 'AP_amplitude', 'min_AHP_voltage']
        case 'IV':
            feature_names = ['stim', 'IV_peak_m', 'IV_peak_c', 'IV_steady_m', 'IV_steady_c']
        case 'PosCheops':
            feature_names = ['volt_for_depol_AP1']
        case 'sAHP':
            feature_names = ['stim', 'min_AHP_voltage']
    return feature_names
