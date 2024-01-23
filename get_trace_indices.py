def get_trace_indices(protocol_to_plot, threshold_key='spikecount', repetition=None):
    if protocol_to_plot in ['sAHP', 'DeHyperPol']:
        trace_indices = [idx for idx, n_spikes in enumerate(repetition[threshold_key]) if n_spikes > 0]
    elif protocol_to_plot in ['FirePattern', 'IV']:
        trace_indices = [-1]
    elif protocol_to_plot in ['PosCheops']:
        trace_indices = [0]
    else:
        trace_indices = [next((idx for idx, elem in enumerate(repetition[threshold_key]) if elem > 0),
                              None)]
    return trace_indices
