def get_trace_indices(protocol_to_plot, threshold_key, repetition=None):
    if protocol_to_plot in ['sAHP', 'DeHyperPol']:
        trace_indices = [idx for idx, n_spikes in enumerate(repetition[threshold_key]) if n_spikes > 0]
    elif protocol_to_plot in ['FirePattern', 'IV']:
        trace_indices = [-1]
    elif protocol_to_plot in ['PosCheops']:
        trace_indices = [0]
    else:
        # Find the index of the first element greater than 0
        first_occurrence = next((idx for idx, elem in enumerate(repetition[threshold_key]) if elem > 0), None)

        # Get all elements after the first occurrence (if it exists)
        trace_indices = list(range(first_occurrence, len(repetition[threshold_key]))) if first_occurrence is not None else []


    return trace_indices
