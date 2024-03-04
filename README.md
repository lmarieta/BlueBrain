CellCount.py: Create the table counting all cells with the species, brain area and cell type. load_data function also useful to read the cellcount.csv file
CellList30-May-2022.csv: List of cells included in the analysis and model building
MLPDroupout.py: Class declaration for MLPDropout used for neural-network model in ML_models.py
ML_models.py: Can modelize the cells with different models
acellsID.txt: Another list of cells (but only cell names) to facilitate retrieval of information
aecode_data.csv: curated data to be used in ML_models in order to not have to read too many files (time-consuming task). To be generated again in case of additional features (data.to_csv() after data_processing.py)
boxplots.py: Useful to generate the boxplots of the lnmcdata website
error_log.txt: Raw data which could not be loaded for raw data vizualisation on lnmcdata website
get_ap_index.py, get_features.py, get_protocols.py, get_trace_indices.py: modify these functions if you want to analyze different AP, set of features, change the protocol parameter and which trace is extracted for analysis
mann_whitney.py: statistical testing and heatmaps from the lnmc website
outliers.txt: outliers curated by Maurizio and excluded from the analysis
plot_raw_traces.py: Used to plot single raw data action potentials (for example the first)
preprocessing.py: extract and convert data from acell json files to a dataframe
protocol_comparison.py: compare protocols using statistical testing from mann_whitnney.py
read_cell.py: read json files, useful for preprocessing.py
read_mat_rcell.py: read .mat rcell files, useful for plot_raw_traces.py
tau_dI.py: create the dtau as a funciton of dI plots seen on the lnmcdata website (obsolete)
update_db.py: file used to generate .sql file to update the lnmcdata website database
