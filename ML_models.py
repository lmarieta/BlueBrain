from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from read_cell import get_all_cells
import os.path
from preprocessing import data_preprocessing
from sklearn.metrics import (f1_score, recall_score, precision_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from MLPDropout import MLPDropout
from get_features import get_features
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import smote_variants as sv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from xgboost import XGBClassifier
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.regularizers import l2


def get_distinct_colors(num_colors):
    # Create a color map from the available colors
    colormap = plt.get_cmap("tab20")

    # Generate a list of distinct colors
    colors = [colormap(i) for i in range(num_colors)]

    return colors


def get_model(model_type, input_shape=(0,), random_state=42, num_classes=7, dropout_rate=0, l2_reg=0,
              num_hidden_layers=1):
    model = None
    match model_type:
        case 'xgb':
            model = XGBClassifier()
        case 'custom_nn':
            n_units = 32

            # Define the neural network model
            model = Sequential()

            # Add an input layer with the appropriate input shape
            model.add(Dense(units=n_units, kernel_regularizer=l2(l2_reg), input_shape=input_shape))
            model.add(BatchNormalization())  # Add Batch Normalization before activation
            model.add(Activation('relu'))
            model.add(Dropout(dropout_rate, seed=random_state))

            # Add one or more hidden layers
            for i in range(num_hidden_layers):
                model.add(Dense(units=n_units, kernel_regularizer=l2(l2_reg)))
                model.add(BatchNormalization())  # Add Batch Normalization before activation
                model.add(Activation('relu'))
                model.add(Dropout(dropout_rate, seed=random_state))

            # Add the output layer (units equal to the number of classes) and softmax activation
            model.add(Dense(units=num_classes, activation='softmax'))

            # Compile the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        case 'knn':
            model = KNeighborsClassifier(n_neighbors=15)
        case 'SVM':
            model = SVC(decision_function_shape='ovr')
        case 'neural_network':
            model = MLPDropout()
        case 'logistic_regression':
            model = LogisticRegression()
        case 'random_forest':
            # Initialize the hyperparameters
            criterion = 'log_loss'  # log_loss because multi-class classification (as opposed to binary)
            max_depth = 10
            bootstrap = False  # Build trees with random sub-sample from dataset if True, otherwise use whole dataset
            class_weight = 'balanced'  # Trying balanced because of imbalanced dataset
            model = RandomForestClassifier(criterion=criterion, max_depth=max_depth,
                                           bootstrap=bootstrap, random_state=random_state,
                                           class_weight=class_weight)

    return model


def param_search_space(model_type):
    # Define the parameter grid to search
    param_dist = {}
    match model_type:
        case 'xgb':
            param_dist = {
                'objective': ['multi:softmax'],
                'num_class': [7],
                'max_depth': [1, 3, 5],
                'learning_rate': [0.01, 0.1, 0.3, 0.5],
                'n_estimators': [100, 200, 500],
                'reg_lambda': [0, 0.01, 0.1, 1, 10, 100],
                'gamma': [0, 1, 5],
            }
        case 'custom_nn':
            param_dist = {
                "optimizer__learning_rate": [0.0001, 0.001, 0.1],
                "dropout_rate": [0, 0.1, 0.3, 0.4],
                "num_hidden_layers": [0, 1, 2, 3, 4],
                "l2_reg": [0, 0.01, 0.1, 1, 10, 100]
            }
        case 'knn':
            param_dist = {
                'weights': ['distance'],
                'p': [1, 2],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['minkowski', 'manhattan', 'euclidean'],
            }
        case 'SVM':
            param_dist = {
                'C': np.logspace(-2, 2, 7),  # Regularization parameter
                'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
                'degree': [2, 3, 4],  # Degree for poly kernel
                'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),  # Kernel coefficient for 'rbf' and 'poly'
            }
        case 'neural_network':
            param_dist = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['adam'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [5000, 10000, 20000],
                'early_stopping': [True, False],
            }
        case 'logistic_regression':
            param_dist = {
                'penalty': ['l1', 'l2'],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear'],
                'max_iter': [100, 200, 300, 400, 500, 600],
            }
        case 'random_forest':
            param_dist = {
                'n_estimators': [5, 10, 100, 150],
                'max_depth': [6, 8, 10, 12, 14],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6],
                'ccp_alpha': [0, 0.5, 1]}
    return param_dist


def oversample(X, y, oversampler='SMOTE', random_state=42):
    if oversampler != 'SMOTE':
        # Use SMOTE to oversample the minority classes
        # Identify the two most frequent classes
        top_class = y.value_counts().idxmax()

        # Filter indices of the two most frequent class
        indices_not_top_class = y[y != top_class].index
        indices_top_class = y[y == top_class].index

        oversampler = sv.MulticlassOversampling(oversampler=oversampler,
                                                oversampler_params={'random_state': random_state})
        X_samp, y_samp = oversampler.sample(X.loc[indices_not_top_class], y.loc[indices_not_top_class])
        X_samp = pd.DataFrame(X_samp, columns=X.columns)
        y_samp = pd.Series(y_samp)
        X = pd.concat([X_samp, X.loc[indices_top_class]], ignore_index=True)
        y = pd.concat([y_samp, y.loc[indices_top_class]], ignore_index=True)
    else:
        oversampler = SMOTE(random_state=random_state)
        X, y = oversampler.fit_resample(X, y)
    return X, y


def convert_to_cell_model(data):
    numeric_columns = data.select_dtypes(include=['number']).columns
    # Filter DataFrame to include 'CellName', 'Group', and numeric columns
    selected_columns = ['CellName', 'Group'] + list(numeric_columns)
    numeric_df = data[selected_columns]
    result = numeric_df.groupby(['CellName', 'Group']).mean().reset_index()
    return result


def scaling(X_train, X_val, X_test=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    if X_test is not None:
        X_test = scaler.transform(X_test)
    return X_train, X_val, X_test


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def data_preparation(data, feature_names, path_to_json, all_cells, protocols, path_to_csv, cell_model=False,
                     test_size=0.2, read_json_files=False, threshold_detector='spikecount'):
    if read_json_files:
        for protocol in protocols:
            data = pd.concat(
                [data, data_preprocessing(path=path_to_json, all_cells=all_cells, feature_names=get_features(protocol),
                                          rm_in_cells=True, threshold_detector=threshold_detector, repetition_index=0,
                                          protocol_to_plot=protocol, cells_to_be_removed=outliers_to_be_removed)])
        data = data.reset_index(drop=True)
        # only extract resistance from IV, intrinsic values for a given cell
        # Duplicate IV values to all rows of a given cell
        for value in data['CellName'].unique():
            # Identify rows with IV value
            rows_with_IV = data['CellName'] == value
            if rows_with_IV.any():
                # Extract the IV values in data
                IV_peak_m = data.loc[rows_with_IV, 'IV_peak_m'].iloc[0]
                IV_steady_m = data.loc[rows_with_IV, 'IV_steady_m'].iloc[0]

                # Fill NaN values in column C with the identified value
                data.loc[rows_with_IV, 'IV_peak_m'] = data.loc[rows_with_IV, 'IV_peak_m'].fillna(IV_peak_m)
                data.loc[rows_with_IV, 'IV_steady_m'] = data.loc[rows_with_IV, 'IV_steady_m'].fillna(IV_steady_m)

            else:
                continue
        # Remove rows with only IV values and no other features
        data = data[data['protocol'] != 'IV']

    else:
        # Read data directly from a csv file
        data = pd.read_csv(path_to_csv, sep=',')

    # Only keep first AP
    data = data.loc[data['AP_index'] == 0]

    # For missing IV values, calculate the median for each group
    median_values_peak = data.groupby('Group')['IV_peak_m'].transform('median')
    # Fill NaN values with the corresponding group's median
    data['IV_peak_m'] = data['IV_peak_m'].fillna(median_values_peak)
    median_values_steady = data.groupby('Group')['IV_steady_m'].transform('median')
    # Fill NaN values with the corresponding group's median
    data['IV_steady_m'] = data['IV_steady_m'].fillna(median_values_steady)

    # Average all traces of each cell
    if cell_model:
        data = convert_to_cell_model(data)

    # Add a column to facilitate split on cell names and cell class
    data['Split column'] = data['CellName'] + ' ' + data['Group']

    # Reset the index to have a sorted dataframe
    data = data.reset_index(drop=True)

    X_train_val, y_train_val, train_val_data, X_test, y_test, test_data = data_splitting(data, feature_names,
                                                                                         test_size=test_size,
                                                                                         random_state=random_state)

    return X_train_val, y_train_val, train_val_data, X_test, y_test, test_data, data


def data_splitting(data, feature_names, test_size, random_state=42):
    # Split training and test data on the cell names and the groups to keep approx the same ratio of cells in
    # train and test
    unique_groups = data['Group'].unique()

    cells_train_val = []
    cells_test = []
    split_train_val = []
    for group in unique_groups:
        split_names = data['Split column'].loc[data['Group'] == group].unique()
        train_split, test_split = train_test_split(split_names, test_size=test_size, random_state=random_state)
        train_name = [split.split(' ')[0] for split in train_split]
        test_name = [split.split(' ')[0] for split in test_split]
        cells_train_val.extend(train_name)
        split_train_val.extend(train_split)
        cells_test.extend(test_name)

    # Filter the original data based on the split cells
    train_val_data = data[data['CellName'].isin(cells_train_val)]
    test_data = data[data['CellName'].isin(cells_test)]

    # Create train and test sets
    X_train_val = train_val_data[feature_names]
    X_test = test_data[feature_names]
    y_train_val = train_val_data['Group']
    y_test = test_data['Group']
    return X_train_val, y_train_val, train_val_data, X_test, y_test, test_data


def hyperparameter_tuning(train_val_data, feature_names, model_type, num_classes, test_size=0.2, epochs=100,
                          oversampling=True, random_state=42):
    # Define the number of folds for cross-validation
    n_splits = 5

    # Initialize parameter search space
    param_dist = param_search_space(model_type)

    # Initialize lists to store results
    best_val_f1 = 0

    # Iterate over the folds
    for i in range(n_splits):
        # Split the data into training and validation sets
        X_train, y_train, train_data, X_val, y_val, val_data = data_splitting(train_val_data, feature_names,
                                                                              test_size=test_size,
                                                                              random_state=random_state + i)

        # Oversample data because of imbalanced dataset, each class gets the same number of cells, only for training
        # these oversamplers are also good ADASYN, MSYN, SMOTE_Cosine, SMOTE_D, NT_SMOTE
        if oversampling:
            X_train, y_train = oversample(X=X_train, y=y_train,
                                          oversampler='SMOTE',
                                          random_state=random_state)

        # Normalize the data
        if model_type != 'logistic_regression':
            X_train, X_val, _ = scaling(X_train, X_val)

        # Model creation
        match model_type:
            case 'random_forest':
                model = get_model(model_type, random_state=random_state)
            case 'logistic_regression':
                model = get_model(model_type, random_state=random_state)
                if interaction_terms:
                    # Use PolynomialFeatures to create interaction terms
                    degree = 3  # You can adjust the degree as needed
                    poly = PolynomialFeatures(degree, interaction_only=True, include_bias=False)
                    X_train = poly.fit_transform(X_train)
                    X_val = poly.transform(X_val)
            case 'SVM':
                model = get_model(model_type, random_state=random_state)
            case 'neural_network':
                model = get_model(model_type, random_state=random_state)
            case 'knn':
                model = get_model(model_type, random_state=random_state)
            case 'custom_nn':
                model = KerasClassifier(model=get_model, model_type=model_type, input_shape=(X_train.shape[1],),
                                        num_classes=num_classes, random_state=random_state, dropout_rate=0,
                                        l2_reg=0, num_hidden_layers=1)
                # Convert labels to number to make it readable to the model
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_val = label_encoder.transform(y_val)
                y_train = to_categorical(y_train, num_classes=num_classes)
                y_val = to_categorical(y_val, num_classes=num_classes)
            case 'xgb':
                model = get_model(model_type, random_state=random_state)
                # Convert labels to number to make it readable to the model
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_val = label_encoder.transform(y_val)
        model = RandomizedSearchCV(model,
                                   param_distributions=param_dist,
                                   n_iter=20,  # Number of random combinations to try
                                   scoring='f1_weighted',  # Use an appropriate scoring metric
                                   cv=None,  # Number of cross-validation folds
                                   n_jobs=-1,  # Use all available cores
                                   random_state=random_state,
                                   )
        # Fit the model to training data
        if model_type == 'custom_nn':
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            clf = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=int(epochs / 4),
                            callbacks=[early_stopping])
        else:
            clf = model.fit(X_train, y_train)
        # Make predictions on the validation set
        y_pred_val = model.predict(X_val)

        # Convert encoded predictions to string label to make it readable to humans
        if model_type == 'custom_nn':
            y_pred_val_indices = np.argmax(y_pred_val, axis=1)
            y_val_indices = np.argmax(y_val, axis=1)
            y_pred_val = label_encoder.inverse_transform(y_pred_val_indices)
            y_val = label_encoder.inverse_transform(y_val_indices)
        elif model_type == 'xgb':
            y_pred_val = label_encoder.inverse_transform(y_pred_val)
            y_val = label_encoder.inverse_transform(y_val)

        # Calculate validation metrics, take weighted average since we are considering multi classes
        f1 = f1_score(y_val, y_pred_val, average='weighted')

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model = clf.best_estimator_

    print('best f1: ' + str(best_val_f1))
    return best_model


def model_fitting(X_train_val, X_test, y_train_val, best_model, model_type, epochs=100,
                  oversampling=True):
    # Oversample data because of imbalanced dataset, each class gets the same number of cells, only apply to training
    if oversampling:
        X_train_val, y_train_val = oversample(X=X_train_val, y=y_train_val,
                                              oversampler='SMOTE', random_state=random_state)
    if model_type != 'logistic_regression':
        X_train_val, X_test, _ = scaling(X_train_val, X_test)
    else:
        if interaction_terms:
            # Use PolynomialFeatures to create interaction terms
            degree = 3  # You can adjust the degree as needed
            poly = PolynomialFeatures(degree, interaction_only=True, include_bias=False)
            X_train_val = poly.fit_transform(X_train_val)
            X_test = poly.transform(X_test)

    if model_type == 'xgb':
        label_encoder = LabelEncoder()
        y_train_val = label_encoder.fit_transform(y_train_val)

    # Convert labels to number to make it readable to the model
    if model_type == 'custom_nn':
        label_encoder = LabelEncoder()
        y_train_val = label_encoder.fit_transform(y_train_val)
        y_train_val = to_categorical(y_train_val)
        # Train best model on train+val data
        best_model = best_model.fit(X_train_val, y_train_val, epochs=epochs)
    else:
        best_model = best_model.fit(X_train_val, y_train_val)

    # Predict the class from features
    y_pred_test = best_model.predict(X_test)

    # Convert encoded predictions to string label to make it readable to humans
    if model_type == 'custom_nn':
        y_pred_test_indices = np.argmax(y_pred_test, axis=1)
        y_pred_test = label_encoder.inverse_transform(y_pred_test_indices)
    elif model_type == 'xgb':
        y_pred_test = label_encoder.inverse_transform(y_pred_test)

    return y_pred_test


def plot_results(y_test, y_pred_test, output_figure_path, class_labels):
    # Confusion matrix and classification report
    confusion = confusion_matrix(y_test, y_pred_test)
    print("Test confusion matrix:\n", confusion)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix, ' + model_type)
    plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=90)
    plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=0)
    plt.tight_layout()
    plt.savefig(output_figure_path + 'confusion_matrix.png')
    plt.show()

    # Calculate validation metrics, take weighted average since we are considering multi classes
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')

    # Display metrics
    print("F1 (weighted) on test set:", f1_test)
    print("Recall (weighted) on test set:", recall_test)
    print("Precision (weighted) on test set:", precision_test)

    # Confusion matrix and classification report
    report = classification_report(y_test, y_pred_test, target_names=np.unique(y_test).astype(str))
    print("Test classification Report:\n", report)

    # Convert the classification report to a DataFrame
    report_df = pd.read_fwf(StringIO(report), index_col=0)

    # Convert the DataFrame to an HTML table
    html_table = report_df.to_html(classes='table table-striped')

    # Save the HTML table to a file
    with open(output_figure_path + 'classification_report.html', 'w') as file:
        file.write(html_table)


if __name__ == "__main__":
    # Adapt these paths to your configuration
    outliers_to_be_removed = '/home/lucas/BBP/Code/outliers.txt'  # List of outliers not used in our model
    path_to_json_files = '/home/lucas/BBP/Data/jsonData'  # acell data in json format (each file is a cell)
    path_to_csv = '/home/lucas/BBP/Code/aecode_data.csv'  # aecode_data.csv was created with data.to_csv() to speed up
    # the execution of this program
    output_figure_path = '/home/lucas/BBP/Code/Images/'  # Path to where you want to store the plots
    threshold_detector = 'spikecount'  # Feature to determine when we are above threshold,
    # for example detect an AP if spikecount is > 0
    protocols = ['IV', 'IDRest', 'APWaveform']
    data = pd.DataFrame()
    read_json_files = False  # Set to not to read data from a csv instead of all json files
    # (much faster but needs to be saved first)

    parameter_scan = True  # Perform grid or random search for different model hyperparameters
    interaction_terms = True  # Only for logistic regression, i.e. include second order terms x1x2
    # instead of only x1 and x2
    model_type = 'logistic_regression'  # pick from ['xgb', 'neural_network', 'SVM', 'logistic_regression',
    # 'random_forest', 'knn', 'custom_nn']
    oversampling = True  # Use SMOTE to generate syntethic samples
    random_state = 42  # random seed to always obtain the same result
    cell_model = True  # Set to True if you want a cell model instead of a first AP model
    epochs = 50  # number of epochs to train the custom neural network, hyperparameter tuning is made on int(epochs/4)
    if cell_model and model_type == 'logistic_regression':
        test_size = 0.25
    elif cell_model and model_type == 'SVM':
        test_size = 0.2
    elif not cell_model:
        test_size = 0.05
    else:
        test_size = 0.2

    all_cells = get_all_cells(path_to_json_files)
    path = os.path.join(os.getcwd(), 'CellList30-May-2022.csv')

    feature_names = [feature for protocol in protocols for feature in get_features(protocol)]
    feature_names = set(feature_names)

    # You can remove features with the line below, stim is never used for prediction
    feature_names = [item for item in feature_names if item != 'stim']
    feature_names = [item for item in feature_names if item != 'ISI_values']
    # feature_names = [item for item in feature_names if item != 'IV_peak_m']
    # feature_names = [item for item in feature_names if item != 'IV_steady_m']

    # Split the data and put it in the correct table format for model training
    (X_train_val, y_train_val, train_val_data,
     X_test, y_test, test_data, data) = data_preparation(data=data,
                                                         feature_names=feature_names,
                                                         path_to_json=path,
                                                         all_cells=all_cells,
                                                         protocols=protocols,
                                                         path_to_csv=path_to_csv,
                                                         cell_model=cell_model,
                                                         test_size=test_size,
                                                         read_json_files=read_json_files,
                                                         threshold_detector=threshold_detector)

    # Classes are species, brain area, cell type such as Mouse Amygdala PC
    short_labels = [label.split(' AP')[0] for label in y_train_val]
    class_labels = np.unique(short_labels)
    num_classes = len(class_labels)

    # Hyperparameter tuning
    best_model = hyperparameter_tuning(train_val_data=train_val_data, feature_names=feature_names,
                                       model_type=model_type,
                                       num_classes=num_classes, epochs=epochs, oversampling=oversampling,
                                       test_size=test_size, random_state=random_state)

    # Create prediction
    y_pred_test = model_fitting(X_train_val=X_train_val, X_test=X_test, y_train_val=y_train_val, best_model=best_model,
                                model_type=model_type, epochs=epochs, oversampling=oversampling)

    # Compute, display and save results
    plot_results(y_test=y_test, y_pred_test=y_pred_test, output_figure_path=output_figure_path,
                 class_labels=class_labels)
    pass
