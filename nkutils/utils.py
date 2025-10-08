import numpy as np
from openpyxl import load_workbook
import pandas as pd

#  auxilary function to read a specific range from an excel spreadsheet. Works with .xlsm (so far)

def read_named_range_to_dataframe(file_path, sheet_name, range_name, header=True):
    workbook = load_workbook(file_path, data_only=True)
    sheet = workbook[sheet_name]
    named_range = workbook.defined_names[range_name]

    destinations = list(named_range.destinations)
    title, coord = destinations[0]

    if ':' not in coord:
        # Single cell (e.g., "A1")
        cell_value = sheet[coord].value
        data = [[cell_value]]
        columns = None

    else:
        # Range of cells (e.g., "A1:B2")
        data = [[cell.value for cell in row] for row in sheet[coord]]
        if header:
            # Use the first row as column names
            columns = data[0]
            data = data[1:]  # Remove the header row from the data
        else:
            columns = None
    return pd.DataFrame(data, columns=columns)

def forward_rate(rfr, t1, t2, compound = "ann"):
    """ Compute the forward rate between two timepoints. """
    if compound == "ann":
        return ((1 + rfr[t2])**t2 / (1 + rfr[t1])**t1)**(t2-t1) - 1
    elif compound == "cont":
        return (rfr[t2] * t2 - rfr[t1] * t1) / (t2 - t1)

def send2clipboard(A,colnames = ["datafrompython"]):
    '''converts a nparray to a dataframe and sends it to the clipboard'''
    A = np.array(A)

    # Reshape if A is 1D (e.g., shape is (N,))
    if A.ndim == 1:
        A = A.reshape(-1, 1)

    if len(colnames) == 1:
        mycolumns = colnames * A.shape[1]
    elif len(colnames) == A.shape[1]:
        mycolumns = colnames
    else:
        raise Exception("Invalid number of column names provided - you can leave empty if easier")
    df = pd.DataFrame(A, columns=mycolumns)
    df.to_clipboard(index=False)


def print_stats(myseries,percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], colnames = None): # myseries need to be an n_sims x n_variables object
    
    stat_names = ['Mean', 'StDev', 'Min', 'Max'] + [f"P{int(p*100)}" for p in percentiles] + ['Skew', 'Kurtosis']
    results = []

    for i in range(myseries.shape[1]):
        series = myseries[:, i]
        stats = [
            np.mean(series),
            np.std(series),
            np.min(series),
            np.max(series),
            *[np.percentile(series, p * 100) for p in percentiles],
            skew(series),
            kurtosis(series)
        ]
        results.append(stats)

    df_stats = pd.DataFrame(np.array(results).T, index=stat_names, columns=[f"Series {i}" for i in range(myseries.shape[1])] if colnames is None else colnames)
    print(df_stats.round(4))


def align_to_schema(subdf: pd.DataFrame, maindf: pd.DataFrame) -> pd.DataFrame:
    ''' concats two dataframes which have some common and some different columns in a sensible manner'''
    data = {}
    for col in maindf.columns:
        if col in subdf.columns:
            data[col] = subdf[col].astype(maindf[col].dtype)
        else:
            if np.issubdtype(maindf[col].dtype, np.number):
                data[col] = 0
            elif np.issubdtype(maindf[col].dtype, np.object_) or maindf[col].dtype == "string":
                data[col] = ""
            else:
                data[col] = pd.Series([""] * len(subdf), dtype=maindf[col].dtype)
    C = pd.DataFrame(data, columns=maindf.columns)
    return C
