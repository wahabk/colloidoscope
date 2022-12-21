"""
Colloids average precision metric function
"""

import numpy as np
from read import read_x, read_y, extract_y
from metric import average_precision

def ap_metric_function(dataframe_y_true, dataframe_y_pred):
    """
        Average precision function for the colloids challenge
        This metric is non symmetric

    Args
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the true values of y.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_true = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_y_pred: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes. This must not be NaN.
    """

    scores = []
    diameters = [6,11,11,11,8,7,8,9,9,6,]
    for index in range(10):
        y_pred = extract_y(dataframe_y_pred, index)
        y_true = extract_y(dataframe_y_true)

        ap, precisions, recalls = average_precision(y_true, y_pred, diameter=diameters[index], canvas_size=(128,128,128))
        scores.append(ap)

    score = np.mean(scores)

    return score

# The following lines show how the csv files are read
if __name__ == '__main__':
    import pandas as pd

    CSV_FILE_Y_TRUE = 'y_test.csv'  # path of the y_true csv file
    CSV_FILE_Y_PRED = 'y_benchmark.csv'  # path of the y_pred csv file
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_true = df_y_true.dropna() # NOTE Drop NA is required to fit website formatting
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    df_y_pred = df_y_pred.dropna()
    # df_y_pred = df_y_pred.loc[df_y_true.index] # This isn't needed in my case

    print(ap_metric_function(df_y_true, df_y_pred))
