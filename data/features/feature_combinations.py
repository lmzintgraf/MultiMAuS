"""
Some simple functions to generate new features by combining existing features

@author Dennis Soemers
"""


def pair_equality(dataframe, column_1, column_2, new_feature_name):
    """
    Adds a new binary feature to an existing dataframe which, for every row,
    is 1 if and only if that row has equal values in two given columns.

    :param dataframe:
        Dataframe to add feature to
    :param column_1:
        Name of first existing column
    :param column_2:
        Name of second existing column
    :param new_feature_name:
        Name of the new column to add
    :return:
        Modified version of given dataframe
    """
    dataframe[new_feature_name] = dataframe.apply(
        lambda row: get_pair_equality(row, column_1, column_2), axis=1)

    return dataframe

def get_pair_equality(row, column_1, column_2):
    """
    Helper function used by pair_equality, to test values of two columns for
    equality in a single row.

    :param row:
        Row from dataframe
    :param column_1:
        Name of first column
    :param column_2:
        Name of second column
    :return:
        1 if and only if the row has the same value in two given columns
    """
    if row[column_1] == row[column_2]:
        return 1
    else:
        return 0
