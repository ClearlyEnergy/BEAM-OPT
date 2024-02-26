import pandas as pd


def dump_optimizer_input(parsed_input, path=None):
    """
    Write or return a JSON string of CompleteData input
    """
    import json
    result = json.dumps(parsed_input)
    if path is None:
        return result
    with open(path, 'w') as f:
        f.write(result)


def dump_optimizer_result(scenarios, levels, path=None):
    """
    Write or return a JSON string of the serialized result
    """
    result = {}
    result['MEASURES'] = scenarios
    result['LEVELS'] = levels
    import json
    result = json.dumps(result)
    if path is None:
        return result

    with open(path, 'w') as f:
        f.write(result)


def multi_col_explode(df, cols):
    """
    Perform explode on mutliple columns. Same as .explode(['A','B',...]) in newer versions
    of pandas.

    :param pandas.Dataframe df: Dataframe to explode with cols
    :param list(str) cols: Name of columns to explode. Underlying series must
    each be the same length.
    :return: Exploded dataframe.
    """

    exploded_cols = []
    for col in cols:
        exploded_col = df.pop(col).explode(col)
        if len(exploded_cols) > 0 and len(exploded_cols[0]) != len(exploded_col):
            raise ValueError('Cannot explode on columns with different length')
        exploded_cols.append(exploded_col)

    return df.merge(pd.concat(exploded_cols, axis=1), how='cross')
