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


def explode_on_fuels(df, cols):
    """
    Perform explode on the fuel columns.
    Assumes that df has a single row.

    :param pandas.Dataframe df: Dataframe to explode with cols
    :param list(str) cols: Name of columns to explode. Underlying series must
    each be the same length.
    :return: Exploded dataframe.
    """
    if len(df) != 1:
        raise ValueError('argument df must be a single row')
    if len(cols) == 0:
        return df
    

    exploded_cols = []
    for col in cols:
        exploded_col = df.pop(col).explode(col)
        if len(exploded_cols) > 0 and len(exploded_cols[0]) != len(exploded_col):
            raise ValueError('Cannot explode on columns with different length')
        exploded_cols.append(exploded_col)

    n = len(exploded_cols[0]) if len(exploded_cols) > 0 else 1
    df = pd.concat([df] * n, ignore_index=True)
    exploded_df = pd.concat(exploded_cols, axis=1)
    return pd.concat([df, exploded_df], axis=1)
