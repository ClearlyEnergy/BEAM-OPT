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