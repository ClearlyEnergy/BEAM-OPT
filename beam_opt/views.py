# !/usr/bin/env python
# encoding: utf-8
"""

"""

from beam_opt.utils.workflow import _preprocess, _optimize, _preprocess_and_optimize
from django.http import JsonResponse
import pkg_resources
from rest_framework.decorators import api_view


@api_view(['GET'])
def version(request):
    """
    Returns the BEAM-OPT version
    """
    return JsonResponse({
        'version': pkg_resources.require("BEAM-Opt")[0].version,
    })


# TODO Add Schema documentation

@api_view(['POST'])
def preprocess(request):
    """
    Parse a Set of Measures in different file formats, vvalidate the data,
    and structure them into a new JSON formatted object.
    Params for BuildingSync:
    :param: file: xml file in BuildingSync format
    :param: electricity_emission_rate: number
    :param: natural_gas_emission_rate: number

    Params for BEAM:
    :param: property_id: property view id of BEAM
    :param: electricity_emission_rate: number
    :param: natural_gas_emission_rate: number

    Params for RMI:
    :param: file, full set of measures for a property
    :param: file, lifetime info for the set of measures
    """
    # TODO Add check for all of the parameters

    return _preprocess(request)


@api_view(['POST'])
def optimize(request):
    """
    With formatted measures, find the optimal set that will bring the buildings emissions under the target amount.
    # Needed Parameters for this section:
    :param: exclusion_file: xlsx file, file of measure groupings
    :param: default_priority: Bool, use default priority or get from file TODO
    :param: timeline: list, list of year intervals
    :param: budget: list, budget at each of the year intervals
    :param: targets: list, property targets at each of the year intervals
    :param: penalty: float [0-1], penalty applied for exceeding targets
    :param: delta: float [0-1], discount rate
    :param: scenario: string, one of 'emission' or 'consumption'
    """
    # TODO Add check for all of the parameters

    return _optimize(request)


@api_view(['POST'])
def preprocess_and_optimize(request):
    """
    Perform the full optimization pipeline in this request

    Params for Measure Preprocessing
    Params for BuildingSync:
    :param: file: xml file in BuildingSync format
    :param: electricity_emission_rate: number
    :param: natural_gas_emission_rate: number

    Params for BEAM:
    :param: property_id: property view id of BEAM
    :param: electricity_emission_rate: number
    :param: natural_gas_emission_rate: number

    Params for RMI:
    :param: file, full set of measures for a property
    :param: file, lifetime info for the set of measures

    Params for Optimization
    :param: exclusion_file: xlsx file, file of measure groupings
    :param: default_priority: Bool, use default priority or get from file TODO
    :param: timeline: list, list of year intervals
    :param: budget: list, budget at each of the year intervals
    :param: targets: list, property targets at each of the year intervals
    :param: penalty: float [0-1], penalty applied for exceeding targets
    :param: delta: float [0-1], discount rate
    :param: scenario: string, one of 'emission' or 'consumption'
    """
    # TODO Add check for all of the parameters
    return _preprocess_and_optimize(request)

