# !/usr/bin/env python
# encoding: utf-8
"""

"""

from beam_opt.models.data_container import CompleteData
from beam_opt.models.optimizer import Optimizer
from beam_opt.utils.parse import parse_measures
from beam_opt.utils.validate import validate_complete_data, pre_validate_parameters, post_validate_parameters
from django.http import JsonResponse


def _preprocess(request, json_response=True):
    # Parse file formats
    parsing_result = parse_measures(request)
    if parsing_result.get('status') != 'success':
        return JsonResponse({'error': parsing_result.get('message')})

    result = parse_measures(request)
    if result['status'] == 'success':
        result.pop('status')
        result.pop('message')

        return JsonResponse(result) if json_response else result
    else:
        return JsonResponse({'error': result['message']})


def _optimize(request, json_response=True, parsing_result=None):
    # Validate the data
    parsing_result = request.data if json_response else parsing_result
    DataObject = CompleteData(parsing_result, request.data.get('exclusion_file'),
                              request.data.get('default_priority', False),
                              source=parsing_result.get('source'))
    building_id = DataObject.baseline['Building'][0]
    errors = validate_complete_data(DataObject, [building_id])
    if errors is not None:
        errors = {'errors': errors}
        return JsonResponse(errors) if json_response else errors

    # ### Optimizer
    obj = Optimizer(DataObject, building_id, request.data.get('timeline'))

    # Validate Parameters
    errors = pre_validate_parameters(obj, request.data.get('budget'), request.data.get('targets'),
                                     request.data.get('penalty'), request.data.get('delta'),
                                     request.data.get('scenario'))
    if errors is not None:
        errors = {'errors': errors}
        return JsonResponse(errors) if json_response else errors

    # Set parameters in the object
    obj.set_parameters(request.data.get('budget'), request.data.get('targets'),
                       request.data.get('penalty'), request.data.get('delta'),
                       request.data.get('scenario'))

    # Validate they were set and processed properly
    errors = post_validate_parameters(obj, request.data.get('scenario'))
    if errors is not None:
        errors = {'errors': errors}
        return JsonResponse(errors) if json_response else errors

    # Run Optimizer
    obj.optimize(request.data.get('scenario'))

    # Get results
    levels = obj.get_level(request.data.get('scenario'))
    if levels is None:
        error = {'error': 'No levels were found for this property'}
        return JsonResponse(error) if json_response else error

    measures = obj.print_solution()
    if measures is None:
        error = {'error': 'No measures were found for this property'}
        return JsonResponse(error) if json_response else error

    output = {'measures': measures.to_json(orient='split'),
              'levels': levels.to_json(orient='split')}

    return JsonResponse(output) if json_response else output


def _preprocess_and_optimize(request):
    parsing_result = _preprocess(request, json_response=False)
    if parsing_result.get('error') is not None:
        return JsonResponse(parsing_result)

    optimize_result = _optimize(request, json_response=False, parsing_result=parsing_result)
    return JsonResponse(optimize_result)
