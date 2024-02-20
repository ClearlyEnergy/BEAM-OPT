# !/usr/bin/env python
# encoding: utf-8
"""
:copyright (c) 2014 - 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Department of Energy) and contributors. All rights reserved.  # NOQA
:author
"""
import json
import numpy as np
import pandas as pd

from beam_opt.models.data_container import CompleteData
from beam_opt.models.optimizer import Optimizer, LOOKUP

BASELINE_VALIDATION_COLUMNS = ['Annual_Bill', 'Electricity_CO2', 'Gas_CO2', 'Total_CO2', 'Electricity_Consumption',
                               'Gas_Consumption', 'Electricity_Bill', 'Gas_Bill']

MEASURE_VALIDATION_COLUMNS = ['Identifier', 'Cost', 'Annual_Saving', 'Total_CO2', 'Electricity_Saving', 'Gas_Saving',
                              'Electricity_Bill_Saving', 'Gas_Bill_Saving', 'Category', 'Group', 'Index']


def validate_complete_data(complete_data: CompleteData, ids):
    """
    Check that a CompleteData object has all of the necessary buildings within it.
    Check that it has all of the columns needed and that those columns are not NaN.
    """
    errors = []
    # Check that all Building IDs are in complete_data
    ids = [str(id_) for id_ in ids]
    if any(id_ not in complete_data.baseline.Building.to_list() for id_ in ids):
        errors.append('Invalid Building ID ' + ','.join(
            list(set(ids) - set(complete_data.baseline.Building.to_list()))))

    test_for_nan = True
    # Check that it has all of the Columns needed
    baseline_columns = list(complete_data.baseline.columns)
    baseline_columns_diff = list(set(BASELINE_VALIDATION_COLUMNS) - set(baseline_columns))
    if baseline_columns_diff:
        test_for_nan = False
        errors.append('Baseline data is missing columns: %s' % ', '.join(baseline_columns_diff))

    # TODO: Convert this to return just the DF in all locations
    measure_columns = list(complete_data.measure_data.columns)
    measure_columns_diff = list(set(MEASURE_VALIDATION_COLUMNS) - set(measure_columns))
    if measure_columns_diff:
        test_for_nan = False
        errors.append('Measure data is missing columns: %s' % ', '.join(measure_columns_diff))

    # Only test for NaN values if all columns are available:
    if test_for_nan:
        # Should already be checked by pre-parsing
        sub_df = complete_data.baseline.loc[complete_data.baseline.Building.notna(), BASELINE_VALIDATION_COLUMNS]
        nan_values = sub_df.isna().values.any()
        if nan_values:
            cols_with_nans = sub_df.columns[nan_values].tolist()[0]
            errors.append('Missing Data in Baseline dataset in columnns: ' + ', '.join(cols_with_nans))

        measures_result = complete_data.get_measure_data(ids)
        measures_df = pd.read_json(json.dumps(measures_result['measure_data']), orient='split')

        sub_df = measures_df.loc[measures_df.Building.notna(), ['Identifier', 'Cost', 'Category', 'Group', 'Index']]
        nan_values = sub_df.isna().values.any()
        if nan_values:
            cols_with_nans = sub_df.columns[nan_values].tolist()[0]
            errors.append('Missing data in one or more Measures: ' + ', '.join(cols_with_nans))

        sub_df = measures_df.loc[measures_df.Building.notna(),
                                 ['Annual_Saving', 'Electricity_Saving',
                                  'Gas_Saving', 'Electricity_Bill_Saving', 'Gas_Bill_Saving']]
        mapping = {'Annual_Saving': 'annual_cost_savings',
                   'Total_CO2': 'annual_electricity_energy/annual_natural_gas_energy',
                   'Electricity_Saving': 'annual_electricity_energy',
                   'Gas_Saving': 'annual_natural_gas_energy',
                   'Electricity_Bill_Saving': 'annual_electricity_savings',
                   'Gas_Bill_Saving': 'annual_natural_gas_savings'
                   }

        nan_values = sub_df.isna().values
        has_nan = [False for i in range(len(sub_df.columns))]
        for row in range(len(nan_values)):
            for col in range(len(nan_values[row])):
                if nan_values[row, col]:
                    has_nan[col] = True

        if sub_df.isna().values.any():
            cols_with_nans = sub_df.columns[has_nan].tolist()
            errors.append('Missing data in one or more Scenarios: ' + ', '.join([mapping[i] for i in cols_with_nans]))

    return errors if errors else None


def pre_validate_parameters(optimizer: Optimizer, budget, target, penalty, delta):
    """
    Check that all of the parameters are valid
    """
    errors = []
    if optimizer.scenario not in ['Consumption', 'Emission']:
        errors.append('Invalid Scenario. Must choose between Consumption/Emission')
    if len(budget) != len(optimizer.timeline) or len(target) != len(optimizer.timeline):
        errors.append('Invalid input: inconsistent with time line')
    if (np.array(budget) < 0).any():  # budget at each time
        errors.append('Invalid input: must choose budget > 0')
    if (np.array(target) < 0).any():
        # target percentage of consumption/emission to remain at each time
        errors.append('Invalid input: must choose target > 0')
    if delta < 0 or delta > 1:  # discount factor
        errors.append('Invalid input: must choose delta in [0,1]')
    if penalty < 0:  # penalty rate for consumption/emission
        errors.append('Invalid input: must choose nonnegative penalty')

    # Check that Target is Achievable
    timeline_df = pd.DataFrame(np.arange(optimizer.timeline[0], optimizer.timeline[-1] + 1), columns=['Year'])
    target_df = pd.DataFrame({'Target': target, 'Year': optimizer.timeline}
                             ).merge(timeline_df, on='Year', how='right').fillna(method='ffill').set_index('Year')

    if optimizer.scenario in LOOKUP:
        lookup = LOOKUP[optimizer.scenario]
        if np.isinf(penalty):
            # Check whether target is achievable
            target = target_df.Target * optimizer.baseline[lookup['optimize']].iloc[0]
            max_reduction = optimizer.df.groupby('Group')[lookup['data']].max().sum()
            ind = getattr(optimizer, lookup['target']) < (optimizer.baseline[lookup['optimize']] - max_reduction)
            if any(ind):
                errors.append('%s target too low to fulfill. See Violating years for detail' % optimizer.scenario)
                # return{'status':'error','message':'Consumption target too low to fulfill. See violating_years for
                # detail','violating_years':pd.DataFrame({'achievable percentage of remaining consumption':1-
                # max_reduction/self.baseline.Total_Consumption.loc[ind],'target persentage':target_df.loc[ind,
                # 'Target']})}
    else:
        errors.append('Could not check whether Target was achievable due to incorrect scenario')

    return errors if errors else None


def post_validate_parameters(optimizer: Optimizer):
    """
    Check that consumption/emission targets were set sucessfully
    """
    lookup = LOOKUP[optimizer.scenario]
    errors = []
    if not hasattr(optimizer, lookup['target']):
        errors.append('Must set Target before calling Optimization')
    return errors if errors else None

