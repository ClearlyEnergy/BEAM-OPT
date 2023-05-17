# !/usr/bin/env python
# encoding: utf-8
"""
:copyright (c) 2014 - 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Department of Energy) and contributors. All rights reserved.  # NOQA
:author
"""
import json
import numpy as np
import pandas as pd

BASELINE_VALIDATION_COLUMNS = ['Annual_Bill', 'Electricity_CO2', 'Gas_CO2', 'Total_CO2', 'Electricity_Consumption',
                               'Gas_Consumption', 'Electricity_Bill', 'Gas_Bill']

MEASURE_VALIDATION_COLUMNS = ['Identifier', 'Cost', 'Annual_Saving', 'Total_CO2', 'Electricity_Saving', 'Gas_Saving',
                              'Electricity_Bill_Saving', 'Gas_Bill_Saving', 'Category', 'Group', 'Index']


def validate_complete_data(FullData, ids):
    """
    Check that a CompleteData object has all of the necessary buildings within it.
    Check that it has all of the columns needed and that those columns are not NaN.
    """
    errors = []
    # Check that all IDs are in FullData
    ids = [str(id_) for id_ in ids]
    if any(bld_ID not in FullData.baseline.Building.to_list() for bld_ID in ids):
        errors.append('Invalid Building ID ' + ','.join(
            list(set(ids) - set(FullData.baseline.Building.to_list()))))

    test_for_nan = True
    # Check that it has all of the Columns needed
    baseline_columns = list(FullData.baseline.columns)
    baseline_columns_diff = list(set(BASELINE_VALIDATION_COLUMNS) - set(baseline_columns))
    if baseline_columns_diff:
        test_for_nan = False
        errors.append('Baseline data is missing columns: %s' % ', '.join(baseline_columns_diff))
    measure_columns = list(FullData.measure_data.columns)  # TODO Convert this to return just the DF in all locations
    measure_columns_diff = list(set(MEASURE_VALIDATION_COLUMNS) - set(measure_columns))
    if measure_columns_diff:
        test_for_nan = False
        errors.append('Measure data is missing columns: %s' % ', '.join(measure_columns_diff))

    # Only test for NaN values if all columns are available:
    if test_for_nan:
        if FullData.baseline.loc[FullData.baseline.Building.notna(), BASELINE_VALIDATION_COLUMNS].isna().values.any():
            errors.append('Missing Data in Baseline dataset')
        measures_result = FullData.get_measure_data(ids)
        measures_df = pd.read_json(json.dumps(measures_result['measure_data']), orient='split')
        if measures_df.loc[measures_df.Building.notna(), ['Identifier', 'Cost', 'Category', 'Group', 'Index']
                           ].isna().values.any():
            errors.append("Missing data in Measures dataset")

        if measures_df.loc[measures_df.Building.notna(), ['Annual_Saving', 'Total_CO2', 'Electricity_Saving',
                                                          'Gas_Saving','Electricity_Bill_Saving', 'Gas_Bill_Saving']
                           ].isna().values.any():
            errors.append("Missing data in Scenarios dataset")

    return errors if errors else None


def pre_validate_parameters(OPT, budget, target, penalty, delta, scenario):
    """
    Check that all of the parameters are valid
    """
    errors = []
    if scenario not in ['Consumption', 'Emission']:
        errors.append('Invalid Scenario. Must choose between Consumption/Emission')
    if len(budget) != len(OPT.timeline) or len(target) != len(OPT.timeline):
        errors.append('Invalid input: inconsistent with time line')
    if (np.array(budget) < 0).any():  # budget at each time
        errors.append('Invalid input: must choose target in [0,1]')
    if (np.array(target) < 0).any() or (np.array(target) > 1).any():
        # target percentage of consumption/emission to remain at each time
        errors.append('Invalid input: must choose target in [0,1]')
    if delta < 0 or delta > 1:  # discount factor
        errors.append('Invalid input: must choose delta in [0,1]')
    if penalty < 0:  # penalty rate for consumption/emission
        errors.append('Invalid input: must choose nonnegative penalty')

    # Check that Target is Achievable
    timeline_df = pd.DataFrame(np.arange(OPT.timeline[0], OPT.timeline[-1] + 1), columns=['Year'])
    target_df = pd.DataFrame({'Target': target, 'Year': OPT.timeline}
                             ).merge(timeline_df, on='Year', how='right').fillna(method='ffill').set_index('Year')

    if scenario in OPT.lookups:
        lookup = OPT.lookups[scenario]
        if np.isinf(penalty):
            # Check whether target is achievable
            target = target_df.Target * OPT.baseline[lookup['optimize']].iloc[0]
            max_reduction = OPT.df.groupby('Group')[lookup['data']].max().sum()
            ind = getattr(OPT, lookup['target']) < (OPT.baseline[lookup['optimize']] - max_reduction)
            if any(ind):
                errors.append('%s target too low to fulfill. See Violating years for detail' % scenario)
                # return{'status':'error','message':'Consumption target too low to fulfill. See violating_years for
                # detail','violating_years':pd.DataFrame({'achievable percentage of remaining consumption':1-
                # max_reduction/self.baseline.Total_Consumption.loc[ind],'target persentage':target_df.loc[ind,
                # 'Target']})}
    else:
        errors.append('Could not check whether Target was achievable due to incorrect scenario')

    return errors if errors else None


def post_validate_parameters(OPT, scenario):
    """
    Check that consumption/emission targets were set sucessfully
    """
    lookup = OPT.lookups[scenario]
    errors = []
    if not hasattr(OPT, lookup['target']):
        errors.append('Must set Target before calling Optimization')
    return errors if errors else None
