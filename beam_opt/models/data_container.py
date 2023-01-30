# !/usr/bin/env python
# encoding: utf-8
"""
:copyright (c) 2014 - 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Department of Energy) and contributors. All rights reserved.  # NOQA
:author
"""
import json
import numpy as np
import pandas as pd
import warnings
from beam_opt.utils.parse import group_identifier_rmi


class CompleteData:
    """
    Take dataframe (measure_data) returned by parse function as input, add exclusion/priority relationship for use
    of optimization program
    """

    def __init__(self, parsed_data, file_exclusion, default_priority=None, source='RMI'):
        # Convert measure data and baseline data back to pandas dataframe
        parsed_data['measure_data'] = pd.read_json(json.dumps(parsed_data['measure_data']), orient='split')
        parsed_data['baseline_data'] = pd.read_json(json.dumps(parsed_data['baseline_data']), orient='split')
        self.baseline = parsed_data['baseline_data']
        self.baseline['Building'] = self.baseline['Building'].astype(str)

        # Add exclusion groups and indices
        if source == 'RMI':
            # Read competing measure file
            compete_df = pd.read_excel(file_exclusion.temporary_file_path(), converters={'Measure #': str},
                                       usecols=lambda x: x != 'Measure Name')
            compete_df.rename({'Measure #': 'Code'}, axis=1, inplace=True)
            compete_df = compete_df.groupby('Code').any()
            self.measure_data = parsed_data['measure_data'].groupby('Building', group_keys=False).apply(
                lambda x: add_group_rmi(x, compete_df))
            self.measure_data.reset_index(drop=True, inplace=True)
        elif source == 'BEAM' or source == 'BuildingSync':
            # Format Measures into Groups based on their Group name
            self.measure_data = parsed_data['measure_data'].groupby('Building', group_keys=False).apply(
                lambda x: add_group_buildingsync(x))
        self.measure_data['Building'] = self.measure_data['Building'].astype(str)

        # Parse priority for all exclusion groups
        self.priority_charts = self.measure_data.groupby('Building', group_keys=False).apply(
            lambda x: bld_priority(x, default_priority)).to_dict()

    def add_start_year(self):
        # This specifies the earliest possible year to install each measure.
        # Here I just randomly set a column for test. It will need to adapt to user input.
        # The result should be an additional column in self.measure_data denoting the start year.
        tmp = np.repeat(2020, self.measure_data.shape[0])
        tmp[(self.measure_data.Group >= 8).values & (self.measure_data.Group < 14).values] = 2025
        tmp[self.measure_data.Group > 14] = 2034
        self.measure_data['Start_Year'] = tmp
        return

    def get_baseline_data(self, bld_list):
        bld_list = [str(x) for x in bld_list]
        df_rtn = self.baseline.loc[self.baseline.Building.isin(bld_list)]
        df_rtn.reset_index(inplace=True, drop=True)
        df_rtn = json.loads(df_rtn.to_json(orient="split"))
        return {'status': 'success',
                'message': '',
                'building_data': df_rtn}

    def get_measure_data(self, bld_list):
        bld_list = [str(x) for x in bld_list]
        df_rtn = self.measure_data.loc[self.measure_data.Building.isin(bld_list)].sort_values(
            by=['Building', 'Group', 'Index'])
        df_rtn.reset_index(inplace=True, drop=True)
        df_rtn = json.loads(df_rtn.to_json(orient="split"))
        return {'status': 'success',
                'message': '',
                'measure_data': df_rtn}

    def get_priority_chart(self, bld_list):
        bld_list = [str(x) for x in bld_list]
        df_rtn = dict([(ID, [*self.priority_charts[ID].values()][0]) for ID in bld_list])
        # df_rtn=json.loads(df_rtn.to_json(orient="split"))  # df_rtn not a DataFrame
        return {'status': 'success',
                'message': '',
                'priority_chart': df_rtn}

# Helper functions for Data Container Object


def add_group_rmi(bld_df, compete_df):
    """
    Add columns to identify exclusion groups and in-group indices for RMI data
    """
    df = bld_df.copy()
    # Add missing LED measures
    compete_df = compete_df.append(pd.DataFrame([compete_df.loc['E-L-03']] * 2, index=['E-L-06-a', 'E-L-07-a']))
    # Check if any missing measure and fill na with similar measures
    ind = [x not in compete_df.index for x in bld_df.Identifier]
    if any(ind):
        warnings.warn("Incomplete data: Missing exclusion relationship of Measure " +
                      ', '.join(df.Identifier[ind].drop_duplicates().values)+" is filled with similar measures.\n")
        for code in df.Identifier[ind]:
            ind1 = (pd.Series(compete_df.index).apply(group_identifier_rmi) == group_identifier_rmi(code))
            if len(ind1) == 0:  # If no similar measure is detected
                raise Exception("Incomplete data: Measure " + ', '.join(df.Identifier[ind].drop_duplicates().values) +
                                " lack exclusion relationship and no similar measures are detected.\n")
            compete_df = compete_df.append(pd.DataFrame([ind1[0]], index=[code]))

    # Add exclusion groups and index within group
    grouped = pd.Series(compete_df.loc[bld_df.Identifier].values.tolist()).apply(tuple).to_frame(0).groupby(0)[0]
    df['Group'] = grouped.ngroup().values
    df['Index'] = grouped.cumcount().values + 1
    return df


def add_group_buildingsync(df):
    """
    Given a df for a single building, add exclusion groups and index within each groups
    Each Measure contains the name of the category it belongs to
    Attach an index to each category, and within the category, index the measures
    :param df: Dataframe object of single building measure data
    """
    df['Group'] = df.groupby('Category').ngroup()
    df['Index'] = df.groupby('Group').cumcount()
    return df


def bld_priority(bld_df, default_priority):
    """
    Construct priority chart based on Group & Category
    """
    category = bld_df.groupby('Group', group_keys=False).Category.first()
    # TODO: Provide category to user to drag the order
    priority_chart = pd.DataFrame(index=category.index, columns=category.index)

    # Add priority relationships. If default_priority==None, no priority; if default_priority==True, only prefer
    # envelope prior to HVAC
    if default_priority:
        ind_env = (category == 'building_envelope_modifications')
        priority_chart.loc[ind_env, category == 'other_hvac'] = 'x'
    elif default_priority is not None:
        x = 1  # TODO
    return {bld_df.Building.iloc[0]: priority_chart}
