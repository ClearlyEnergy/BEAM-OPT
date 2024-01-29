# !/usr/bin/env python
# encoding: utf-8
"""
Provide Various Parsing Functions for Measures formats
"""
from django.conf import settings
import json
import numpy as np
import os.path
import pandas as pd
import warnings
import xml.etree.ElementTree as Et
from seed.models.scenarios import Scenario
from seed.serializers.scenarios import ScenarioSerializer

# TODO: Remove columns that optimizer doesn't use.

MEASURE_DF_COLUMNS = [
    'Building',
    'Identifier',
    'Description',
    'Cost',
    'Annual_Saving',
    'Scenario',
    'Electricity_CO2',
    'Gas_CO2',
    'Total_CO2',
    'Electricity_Saving',
    'Gas_Saving',
    'Electricity_Bill_Saving',
    'Gas_Bill_Saving',
    'Category',
    'Cost_Incremental',
    'Cost_Bulk',
    'Life',
]

BASELINE_DF_COLUMNS = [
    'Building',
    'Annual_Bill',
    'Electricity_CO2',
    'Gas_CO2',
    'Total_CO2',
    'Electricity_Consumption',
    'Gas_Consumption',
    'Electricity_Bill',
    'Gas_Bill',
]

PROPERTY_STATE_REQUIRED_COLUMNS = [
    'Annual Savings',
    'Electricity Savings (kBtu)',
    'Natural Gas Savings (kBtu)',
    'Electricity Bill Savings',
    'Natural Gas Bill Savings'
]

standalone = getattr(settings, 'STANDALONE', False)
if not standalone:      # If running with BEAM, import BEAM models
    from seed.models import PropertyView
    from seed.building_sync.building_sync import BuildingSync
    from seed.hpxml.hpxml import HPXML


def get_schema(file):
    """
    Helper function to Parse XML file for it's Schema type
    """
    tree = Et.parse(file)
    root = tree.getroot()
    if root.tag.endswith('BuildingSync'):
        return 'BuildingSync'
    elif root.tag.endswith('HPXML'):
        return 'HPXML'
    else:
        return 'None'


def parse_measures(request):
    """
    Parse Curl Request based on inputs
    """
    files = request.FILES.getlist('file')
    if len(files) > 3:
        return {'status': 'error', 'message': 'Received incorrect file amount'}
    elif 0 < len(files) < 3:
        _, extension = os.path.splitext(files[0].name)
    else:
        file, extension = None, None

    if extension == '.xlsx' and len(files) == 2:            # Parse RMI Files
        blds = ['10112', '10417', '10156']  # This needs to be adapted as user input
        return parse_xlsx(files[0], files[1], bld_list=blds)
    elif extension == '.xml' and len(files) == 1:   # Parse HPXML/BuildingSync
        file = files[0]
        # path = default_storage.save('heart_of_the_swarm.txt', ContentFile(file.read()))
        schema = get_schema(file)  # Determine if it's BuildingSync or HPXML Schema
        if schema == 'BuildingSync':
            # TODO: can't use BS path until it's been converted to using time-dependent data
            return parse_buildingsync(file,
                                      request.data.get('electricity_emission_rate'),
                                      request.data.get('natural_gas_emission_rate'))
        elif schema == 'HPXML':
            return parse_hpxml(file)
        else:
            return {'status': 'error', 'message': 'Could not parse XML. No valid Schema found.'}
    elif extension is None and not standalone:      # Parse BEAM Files
        # TODO: make sure we have this data in the request
        return parse_beam_measures(request.data.get('property_id'),
                                   request.data.get('emission_rates'),
                                   request.data.get('timeline'))
    else:
        return {'status': 'error', 'message': 'No Valid Parse method found'}


def parse_hpxml(file):
    """
    Parse HPXML File
    """
    hpxml = HPXML()
    hpxml.import_file(file.open())
    hpxml_args, hpxml_kwargs = [], {}
    data, messages = hpxml.process(*hpxml_args, **hpxml_kwargs)

    # ### hpxml.process does not return the full list of measures
    # But since it's an audit, it means measures are already implemented
    # no way to get future measures unless we pull set from BEAM...?

    measures = []

    df = pd.DataFrame(measures, columns=MEASURE_DF_COLUMNS)

    measures_json = json.loads(df.to_json(orient="split"))
    return {'status': 'success',
            'message': '',
            'measure_data': measures_json}  # 'baseline_data':df_baseline}  TODO


def convert_to_kg(emission, starting_units, emission_rate):
    if not emission:
        return 0

    emission = float(emission)
    # Assume emission is in mmbtu, and convert to if starting units is kbtu
    if starting_units == 'kbtu':
        emission = emission / 1000
    return emission * emission_rate


def parse_buildingsync(file, elec_emission_rate, gas_emission_rate):
    """
    Parse file for Measures and Scenarios, and format similarly to parse_beam_measures
    """
    bs = BuildingSync()
    bs.import_file(file.open())
    bs_args, bs_kwargs = [], {}
    data, messages = bs.process(*bs_args, **bs_kwargs)
    if len(messages['errors']) > 0 or not data:
        return {'status': 'error', 'message': 'Failed to parse BuildingSync file'}

    if elec_emission_rate is None or gas_emission_rate is None:
        return {'status': 'error', 'message': 'Must pass valid fuel type emission rates.'}

    measures = []
    scenarios = data.get('scenarios', [])
    for m in data.get('measures', []):
        best_scenario = None
        elec_savings, ngas_savings = 0, 0
        # Filter scenarios by those with measure
        filtered_scenarios = [s for s in scenarios if m.get('property_measure_name') in s['measures']]

        # Get scenario that has the best savings
        for scenario in filtered_scenarios:
            elec_saving_diff, ngas_saving_diff = 0, 0
            annual_elec_savings = scenario.get('annual_electricity_energy')
            annual_ngas_savings = scenario.get('annual_natural_gas_energy')
            if annual_elec_savings is not None:
                elec_saving_diff = elec_savings - annual_elec_savings
            if annual_ngas_savings is not None:
                ngas_saving_diff = ngas_savings - annual_ngas_savings
            if elec_saving_diff + ngas_saving_diff >= 0:
                elec_savings = annual_elec_savings
                ngas_savings = annual_ngas_savings
                best_scenario = scenario

        if best_scenario is None:
            # TODO add message that measure was skipped?
            continue

        electricity_co2 = convert_to_kg(best_scenario.annual_electricity_energy, 'mmbtu', elec_emission_rate)
        gas_co2 = convert_to_kg(best_scenario.annual_natural_gas_energy, 'mmbtu', gas_emission_rate)
        bulk_cost = m.get('measure_total_first_cost', 0) + m.get('measure_installation_cost', 0) + m.get(
            'measure_material_cost', 0)
        bulk_cost = bulk_cost or None  # Change to None in case costs are not available

        measures.append([
            str(data.get('property_name')),                             # Building ID
            m.get('property_measure_name'),                             # Identifier
            m.get('name'),                                              # Description
            m.get('measure_total_first_cost'),                          # Cost
            best_scenario.get('annual_cost_savings'),                   # Annual_Saving
            best_scenario.get('name'),                                  # Scenario
            electricity_co2,                                            # Electricity CO2 (mmbtu)  to (kg)
            gas_co2,                                                    # Gas CO2 (mmbtu)  to (kg)
            electricity_co2 + gas_co2,                                  # Total CO2 use (mmbtu)  to (kg)
            best_scenario.get('annual_electricity_energy'),             # Electricity Savings (mmbtu)
            best_scenario.get('annual_natural_gas_energy'),             # Gas Savings (mmbtu)
            (best_scenario.annual_cost_savings or 0) / 2,               # Electricity Bill Savings
            (best_scenario.annual_cost_savings or 0) / 2,               # Gas Bill Savings
            m.get('category'),                                          # Measure Category Name
            m.get('mv_cost'),                                           # Cost Incremental
            bulk_cost,                                                  # Cost Bulk
            m.get('useful_life'),                                       # Measure Lifetime
        ])
    df_measures = pd.DataFrame(measures, columns=MEASURE_DF_COLUMNS)

    # Build dataframe with base information about the property. Stored in benchmark/baseline scenario
    # User will need to either create or edit it in in order to work properly
    baseline_scenario = [s for s in scenarios if s.get('name', '').lower() == 'baseline']
    if not baseline_scenario:
        return {'status': 'error',
                'message': 'No Baseline Scenario found for this Property. Please create.'}

    baseline_scenario = baseline_scenario[0]
    electricity_co2 = convert_to_kg(baseline_scenario.annual_electricity_energy, 'mmbtu', elec_emission_rate)
    gas_co2 = convert_to_kg(baseline_scenario.annual_natural_gas_energy, 'mmbtu', gas_emission_rate)
    data = [[
        str(data.get('property_name')),                                     # Building ID
        baseline_scenario.get('annual_cost_savings'),                       # Annual_Saving
        baseline_scenario.get('name'),                                      # Scenario
        electricity_co2,                                                    # Electricity CO2 (mmbtu)  to (kg)
        gas_co2,                                                            # Gas CO2 (mmbtu)  to (kg)
        electricity_co2 + gas_co2,                                          # Total CO2 use (mmbtu)  to (kg)
        baseline_scenario.get('annual_electricity_energy'),                 # Electricity Savings (mmbtu)
        baseline_scenario.get('annual_natural_gas_energy'),                 # Gas Savings (mmbtu)
    ]]

    df_baseline = pd.DataFrame(data, columns=BASELINE_DF_COLUMNS)

    measures_json = json.loads(df_measures.to_json(orient='split'))
    baseline_json = json.loads(df_baseline.to_json(orient='split'))
    return {'status': 'success',
            'message': '',
            'measure_data': measures_json,
            'baseline_data': baseline_json,
            'source': 'BuildingSync'}


def _reduce_scenarios(scenarios_data):
    """ Removes scenarios that overlap in measures. Criteria for picking between
        two Scenarios is defined by _worse_scenario.

        :param scenarios_data: serialized list of Scenario data
        :return: list of scenarios that do not have overlapping measures.
    """
    scenario_lookup = {s['id']: (s, set([m['display_name'] for m in s['measures']])) for s in scenarios_data}
    results = set()

    def _reduce_scenarios_rec(scenario_ids: list):
        if len(scenario_ids) == 0:
            return
        if len(scenario_ids) == 1:
            results.add(scenario_ids[0])
            return

        l_scenario, l_measures = scenario_lookup[scenario_ids[0]]
        worse_id = None
        for s_id in scenario_ids[1:]:
            r_scenario, r_measures = scenario_lookup[s_id]
            if not l_measures.isdisjoint(r_measures):
                worse_id = _worse_scenario(l_scenario, r_scenario)['id']
                break

        if worse_id is None:
            results.add(l_scenario['id'])
            scenario_ids = scenario_ids[1:]
        else:
            scenario_ids = [_id for _id in scenario_ids if _id != worse_id]

        _reduce_scenarios_rec(scenario_ids)

    _reduce_scenarios_rec([s['id'] for s in scenarios_data])
    return [scenario_lookup[s_id][0] for s_id in results]


def validate_emission_rates(emission_rates: dict) -> bool:
    """Validate that there are emission rates for each time series

    :param emission_rates: dict of {year: {fuel type: fuel rate}}

    :returns: bool
    """
    for year, rates in emission_rates.items():
        for fuel_type, fuel_rate in rates.items():
            if fuel_rate is None:
                return False
    return True


def parse_beam_measures(property_view_id: int, emission_rates: dict, timeline: list):
    """
    Parse BEAM set of Measures and Scenarios, when called through BEAM Analysis Framework

    :param int property_view_id: PropertyView object id
    :param dict emission_rates: mapping of emission rates to year
    :param list timeline: time steps used in emission rates

    :returns: dict of parsed measure and baseline data
    """
    pv = PropertyView.objects.select_related('state').get(id=property_view_id)
    if not pv:
        return {'status': 'error', 'message': 'Must pass property_view_id as data parameter'}
    state = pv.state

    if not validate_emission_rates(emission_rates):
        return {'status': 'error', 'message': 'Must pass valid fuel type emission rates.'}

    scenarios = Scenario.objects \
        .filter(property_state_id=state.id) \
        .prefetch_related("measures")
    scenarios = ScenarioSerializer(scenarios, many=True).data
    scenarios = _reduce_scenarios(scenarios)
    measures = []
    for scenario in scenarios:
        scenario_measures = scenario['measures']
        if len(scenario_measures) == 0:
            continue

        cost_total_first = 0
        cost_mv = 0
        cost_material = 0
        max_useful_life = 0
        for pm in scenario_measures:
            cost_total_first += (pm['cost_total_first'] or 0)
            cost_mv += (pm['cost_mv'] or 0)
            cost_material += (pm['cost_material'] or 0)
            max_useful_life = max(pm['useful_life'] or 0, max_useful_life)

        bulk_cost = (cost_total_first + cost_mv + cost_material) or None
        category_name = scenario['name'] if len(scenario_measures) > 1 else scenario_measures[0]['id']

        # Calculate CO2 at each time interval
        elec_co2_list = []
        gas_co2_list = []
        total_co2 = []
        for time in timeline:
            elec_rate = emission_rates['electricity'][time]
            gas_rate = emission_rates['natural_gas'][time]
            elec_co2 = convert_to_kg(scenario['annual_electricity_energy'], 'mmbtu', elec_rate)
            gas_co2 = convert_to_kg(scenario['annual_natural_gas_energy'], 'mmbtu', gas_rate)
            elec_co2_list.append(elec_co2)
            gas_co2_list.append(gas_co2)
            total_co2.append(elec_co2 + gas_co2)

        measures.append([
            str(property_view_id),                                              # Building ID
            scenario['id'],                                                     # Identifier
            scenario['name'],                                                   # Description
            cost_total_first,                                                   # Cost
            scenario['annual_cost_savings'],                                    # Annual_Saving
            scenario['name'] if scenario['name'] else 'N/A',                    # Scenario
            elec_co2_list,                                                      # Electricity CO2 (mmbtu) to (kg)
            gas_co2_list,                                                       # Gas CO2 (mmbtu) to (kg)
            total_co2,                                                          # Sum of Fuel Uses CO2 (kg)
            [scenario['annual_electricity_energy']] * len(timeline),            # Electricity Savings (mmbtu)
            [scenario['annual_natural_gas_energy']] * len(timeline),            # Gas Savings (mmbtu)
            (scenario['annual_cost_savings'] or 0) / 2,                         # Electricity Bill Savings
            (scenario['annual_cost_savings'] or 0) / 2,                         # Gas Bill Savings
            category_name,                                                      # Measure Category Name
            cost_mv,                                                            # Cost of Measurement and Validation
            bulk_cost,                                                          # Cost Bulk (Sum of all costs)
            max_useful_life,                                                    # Measure Lifetime
        ])

    df = pd.DataFrame(measures, columns=MEASURE_DF_COLUMNS)

    # Discard expensive measures with the same effect
    df = df.loc[df.fillna(np.inf).groupby(['Building', 'Annual_Saving']).Cost.idxmin().fillna(0).astype(int)]
    df.sort_index(inplace=True)

    # Discard less effective measures with the same cost
    df = df.loc[df.fillna(np.inf).groupby(['Building', 'Cost']).Annual_Saving.idxmax().fillna(0).astype(int)]
    df.sort_index(inplace=True)

    # ###  Build dataframe with base information about the property
    elec_rates = emission_rates['electricity']
    gas_rates = emission_rates['natural_gas']
    base_elec_rate = elec_rates[timeline[0]]
    base_gas_rate = gas_rates[timeline[0]]

    elec_use_co2 = convert_to_kg(state.extra_data.get('Electricity Use - Grid Purchase (kBtu)'), 'kbtu', base_elec_rate)
    gas_use_co2 = convert_to_kg(state.extra_data.get('Natural Gas Use (kBtu)'), 'kbtu', base_gas_rate)
    total_use_co2 = elec_use_co2 + gas_use_co2

    # missing_columns = []
    # for column in PROPERTY_STATE_REQUIRED_COLUMNS:
    #     if column not in state.extra_data:
    #         missing_columns.append(column)
    #
    # if missing_columns:
    #     return {
    #         'status': 'error',
    #         'message': 'The property is missing the following columns, please add %s' % ', '.join(missing_columns)
    #     }

    property_baseline_data = [[
        str(property_view_id),                                              # Building
        state.extra_data.get('Annual Savings', 0),                          # Annual_Saving
        elec_use_co2,                                                       # Electricity CO2 (kbtu) to (kg)
        gas_use_co2,                                                        # Gas CO2 (kbtu) to (kg)
        total_use_co2,                                                      # Total sum of above fuels
        state.extra_data.get('Electricity Savings (kbtu)', 0) / 1000,       # Electricity Savings (mmbtu)
        state.extra_data.get('Natural Gas Savings (kbtu)', 0) / 1000,       # Gas Savings (mmbtu)
        state.extra_data.get('Electricity Bill Savings', 0),                # Electricity Bill Savings
        state.extra_data.get('Natural Gas Bill Savings', 0),                # Gas Bill Savings
    ]]

    df_baseline = pd.DataFrame(property_baseline_data, columns=BASELINE_DF_COLUMNS)
    measures_json = json.loads(df.to_json(orient='split'))
    baseline_json = json.loads(df_baseline.to_json(orient='split'))
    return {
        'status': 'success',
        'message': '',
        'measure_data': measures_json,
        'baseline_data': baseline_json,
        'source': 'BEAM'
    }


# Hard-coded dictionary to reform RMI measures into BuildingSync naming
RMICODE_TO_BUILDINGSYNC = {
    'baseline': 'baseline',
    'E-DHW-01': {'category': 'other_hvac', 'name': 'install_air_source_heat_pump'},
    'E-DM-01': {'category': 'electrical_peak_shaving_load_shifting', 'name': 'other'},
    'E-DM-03': {'category': 'electrical_peak_shaving_load_shifting', 'name': 'other'},
    'E-DM-07': {'category': 'electrical_peak_shaving_load_shifting', 'name': 'other'},
    'E-DM-08': {'category': 'electrical_peak_shaving_load_shifting', 'name': 'other'},
    'E-ENV-01': {'category': 'building_envelope_modifications', 'name': 'increase_roof_insulation'},
    'E-ENV-02': {'category': 'building_envelope_modifications', 'name': 'increase_wall_insulation'},
    'E-ENV-04': {'category': 'building_envelope_modifications', 'name': 'replace_windows'},
    'E-ENV-05': {'category': 'building_envelope_modifications', 'name': 'add_window_films'},
    'E-ENV-10': {'category': 'building_envelope_modifications', 'name': 'other'},
    'E-ENV-11': {'category': 'building_envelope_modifications', 'name': 'other'},
    'E-ENV-13': {'category': 'building_envelope_modifications', 'name': 'other'},
    'E-HVAC-10': {'category': 'other_hvac', 'name': 'replace_package_units'},
    'E-HVAC-28': {'category': 'other_hvac', 'name': 'replace_or_modify_ahu'},
    'E-HVAC-53': {'category': 'other_hvac', 'name': 'other_cooling'},
    'E-L': {'category': 'lighting_improvements', 'name': 'retrofit_with_light_emitting_diode_technologies'},
    # For current RMI data all E-L category measures are LED measures. Needs updata if new data is provided
    'E-RE-01': {'category': 'renewable_energy_systems', 'name': 'install_photovoltaic_system'}
}


def parse_xlsx(file_measure, file_life, bld_list, sync_dict=RMICODE_TO_BUILDINGSYNC):
    """
    Parse measure info from RMI spreadsheet.
        1. 'measure_data' is a dataframe containing all measure relevant data for selected buildings, with columns
            [Category],[Name] being BuildingSync names (these two columns may not be necessary!)
        2. 'baseline_data' is a dataframe containing baseline information of selected buildings.
    """

    # Read measure life file
    data_life = pd.read_excel(file_life.temporary_file_path(), header=1)
    data_life = data_life.loc[:, ['Energy Analysis Measure #', 'Measure Name', 'Measure Lifetime (yrs)']]
    data_life.columns = ['Code', 'Measure', 'Life']

    # Read measure data file
    df = pd.read_excel(file_measure.temporary_file_path(), converters={'Building': str}, na_values=['-'])
    df = df.loc[:, ['Building', 'Measure', 'Measure Name', 'Net Cost', 'YR1 Energy Savings (not adj.)',
                    'Scenario', 'site:environmentalimpactelectricityco2emissionsmass[kg](timestep)',
                    'site:environmentalimpactnaturalgasco2emissionsmass[kg](timestep)',
                    'site:environmentalimpacttotalco2emissionscarbonequivalentmass[kg](timestep)',
                    'Electricity end use [kBTU]', 'Natural Gas end use [kBTU]',
                    'Electricity Consumption Tariff', 'Electricity Demand 1 Tariff', 'Natural Gas Tariff']]
    df.columns = ['Building', 'Code', 'Measure', 'Cost', 'Annual_Saving',
                  'Scenario', 'Electricity_CO2', 'Gas_CO2', 'Total_CO2', 'Electricity_Saving', 'Gas_Saving',
                  'Electricity_Bill_Saving', 'Electricity_Demand_Tariff', 'Gas_Bill_Saving']
    df.Code = df.Code.str.strip()
    df.Electricity_Bill_Saving = df.Electricity_Bill_Saving.fillna(0) + df.Electricity_Demand_Tariff.fillna(0)
    df.Gas_Bill_Saving.fillna(0, inplace=True)
    df = df.loc[:, [x != 'Electricity_Demand_Tariff' for x in df.columns]]

    # Parse baseline data
    df_baseline = df.loc[(df.Building.isin(bld_list)) & (df.Code == 'baseline')]
    df_baseline.reset_index(drop=True, inplace=True)
    df_baseline = df_baseline.loc[:, [x not in ['Code', 'Measure', 'Cost', 'Scenario'] for x in df_baseline.columns]]
    df_baseline.rename(columns={'Annual_Saving': 'Annual_Bill', 'Electricity_Saving': 'Electricity_Consumption',
                                'Gas_Saving': 'Gas_Consumption', 'Gas_Bill_Saving': 'Gas_Bill',
                                'Electricity_Bill_Saving': 'Electricity_Bill'}, inplace=True)

    # Parse measure data
    df = df.loc[(df.Building.isin(bld_list)) & (df.Code != 'baseline')]
    df = df.loc[df.Measure.notna()]
    ind = (abs(df.Annual_Saving) < 1e-8)
    if ind.any():
        warnings.warn("Incomplete Data: " + ', '.join(['Measure ' + x + ' of Building ' + y for x, y in
                                                       zip(df.Code[ind].values, df.Building[ind].values)]) +
                      " have no effect and are discarded.\n")
    df = df[~ (ind | (df.Measure.str.contains(pat='bundle')))]

    # Reformat bulk and incremental costs as extra columns called Cost_Incremental and Cost_Bulk
    ind_inc = df.Measure.str.contains(pat='incremental')
    ind_blk = df.Measure.str.contains(pat='bulk')
    df.Measure = df.Measure.str.replace('|'.join(['\(absolute\)', '\(incremental\)', '\(absolute bulk\)']), '',
                                        regex=True).str.strip()
    df = df[~(ind_inc | ind_blk)].merge(
        df[ind_inc][['Building', 'Code', 'Measure', 'Cost']], on=['Building', 'Code', 'Measure'], how='left',
        suffixes=('', '_Incremental')
    ).merge(
        df[ind_blk][['Building', 'Code', 'Measure', 'Cost']], on=['Building', 'Code', 'Measure'], how='left',
        suffixes=('', '_Bulk')
    )

    # Discard expensive measures with the same effect
    df = df.loc[df.fillna(np.inf).groupby(['Building', 'Annual_Saving']).Cost.idxmin().fillna(0).astype(int)]
    df.sort_index(inplace=True)
    # Discard less effective measures with the same cost
    df = df.loc[df.fillna(np.inf).groupby(['Building', 'Cost']).Annual_Saving.idxmax().fillna(0).astype(int)]
    df = df.apply(lambda x: x)
    df.sort_index(inplace=True)

    # Merge with life time
    df = df.merge(data_life.groupby('Code').Life.first(), on='Code', how='left')
    # Check if any life time is missing
    ind = pd.isna(df.Life)
    if ind.any():
        warnings.warn("Incomplete data: Missing life time of Measure " + ', '.join(
            df.Code[ind].drop_duplicates().values) + " is filled with similar measures.\n")
    df.loc[ind, 'Life'] = df[ind].set_index("Measure").rename(lambda s: ' '.join(s.split()[:3])).Life.fillna(
        data_life.set_index("Measure").rename(lambda s: ' '.join(s.split()[:3])).reset_index().drop_duplicates(
            subset='Measure', keep='first').set_index("Measure").Life).values
    if pd.isna(df.Life).any():
        warnings.warn("Incomplete data: Measure " + ', '.join(
            df.Code[pd.isna(df.Life)].drop_duplicates().values) + " lack life time and are discarded.\n")
        df = df[~df.Life.isna()]

    # Unify output dataframe format with other parse functions
    df.Measure = df.Measure + ' ' + df.Scenario.fillna('').astype(str).str.strip()
    df.rename(columns={'Code': 'Identifier', 'Measure': 'Description'}, inplace=True)
    df = df.loc[:, [x not in ['Scenario'] for x in df.columns]]
    # Check validity of identifier
    if df.duplicated(subset=['Building', 'Identifier']).any():
        return {'status': 'error', 'message': 'Duplicated identifier'}

    # Reform with BuildingSync naming convention
    # NOTE this may be unnecessary depending on how the other parse functions work and what output is desired
    df = df.groupby('Building', group_keys=False).apply(lambda x: reform_buildingsync_rmi(x, sync_dict))

    df = json.loads(df.to_json(orient="split"))
    df_baseline = json.loads(df_baseline.to_json(orient="split"))
    return {'status': 'success',
            'message': '',
            'measure_data': df,
            'baseline_data': df_baseline,
            'source': 'RMI'}


# =============================================== Auxillary Functions================================================
# Not sure if dictionary returns are needed in auxiliary functions, so raised exceptions

def reform_buildingsync_rmi(bld_df, sync_dict=RMICODE_TO_BUILDINGSYNC):
    """
    Add BuildingSync naming to parsed RMI data if necessary
    NOTE this may be unnecessary depending on how the other parse functions work and what output is desired
    """
    df = bld_df.copy()
    rough_codes = bld_df.Identifier.apply(group_identifier_rmi)
    ind = [x not in sync_dict.keys() for x in rough_codes]
    if any(ind):
        raise Exception("Unmathced measure found: " + bld_df.Identifier[ind] + " not in BuildingSync schema.\n")
    df['Category'] = [sync_dict[x]['category'] for x in rough_codes]
    df['Name'] = [sync_dict[x]['name'] for x in rough_codes]

    return df


def group_identifier_rmi(s):
    """
    Auxillary function. Group RMI measures by identifiers, all LED measures(E-L-02/03/04/06/07) grouped into one
    """
    tmp = '-'.join(s.split(sep='-')[:3])
    if tmp in ['E-L-02', 'E-L-03', 'E-L-04', 'E-L-06', 'E-L-07']:
        return 'E-L'
    else:
        return tmp


def _worse_scenario(left_scenario, right_scenario):
    """ Select the scenario with lower energy savings. If tie, return left scenario

    :param left_scenario: a serialized Scenario object
    :param right_scenario: a serialized Scenario object
    :return: whichever scenario is less favorable by the defined criteria
    """
    left_savings = left_scenario['annual_electricity_savings'] \
        + left_scenario['annual_natural_gas_savings']
    right_savings = right_scenario['annual_electricity_savings'] \
        + right_scenario['annual_natural_gas_savings']

    return left_scenario if left_savings <= right_savings else right_scenario

