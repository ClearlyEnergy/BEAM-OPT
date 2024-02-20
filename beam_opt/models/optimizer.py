# !/usr/bin/env python
# encoding: utf-8
"""
:copyright (c) 2014 - 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Department of Energy) and contributors. All rights reserved.  # NOQA
:author
"""

import json
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import itertools

from beam_opt.models.data_container import CompleteData

LOGGER = logging.getLogger(__name__)
ACCEPT_FIRST_SOLUTION = True
LOOKUP = {
    'Consumption': {
        'data': 'Total_Saving',
        'target': 'consumption_target',
        'optimize': 'Total_Consumption',
        'level': 'consumption_level',
        'reduction': 'annual_energy_saving',
        'electricity': 'Electricity_Saving',
        'gas': 'Gas_Saving',
        'baseline_electricity': 'Electricity_Consumption',
        'baseline_gas': 'Gas_Consumption',
    },
    'Emission': {
        'data': 'Total_CO2',
        'target': 'emission_target',
        'optimize': 'Total_CO2',
        'level': 'emission_level',
        'reduction': 'annual_emission_reduction',
        'electricity': 'Electricity_CO2',
        'gas': 'Gas_CO2',
        'baseline_electricity': 'Electricity_CO2',
        'baseline_gas': 'Gas_CO2',
    }
}


class Optimizer:

    def __init__(self, complete_data: CompleteData, scenario, bldg_id, timeline: list):
        bldg_id = [str(bldg_id)]
        if len(bldg_id) != 1:
            raise Exception("Received incorrect number of building ids")

        # Retrieve and store baseline data from complete_data
        baseline_result = complete_data.get_baseline_data(bldg_id)
        baseline_df = pd.read_json(json.dumps(baseline_result['building_data']), orient='split')
        self.baseline = baseline_df.copy()

        # Perform Baseline Preprocessing
        self.baseline['Total_Consumption'] = self.baseline.Electricity_Consumption + self.baseline.Gas_Consumption

        # This next line should be changed once more data available
        self.baseline['Year'] = timeline[0] + np.arange(0, self.baseline.shape[0])
        # Retrieve and store measures data from complete_data
        measures_result = complete_data.get_measure_data(bldg_id)
        measures_df = pd.read_json(json.dumps(measures_result['measure_data']), orient='split')
        self.measure_df = measures_df.copy()

        self.measure_df['Total_Saving'] = self.measure_df[['Electricity_Saving', 'Gas_Saving']].apply(
            lambda r: [i + j if i is not None and j is not None else None for i, j in zip(*r)], axis=1
        )

        self.measure_df['Total_Bill_Saving'] = self.measure_df.Electricity_Bill_Saving + self.measure_df.Gas_Bill_Saving
        self.measure_df = self.measure_df.sort_values(['Group', 'Index'], ascending=[True, True])
        self.measure_df = self.measure_df.reset_index(drop=True)

        # Retrieve and store Priority data from complete_data
        priority_results = complete_data.get_priority_chart(bldg_id)
        self.priority: pd.DataFrame = [*priority_results['priority_chart'].values()][0]

        self.total_years = timeline[-1] - timeline[0] + 1
        self.timeline = np.asarray(timeline)
        self.T = len(self.timeline)
        self.time_diff = np.diff(self.timeline)
        self.timeline_df = pd.DataFrame(np.arange(timeline[0], timeline[-1] + 1), columns=['Year'])

        FUEL_COLS = [LOOKUP[scenario]['electricity'], LOOKUP[scenario]['gas'], LOOKUP[scenario]['optimize']]
        self.baseline = self.baseline.explode(FUEL_COLS).reset_index(drop=True)
        self.baseline['Year'] = self.timeline_df
        self.baseline = self.baseline.set_index('Year')

        # Parameters set by set_parameters func
        self.delta = None
        self.budget = None
        self.penalty = None
        self.selected_df = None

        # Parameters set in optimize func
        self.selected_groups = None
        self.M = None
        self.Msub = None
        self.Xoptimal = None
        self.Xoptimal_ind = None
        self.total_cost = None
        self.solution = None

    def set_parameters(self, budget, target, penalty, delta, scenario='Consumption'):
        self.delta = delta
        self.budget = np.array(budget)
        self.penalty = penalty

        # Set target attribute
        target_df = pd.DataFrame({'Target': target, 'Year': self.timeline})
        target_df = target_df.merge(self.timeline_df, on='Year', how='right').fillna(method='ffill').set_index('Year')
        setattr(self, LOOKUP[scenario]['target'], target_df.Target * 1000)  # target_df.Target is in mtCO2e, convert to kg

        return {'status': 'success', 'message': ''}

    def _preselect(self, target_num=15, scenario='Consumption', discard_thres=1e-3):
        # Discard measures with negative electricity and gas savings
        for time in self.timeline:
            self.selected_df = self.measure_df[(self.measure_df[col_label_by_year(LOOKUP[scenario]['electricity'], time)] >= 0) |
                                               self.measure_df[col_label_by_year(LOOKUP[scenario]['gas'], time)] >= 0]

        # Discard measures whose annual saving is less than discard_thres * baseline expenditure
        if self.selected_df.shape[0] > target_num:
            self.selected_df = self.selected_df.loc[self.selected_df.Annual_Saving >=
                                                    (discard_thres * self.baseline.Annual_Bill.values.min())]

        # Select measures by cost efficiency within each exclusion group
        if self.selected_df.shape[0] > target_num:
            self.selected_df = self.selected_df.groupby('Group', group_keys=False).apply(
                lambda x: pick_cost_efficiency(x, self.timeline, scenario))

        # Select measures by saving/cost ratios
        ratio = np.empty(target_num)
        if self.selected_df.shape[0] > target_num:
            # TODO: with Total_CO2 being expanded, how to select? Using initial year as temp solution
            value = self.selected_df[col_label_by_year(LOOKUP[scenario]['data'], self.timeline[0])] * self.selected_df.Life
            ratio = value / self.selected_df.Cost
            ratio = ratio.sort_values(ascending=False).dropna()[:target_num]
            self.selected_df = self.selected_df.loc[ratio.index]

        # Check if any missing pre-requesite
        pos = target_num - 1
        for j in self.measure_df.Group:
            if j not in self.selected_df.Group:
                prereq_group = self.priority.index[self.priority.loc[:, j].notna()]
                # If there are, add the one with the highest saving/cost ratio by replacing the previously selected
                # measure with the lowest ratio
                if len(prereq_group) > 0 and not self.selected_df.Group.isin(prereq_group).any():
                    missing_inds = self.measure_df.Group.isin(prereq_group)
                    # TODO: expanded data handled the same as above
                    ratio_missing = self.measure_df[col_label_by_year(LOOKUP[scenario]['data'], self.timeline[0])][missing_inds]
                    ratio_missing = ratio_missing * self.measure_df.Life[missing_inds]
                    ratio_missing = (ratio_missing / self.measure_df.Cost[missing_inds]).sort_values(ascending=False).dropna()
                    if self.selected_df.Group[ratio.index[pos]] == j:
                        pos -= 1
                    self.selected_df.loc[ratio.index[pos], :] = self.measure_df.loc[ratio_missing.index[0], :]
                    pos -= 1

        # Store selected data
        self.selected_df = self.selected_df.sort_values(by=['Group', 'Index'], ascending=True).reset_index(drop=True)
        self.selected_groups = pd.Series(self.selected_df.Group.unique()).sort_values(ascending=True)
        self.M = len(self.selected_groups)                                  # number of exclusion groups
        self.Msub = self.selected_df.groupby("Group").Index.count()         # number of measures per exclusion group
        return {'status': 'success',
                'message': ''}

    def _prep(self, scenario):
        # State matrix
        indices_by_group = self.selected_df.groupby("Group").Index.apply(list)
        indices_by_group = [[0] + indices for indices in indices_by_group]
        self.Xmat = np.array(list(itertools.product(*indices_by_group)))
        self.ns = self.Xmat.shape[0]  # number of states

        # Exclude infeasible states by priority
        ind_priority = np.ones(self.ns, dtype=bool)
        common_indices = list(set(self.priority.index[self.priority.any(axis=1)]) & set(self.selected_groups))
        for i in common_indices:
            pos_i = self.selected_groups.index[self.selected_groups == i][0]
            common_cols = list(set(self.priority.columns[self.priority.iloc[i].notna()]) & set(self.selected_groups))
            for j in common_cols:
                pos_j = self.selected_groups.index[self.selected_groups == j][0]
                try:
                    ind1 = (self.Xmat[:, pos_i] > 0) | (self.Xmat[:, pos_j] == 0)
                except IndexError:
                    print("!!" + str([pos_i, pos_j]))
                ind_priority = ind_priority & ind1

        self.Xmat = self.Xmat[ind_priority]
        self.ns = self.Xmat.shape[0]
        # State matrix converted to indicator
        self.Xmat_ind = np.zeros([self.ns, np.sum(self.Msub)], dtype=bool)
        marks = np.insert(np.array(np.cumsum(self.Msub)), 0, 0)[:-1]
        df_tmp = pd.DataFrame(self.selected_df.groupby('Group').cumcount().values + 1, index=self.selected_df.Group,
                              columns=['Index_re'])
        Xmat_re = np.array(list(itertools.product(*[[0] + x for x in df_tmp.groupby("Group").Index_re.apply(list)])))
        for i, Xvec in enumerate(Xmat_re[ind_priority]):
            self.Xmat_ind[i, marks[Xvec > 0] + Xvec[Xvec > 0] - 1] = 1

        self.reduction_per_cycle = []
        for t, _ in enumerate(self.timeline):
            reduction_at_t = self.Xmat_ind @ self.selected_df[col_label_by_year(LOOKUP[scenario]['data'], self.timeline[t])]
            self.reduction_per_cycle.append(reduction_at_t)

        self.reduction_per_cycle = np.array(self.reduction_per_cycle)
        self.annual_bill_saving = self.Xmat_ind @ self.selected_df.Annual_Saving

    def _optimize(self, scenario='Consumption', target_only=False):
        """
        Perform optimization of measures with backwards recursion.

        NOTE: All cashflows are computed up to terminal time
        The argument target_only is not really used. I just set it here in case
        someone wants to know what if I don't care about the cost and saving
        cashflows, but only want to see if I can meet the target within budget,
        priority and start year constraints.
        """
        delta_n: np.ndarray[np.float64] = (self.delta ** self.selected_df.Life).to_numpy()
        all_indices: np.ndarray = np.arange(self.ns, dtype=np.int64)
        cost_inc: np.ndarray[np.int64] = self.selected_df.Cost_Incremental.fillna(self.selected_df.Cost).to_numpy()
        ind_priority, ind_feasible_list = self._feasible_states()
        penalty_per_state: np.ndarray = self._penalty_per_state(scenario)

        # Precompute feasibility due to start-year constraint
        if 'Start_Year' in self.measure_df.columns:
            start_years = self.selected_df.groupby('Group').Start_Year.first().values[None, :]
            unavailable_groups: npt.NDArray = start_years > self.timeline[:, None]
        else:
            unavailable_groups: npt.NDArray = np.repeat(False, self.T)

        V: np.ndarray[np.float32] = np.full((self.ns), np.inf, np.float32)  # Value function
        Xt_tuple = np.dtype([('decision', np.int64), ('penalty', np.float32)])
        initial_value = np.array((-1, 0), dtype=Xt_tuple)
        Xt_idx: np.ndarray = np.full([self.T, self.ns], initial_value)  # Optimal decision for each state and its penalty
        Vnext: np.ndarray[np.float32] = np.zeros(self.ns, dtype=np.float32)  # Terminal value function
        cost_values: np.ndarray[np.float64] = self.selected_df.Cost.to_numpy()

        # Backward recursion
        for t in range(self.T - 1, -1, -1):
            years_past = self.timeline[t] - self.timeline[0]
            current_cycle_len = self.time_diff[t] if t < len(self.time_diff) else 1
            current_cycle_range = range(current_cycle_len - 1, -1, -1)

            # Discount factor for discounting V_{t+1}
            disc = (self.delta ** self.time_diff[t] if t < self.T - 1 else 1)

            # Discount factor for discounting annual bill savings between t and t+1
            if t == self.T - 1:
                sum_disc = 1
            elif self.delta == 1:
                sum_disc = self.time_diff[t]
            else:
                sum_disc = (1 - self.delta ** self.time_diff[t]) / (1 - self.delta)

            # Discount factor for discounting all future incremental costs between t and T
            n_life = np.floor((self.timeline[-1] - self.timeline[t]) / self.selected_df.Life)
            if self.delta == 1:
                sum_disc_life = n_life
            else:
                sum_disc_life = delta_n * (1 - delta_n ** n_life) / (1 - delta_n)

            sum_disc_life: np.ndarray[np.float64] = sum_disc_life.to_numpy()
            discounted_cost = cost_inc * sum_disc_life
            budget = self.budget[t]
            unavail_factors = (self.Xmat[:, unavailable_groups[t]] == 0).all(axis=1)
            unavailable_group_t = unavailable_groups[t - 1]
            gas_usage = self._gas_usage_per_state(scenario, t)

            # Compute V_t for each state
            for i in range(self.ns):
                # Check if the state is possible at all by start year
                if t > 0 and (self.Xmat[i, unavailable_group_t] > 0).any():
                    continue

                # Choose feasible decision variables by start year
                if unavailable_groups[t]:
                    ind_feasible = ind_feasible_list[i] & unavail_factors
                else:
                    ind_feasible = ind_feasible_list[i]

                # Exclude infeasible decision variables by priority, budget, reduction
                ind_feasible = ind_feasible & ind_priority[i]
                ind_feasible = self._exclude_redundant_variables(ind_feasible, i, gas_usage)

                Xnew_ind = self.Xmat_ind[ind_feasible] & ~self.Xmat_ind[i]
                cost_per_state = Xnew_ind @ cost_values
                ind_cost = (cost_per_state <= budget)

                if not ind_cost.any():
                    continue

                # Index of feasible decision variables in self.Xmat (self.Xmat_ind)
                index_feasible = all_indices[ind_feasible][ind_cost]
                if len(index_feasible) == 0:
                    continue

                # Compute V_t values
                penalty_in_cycle = np.zeros_like(penalty_per_state[index_feasible, years_past])
                for y in current_cycle_range:
                    penalty_in_cycle = penalty_per_state[index_feasible, years_past + y] + self.delta * penalty_in_cycle

                # Add cost of next least expensive transition
                obj_vals = penalty_in_cycle + disc * Vnext[index_feasible]
                if not target_only:
                    costs_inc = self.Xmat_ind[index_feasible] @ (discounted_cost)  # Cost for replacement
                    obj_vals = obj_vals + cost_per_state[ind_cost] + costs_inc - sum_disc * self.annual_bill_saving[index_feasible]

                idx_min = obj_vals.argmin()
                Xt_idx[t, i]['decision'] = index_feasible[idx_min]
                Xt_idx[t, i]['penalty'] = penalty_in_cycle[idx_min]
                V[i] = obj_vals[idx_min]
                if t == 0:  # Note for t=0, only V(0) needs assessment
                    break

            Vnext = V

        # Forward recursion
        Xstar_idx = np.zeros(self.T, dtype=np.int64)
        Xstar_penalty = np.zeros(self.T, dtype=np.float32)
        for t in range(self.T):
            decision_tuple = Xt_idx[t, 0] if t == 0 else Xt_idx[t, Xstar_idx[t - 1]]
            Xstar_idx[t] = decision_tuple['decision']
            Xstar_penalty[t] = decision_tuple['penalty']

        # Output
        self.Xoptimal = self.Xmat[Xstar_idx]
        self.Xoptimal_ind = self.Xmat_ind[Xstar_idx]
        self.total_cost = V[0]
        result = self._calculate_forward_reduction(self.Xoptimal_ind, scenario)
        return result, Xstar_penalty.tolist()

    def _feasible_states(self):
        """
        """
        ind_priority: np.ndarray[np.bool_] = np.ones([self.ns, self.ns], dtype=bool)
        ind_feasible_list: list[list[bool]] = []
        selected_priority_df: pd.DataFrame = self.priority.loc[self.selected_groups, self.selected_groups]
        selected_priority_np: np.ndarray = selected_priority_df.to_numpy()
        ind_need_prereq: np.ndarray[np.bool_] = (~selected_priority_df.isna()).any(axis=0).to_numpy()
        for i in range(self.ns):
            ind_feasible = self.Xmat_ind | ~(self.Xmat_ind[i])
            ind_feasible = ind_feasible.all(axis=1) & ind_priority[i]
            ind_feasible_list.append(ind_feasible)
            ind_installed: np.ndarray[np.bool_] = self.Xmat.astype(bool)
            subind_unready = ~selected_priority_np[ind_installed[i]][:, ind_need_prereq].any(axis=0)
            for j in np.where(subind_unready)[0]:
                # groups having prerequisites and installed
                ind1 = ind_installed[:, (self.selected_groups == j)].reshape(-1)
                # groups that are prerequisites and installed
                ind2 = self.Xmat[:, selected_priority_np[j].notna()].any(axis=1)
                ind_priority[i, ind1 & (~ind2)] = False

        return ind_priority, ind_feasible_list

    def _exclude_redundant_variables(self, ind_feasible, state_i, gas_usage):
        """
        Remove feasible states when the transition does not reduce any fuel
        that state_i has not entirely reduced.

        :param ind_feasible: decision variables known to be feasible from state_i
        :param state_i: state index from which transition may occur
        :param gas_usage: gas usage per state (after reduction applied)
        """
        gas_delta = gas_usage - gas_usage[state_i]
        if gas_usage[state_i] > 0:
            return ind_feasible
        else:
            return ind_feasible & (gas_delta == 0)

    def _gas_usage_per_state(self, scenario, cycle_idx):
        """
        Precompute gas usage for each state in given year.

        :param str scenario:
        :param int cycle_idx:
        """
        year_idx = self.time_diff[:(cycle_idx + 1)].sum()
        baseline_gas = getattr(self.baseline, LOOKUP[scenario]['baseline_gas']).values[year_idx]
        gas_col_label = col_label_by_year(LOOKUP[scenario]['gas'], self.timeline[cycle_idx])
        gas_reduction = getattr(self.measure_df, gas_col_label).to_numpy()
        gas_reduction = np.sum(self.Xmat_ind * gas_reduction, axis=1)
        gas_usage = baseline_gas - gas_reduction
        return gas_usage

    def _penalty_per_state(self, scenario):
        """
        :param str scenario:
        :return NDArray: 2D array of penalties per year per state.
        The first axis of result is the state index, and the second axis is over
        every year of the timeline.
        """
        reduction_per_year = []
        for t, _ in enumerate(self.timeline):
            cycle_length = self.time_diff[t] if t < len(self.time_diff) else 1
            reduction = self.reduction_per_cycle[t]
            for _ in range(cycle_length):
                reduction_per_year.append(reduction)

        reduction_per_year = np.array(reduction_per_year)
        baseline_usage = getattr(self.baseline, LOOKUP[scenario]['optimize']).values[:,None]
        target_usage = np.array(getattr(self, LOOKUP[scenario]['target']))[:,None]
        excess = baseline_usage - reduction_per_year - target_usage
        excess = np.transpose(excess)
        penalty = np.where(excess > 0, excess, 0) * self.penalty
        if scenario == 'Emission':
            penalty = penalty / 1000

        return penalty

    def _calculate_forward_reduction(self, scenario_selection, scenario='Consumption'):
        """
        Perform forward calculation of energy reductions given a configuration of scenario installations.
        """

        data = {scenario: [], 'Electricity': [], 'Gas': [], 'Year': self.timeline}
        for t, time in enumerate(self.timeline):
            scenarios_for_t = scenario_selection[t]
            for name in ['data', 'electricity', 'gas']:
                store_as = scenario if name == 'data' else name.capitalize()
                data_by_t = self.measure_df[col_label_by_year(LOOKUP[scenario][name], time)]
                data[store_as].append((scenarios_for_t @ data_by_t.values.reshape([-1, 1])).reshape(-1)[0])
        scenario_df = pd.DataFrame(data).merge(self.timeline_df, on='Year', how='right')
        scenario_df = scenario_df.fillna(method='ffill').set_index('Year')

        # Collect what the new reduced emissions/consumption will be
        output_df = pd.DataFrame({
            'levels_reduced_to': self.baseline[LOOKUP[scenario]['optimize']] - scenario_df[scenario],
            'electricity_reduced_to': self.baseline[LOOKUP[scenario]['baseline_electricity']] - scenario_df['Electricity'],
            'gas_reduced_to': self.baseline[LOOKUP[scenario]['baseline_gas']] - scenario_df['Gas'],
        })

        setattr(self, LOOKUP[scenario]['level'], output_df)
        return {'solution': self.Xoptimal, 'objective': self.total_cost}

    def optimize(self,
                 scenario='Consumption',
                 target_num=15,
                 discard_thres=1e-3,
                 max_iter=None,
                 scenario_selection=None,
                 scenario_costs_savings=None,
                 measure_df=None):

        if max_iter is None:
            max_iter = target_num

        # Expand Elec/NatGas and Total C02/Savings column of lists into separate columns
        for column in [LOOKUP[scenario]['electricity'], LOOKUP[scenario]['gas'], 'Total_CO2', 'Total_Saving']:
            expanded_cols = [col_label_by_year(column, time) for time in self.timeline]
            self.measure_df[expanded_cols] = pd.DataFrame(self.measure_df[column].tolist(), index=self.measure_df.index)

        # Compute base scenario (install measures with maximal reducing power subject to budget constraint) at year 0
        first_year_data = col_label_by_year(LOOKUP[scenario]['data'], self.timeline[0])
        top_measures_by_group = self.measure_df.groupby('Group', group_keys=False)[first_year_data]
        df_base = self.measure_df.loc[top_measures_by_group.idxmax().values]
        df_base = df_base.sort_values(by=first_year_data)

        Xbase = np.zeros([self.T, df_base.shape[0]], dtype=int)  # base configuration
        obj_base = 0
        reducing_power = 0

        Cost_Inc = df_base.Cost_Incremental.fillna(df_base.Cost)
        self._preselect(target_num, scenario, discard_thres)
        delta_n = self.delta ** self.selected_df.Life

        baseline_optimize = getattr(self.baseline, LOOKUP[scenario]['optimize'])
        baseline_target = getattr(self, LOOKUP[scenario]['target'])
        for t in range(self.T):
            y_past = self.time_diff[:t].sum() if t > 0 else 0
            cost = 0
            # Compute discounting for future annual bill savings (to time t)
            if self.delta == 1:
                sum_disc = self.timeline[-1] - self.timeline[t]
            else:
                sum_disc = (1 - self.delta ** (self.timeline[-1] - self.timeline[t])) / (1 - self.delta)

            # Compute discounting for discounting all future incremental costs between t and T
            n_life = np.floor((self.timeline[-1] - self.timeline[t]) / self.measure_df.Life)
            sum_disc_life = n_life if self.delta == 1 else delta_n * (1 - delta_n ** n_life) / (1 - delta_n)

            # Check measures to be added
            for i in df_base.index:
                group_idx = df_base.Group[i]
                year_oob = ('Start_Year' in df_base.columns) and (df_base.loc[i, 'Start_Year'] > self.timeline[t])
                if Xbase[t, group_idx] > 0 or year_oob:
                    continue

                prereq_idx = self.priority.index[self.priority.loc[:, group_idx].notnull()]
                if len(prereq_idx) > 0 and not (Xbase[t, prereq_idx].astype(bool)).any():
                    continue

                if cost + df_base.Cost[i] > self.budget[t]:
                    continue

                cost = cost + df_base.Cost[i]
                Xbase[[s >= t for s in range(self.T)], group_idx] = df_base.Index[i]

                # Compute total cost (objective value)
                reducing_power += getattr(df_base, col_label_by_year(LOOKUP[scenario]['data'], self.timeline[t]))[i]
                time_delta = self.timeline[t] - self.timeline[0]
                excess_penalty = baseline_optimize.iloc[time_delta] - reducing_power - baseline_target.iloc[time_delta]
                excess_penalty = np.maximum(excess_penalty, 0)
                excess_penalty *= self.penalty * self.time_diff[t - 1]
                if not np.isinf(excess_penalty) and t < (self.T - 1):
                    for y in range(self.time_diff[t] - 1, -1, -1):
                        if np.isinf(excess_penalty):
                            break
                        excess = baseline_optimize.iloc[y_past + y] - reducing_power - baseline_target.iloc[y_past + y]
                        excess = np.maximum(excess, 0)
                        excess_penalty = self.delta * excess_penalty + excess * self.penalty * self.time_diff[t - 1]

                if np.isinf(excess_penalty):
                    obj_base = np.inf
                    break

                measure_cost = df_base.Cost[i] + sum_disc_life[i] * Cost_Inc[i] - \
                    sum_disc * df_base.Annual_Saving[i] + excess_penalty
                obj_base += self.delta ** (self.timeline[t] - self.timeline[0]) * measure_cost

        # Begin optimization
        for i in range(max_iter):
            self._prep(scenario)
            if scenario_selection:
                self.measure_df = measure_df
                self._forward(scenario_selection, scenario)
                self.Xoptimal_ind = np.array(scenario_selection)
                sol = self._compile_solution(filtered=True)
            else:
                _, penalties = self._optimize(scenario)
                self.penalties = penalties
                sol = self._compile_solution(filtered=False)

            self.solution = pd.DataFrame(sol)
            # If the suggested solution is no inferior to the base case, return as solution found
            # If a preconfigured scenario selection is provided, always return recalculated solution
            if scenario_selection or self.total_cost < obj_base or ACCEPT_FIRST_SOLUTION:
                return {'status': 'success', 'message': 'Solution found'}

            # If the suggested optimized solution is strictly worse than the base case, replace one candidate measure
            # with an un-preselected one and redo optimization

            measure_unchosen = list(set(self.selected_df.Identifier) -
                                    set([x for y in list(self.solution['New Measure']) for x in y]))

            # If there are not any more measures to consider, return found solution
            if len(measure_unchosen) == 0:
                return {'status': 'success', 'message': 'Solution found'}

            measure_ids = [ID in measure_unchosen for ID in self.selected_df.Identifier]
            # TODO: with Total_CO2 being expanded, how to select? Using initial year as temp solution
            data = col_label_by_year(LOOKUP[scenario]['data'], self.timeline[0])
            measure_to_reduce = self.selected_df.iloc[self.selected_df.loc[measure_ids][data].idxmin()]
            measure_to_reduce = measure_to_reduce.Identifier

            # Modify set of selected measures
            measure_unselected = list(set(self.measure_df.Identifier) - set(self.selected_df.Identifier))
            if measure_unselected:
                measure_ids = [ID in measure_unselected for ID in self.measure_df.Identifier]
                # TODO: with Total_CO2 being expanded, how to select? Using initial year as temp solution
                data = col_label_by_year(LOOKUP[scenario]['data'], self.timeline[0])
                measure_to_add = self.measure_df.iloc[self.measure_df.loc[measure_ids][data].idxmax()].Identifier
                self.selected_df = self.selected_df[self.selected_df.Identifier != measure_to_reduce]

                selected_measure_df = self.measure_df.loc[self.measure_df.Identifier == measure_to_add].iloc[0, :].copy(deep=True)
                self.selected_df = self.selected_df.append(selected_measure_df, ignore_index=True)
                self.selected_df = self.selected_df.sort_values(by=['Group', 'Index'], ascending=True)

            self.selected_groups = pd.Series(self.selected_df.Group.unique()).sort_values(ascending=True)
            self.M = len(self.selected_groups)
            self.Msub = self.selected_df.groupby("Group").Index.count()

        # If max iteration number is reached, return base case solution
        if np.isinf(obj_base):
            return {'status': 'failed',
                    'message': 'No solution found within the budget, priority and start year constraints'}

        self.Xoptimal = Xbase
        self.Xoptimal_ind = np.zeros([self.T, self.measure_df.shape[0]], dtype=bool)

        for t in range(self.T):
            for i in range(Xbase.shape[1]):
                if Xbase[t, i] > 0:
                    self.Xoptimal_ind[t, (self.measure_df.Group == i).values & (self.measure_df.Index == Xbase[t, i]).values] = True

        self.total_cost = obj_base
        sol = self._compile_solution(filtered=False)
        self.solution = pd.DataFrame(sol)

        # Format output DF
        data = {scenario: [], 'Electricity': [], 'Gas': [], 'Year': self.timeline}
        for t, time in enumerate(self.timeline):
            Xoptimal_ind_for_t = self.Xoptimal_ind[t]
            for name in ['data', 'electricity', 'gas']:
                store_as = scenario if name == 'data' else name.capitalize()
                data_by_t = self.measure_df[LOOKUP[scenario][name] + ' ' + str(time)]
                data[store_as].append((Xoptimal_ind_for_t @ data_by_t.values.reshape([-1, 1])).reshape(-1)[0])

        scenario_df = pd.DataFrame(data).merge(self.timeline_df, on='Year', how='right')
        scenario_df = scenario_df.fillna(method='ffill').set_index('Year')

        # Collect what the new reduced emissions/consumption will be
        output_df = pd.DataFrame({
            'levels_reduced_to': self.baseline[LOOKUP[scenario]['optimize']] - scenario_df[scenario],
            'electricity_reduced_to': self.baseline[LOOKUP[scenario]['baseline_electricity']] - scenario_df['Electricity'],
            'gas_reduced_to': self.baseline[LOOKUP[scenario]['baseline_gas']] - scenario_df['Gas'],
        })
        setattr(self, LOOKUP[scenario]['level'], output_df)

        return {'status': 'success', 'message': 'Max iteration number reached, returning base case solution'}

    def _compile_solution(self, filtered: bool):
        if filtered:
            measures = self.selected_df
        else:
            measures = self.measure_df

        sol = [
            {'Year': self.timeline[0],
             'New Measure': measures.Identifier[self.Xoptimal_ind[0]].tolist()}
        ] + [
            {'Year': self.timeline[t],
             'New Measure': measures.Identifier[self.Xoptimal_ind[t] & ~self.Xoptimal_ind[t - 1]].tolist()}
            for t in range(1, self.T)
        ]
        return sol

    def print_solution(self):
        # Retrieve the measures installed at each time point
        return getattr(self, 'solution', None)

    def get_solution(self):
        # Retrieve the measures installed at each time point
        return getattr(self, 'solution', None)

    def get_penalties(self):
        return self.penalties

    def get_level(self, scenario):
        """
        Retrieve new Consumption/Emission level based on baseline
        Retrieve the associated reduction from the measures data by Electricity and Natural Gas
        Retrieve the cost of implementing the measures at each interval
        """

        solution_df = self.get_solution()
        measures = solution_df['New Measure'].to_list()

        data = {'Year': self.timeline,
                'electricity_reduced_by': [0] * self.T,
                'gas_reduced_by': [0] * self.T,
                'electricity_cycle_reduction': [0] * self.T,
                'gas_cycle_reduction': [0] * self.T,
                'cost': [0] * self.T}
        total_gas = 0

        accumulated_measures = []
        cumulative_list = []
        for sub_lst in measures:
            cumulative_list.extend(sub_lst)
            accumulated_measures.append(cumulative_list.copy())

        for t, set_of_measures in enumerate(measures):
            set_of_measures_df = self.measure_df[self.measure_df['Identifier'].isin(set_of_measures)]
            elec_col = col_label_by_year(LOOKUP[scenario]['electricity'], self.timeline[t])
            gas_col = col_label_by_year(LOOKUP[scenario]['gas'], self.timeline[t])
            data['electricity_cycle_reduction'][t] = set_of_measures_df[elec_col].sum()
            data['gas_cycle_reduction'][t] = set_of_measures_df[gas_col].sum()
            data['cost'][t] = set_of_measures_df['Cost'].sum()
            current_set_of_measures = accumulated_measures[t]
            current_set_of_measures_df = self.measure_df[self.measure_df['Identifier'].isin(current_set_of_measures)]
            data['electricity_reduced_by'][t] += current_set_of_measures_df[elec_col].sum()
            data['gas_reduced_by'][t] = set_of_measures_df[gas_col].sum() + total_gas
            total_gas += set_of_measures_df[gas_col].sum()

        output_df = pd.DataFrame(data).merge(self.timeline_df, on='Year', how='right').fillna(method='ffill')
        output_df = output_df.set_index('Year').merge(getattr(self, LOOKUP[scenario]['level']), left_index=True, right_index=True)
        output_df['levels_reduced_by'] = output_df['electricity_reduced_by'] + output_df['gas_reduced_by']
        return output_df


# ================================== Auxillary functions ===========================
def pick_cost_efficiency(group_df, timeline, scenario):
    """
    Auxillary function for Optimizer._preselect. Discard measures with higher costs but strictly inferior
    energy saving/emission
    """
    if group_df.shape[0] == 1:
        return group_df

    tmp = group_df.sort_values(by='Cost', ascending=True)
    record = np.ones(tmp.shape[0], dtype=int)
    index = 1
    record[0] = tmp.index[0]

    # TODO: with Total_CO2 being expanded, how to select? Using initial year as temp solution
    year = ' ' + str(timeline[0])
    col = tmp[LOOKUP[scenario]['electricity'] + year] + tmp[LOOKUP[scenario]['gas'] + year]
    col *= tmp['Life']

    for i in range(1, tmp.shape[0]):
        if col[tmp.index[i]] >= col[tmp.index[index - 1]]:
            record[index] = tmp.index[i]
            index = index + 1

    return group_df.loc[record[:index]]


def col_label_by_year(col, year):
    return col + ' ' + str(year)
