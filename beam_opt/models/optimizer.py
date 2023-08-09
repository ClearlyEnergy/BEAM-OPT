# !/usr/bin/env python
# encoding: utf-8
"""
:copyright (c) 2014 - 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Department of Energy) and contributors. All rights reserved.  # NOQA
:author
"""
import json
import numpy as np
import pandas as pd
import itertools

from beam_opt.models.data_container import CompleteData

class Optimizer:
    lookups = {
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

    def __init__(self, complete_data: CompleteData, bldg_id, timeline: list):
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
        self.baseline['Year'] = timeline[0]+np.arange(0, self.baseline.shape[0])

        # Retrieve and store measures data from complete_data
        measures_result = complete_data.get_measure_data(bldg_id)
        measures_df = pd.read_json(json.dumps(measures_result['measure_data']), orient='split')
        self.df = measures_df.copy()
        self.df['Total_Saving'] = self.df.Electricity_Saving + self.df.Gas_Saving
        self.df['Total_Bill_Saving'] = self.df.Electricity_Bill_Saving + self.df.Gas_Bill_Saving
        self.df = self.df.sort_values(['Group', 'Index'], ascending=[True, True])
        self.df.reset_index(inplace=True, drop=True)

        # Retrieve and store Priority data from complete_data
        priority_results = complete_data.get_priority_chart(bldg_id)
        self.priority = [*priority_results['priority_chart'].values()][0]

        self.total_years = timeline[-1] - timeline[0] + 1
        # if self.total_years!=self.baseline.shape[0]: # I'm not sure whether it's important to return error messages, but the initializer cannot return objects so I raised an exception
        #     asw=input("Received incorrect timeline or baseline data. Do you want to replicate/resample to fill baseline data? Y/N")
        #     if asw=='Y' or asw=='y':
        #         self.baseline=self.baseline.merge(pd.DataFrame(np.arange(timeline[0],timeline[-1]+1),columns=['Year']),how='right',on='Year').fillna(method='ffill')
        #     elif asw=='N' or asw=='n':
        #         raise Exception("Received incorrect timeline or baseline data: time horizon ("+str(self.total_years)+" years) doesn\'t match length of baseline projection ("+str(self.baseline.shape[0])+" years)")
        #     else:
        #         raise Exception("Invalid input: only Y/N allowed")
        self.timeline = np.asarray(timeline)  # time points
        self.timeline_df = pd.DataFrame(np.arange(timeline[0], timeline[-1] + 1), columns=['Year'])
        self.baseline = self.baseline.merge(self.timeline_df, how='right',
                                            on='Year').fillna(method='ffill').set_index('Year')
        self.T = len(self.timeline)

        # ### Initialize Parameters that will be set later
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
        lookup = self.lookups[scenario]

        self.delta = delta
        self.budget = np.array(budget)
        self.penalty = penalty

        # Set target attribute
        target_df = pd.DataFrame({'Target': target, 'Year': self.timeline})
        target_df = target_df.merge(self.timeline_df, on='Year', how='right').fillna(method='ffill').set_index('Year')
        setattr(self, lookup['target'], target_df.Target * self.baseline[lookup['optimize']].iloc[0])

        return {'status': 'success', 'message': ''}

    def _preselect(self, target_num=16, scenario='Consumption', discard_thres=1e-3):
        # Discard measures with negative electricity and gas savings
        lookup = self.lookups[scenario]
        self.selected_df = self.df[(self.df[lookup['electricity']] >= 0) | self.df[lookup['gas']] >= 0]

        # Pick measures meeting certain financial standards
        ratio = np.empty(target_num)
        if self.selected_df.shape[0] > target_num:
            # Discard measures whose annual saving is less than discard_thres * baseline expenditure
            self.selected_df = self.selected_df.loc[self.selected_df.Annual_Saving >=
                                                    (discard_thres * self.baseline.Annual_Bill.values.min())]
            if self.selected_df.shape[0] > target_num:
                # Select measures by cost efficiency within each exclusion group
                self.selected_df = self.selected_df.groupby('Group', group_keys=False).apply(
                    lambda x: pick_cost_efficiency(x, scenario))
                if self.selected_df.shape[0] > target_num:
                    # Select measures by saving/cost ratios
                    ratio = (self.selected_df[lookup['data']] * self.selected_df.Life / self.selected_df.Cost)
                    ratio = ratio.sort_values(ascending=False).dropna()[:target_num]
                    self.selected_df = self.selected_df.loc[ratio.index]

        # Check if any missing prerequesite
        pos = target_num - 1
        for j in self.df.Group:
            if j in self.selected_df.Group:
                prereq_group = self.priority.index[self.priority.loc[:, j].notna()]
                # If there are, add the one with the highest saving/cost ratio by replacing the previously selected
                # measure with the lowest ratio
                if len(prereq_group) > 0 and not self.selected_df.Group.isin(prereq_group).any():
                    missing_inds = self.df.Group.isin(prereq_group)
                    ratio_missing = self.df[lookup['data']][missing_inds] * self.df.Life[missing_inds]
                    ratio_missing = (ratio_missing / self.df.Cost[missing_inds]).sort_values(ascending=False).dropna()
                    if self.selected_df.Group[ratio.index[pos]] == j:
                        pos -= 1
                    self.selected_df.loc[ratio.index[pos], :] = self.df.loc[ratio_missing.index[0], :]
                    pos -= 1

        # Store selected data
        self.selected_df = self.selected_df.sort_values(by=['Group', 'Index'], ascending=True).reset_index(drop=True)
        self.selected_groups = pd.Series(self.selected_df.Group.unique()).sort_values(ascending=True)
        self.M = len(self.selected_groups)                                  # number of exclusion groups
        self.Msub = self.selected_df.groupby("Group").Index.count()         # number of measures per exclusion group
        return {'status': 'success',
                'message': ''}

    def _prep(self):
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

        # Pre-computation
        self.annual_bill_saving = self.Xmat_ind @ self.selected_df.Annual_Saving
        self.annual_energy_saving = self.Xmat_ind @ self.selected_df.Total_Saving
        self.annual_emission_reduction = self.Xmat_ind @ self.selected_df.Total_CO2

    def _optimize(self, scenario='Consumption', target_only=False):
        """
        Note: All cashflows are computed up to terminal time
        The argument target_only is not really used. I just set it here in case someone wants to know what if I don't
        care about the cost and saving cashflows, but only want to see if I can meet the target within budget, priority
        and start year constraints
        """
        lookup = self.lookups[scenario]
        excess = getattr(self.baseline, lookup['optimize']).values[None, :] - getattr(self, lookup['reduction'])[:, None]
        excess = excess - np.array(getattr(self, lookup['target']))[None, :]

        excess_payment = np.zeros([self.ns, self.total_years])
        excess_payment[excess > 0] = excess[excess > 0] * self.penalty

        time_diff = np.diff(self.timeline)
        delta_n = self.delta ** self.selected_df.Life
        all_indices = np.arange(self.ns, dtype=np.int64)
        Cost_Inc = self.selected_df.Cost_Incremental.fillna(self.selected_df.Cost)

        # Precompute feasible states by priority
        selected_priority = self.priority.loc[self.selected_groups, self.selected_groups]
        ind_priority = np.ones([self.ns, self.ns], dtype=bool)
        ind_needPrereq = (~selected_priority.isna()).any(axis=0)
        ind_installed = self.Xmat.astype(bool)
        for i in range(self.ns):
            subind_unready = ~selected_priority.loc[ind_installed[i], ind_needPrereq].any(axis=0)
            ind_unready = pd.Series(np.zeros(len(self.selected_groups), dtype=bool), index=self.selected_groups)
            ind_unready[subind_unready.index[subind_unready]] = True
            for j in subind_unready.index[subind_unready]:
                # ind1 is indicator for groups having prerequisites and installed
                ind1 = ind_installed[:, (self.selected_groups == j)].reshape(-1)
                # ind2 is indicator for groups that are prerequisites and installed
                ind2 = self.Xmat[:, selected_priority[j].notna()].any(axis=1)
                ind_priority[i, ind1 & (~ind2)] = False

        # Precompute feasibility due to start-year constraint
        if 'Start_Year' in self.df.columns:
            unavailable_groups = self.selected_df.groupby('Group').Start_Year.first().values[None, :] > self.timeline[:, None]
        else:
            unavailable_groups = np.repeat(False, len(self.selected_groups))

        V = np.zeros(self.ns)  # Value function
        Xt_idx = np.zeros([self.T, self.ns], dtype=np.int64)  # Optimal decision for each state
        Vnext = np.zeros(self.ns)  # Terminal value function
        # Backward recursion
        for t in range(self.T - 1, -1, -1):
            # Discount factor for discounting V_{t+1}
            disc = (self.delta ** time_diff[t] if t < self.T - 1 else 1)
            # Discount factor for discounting annual bill savings between t and t+1
            if t == self.T - 1:
                sum_disc = 1
            elif self.delta == 1:
                sum_disc = time_diff[t]
            else:
                sum_disc = (1 - self.delta ** time_diff[t]) / (1 - self.delta)
            y_past = time_diff[:t].sum() if t > 0 else 0
            # Discount factor for discounting all future incremental costs between t and T
            n_life = np.floor((self.timeline[-1] - self.timeline[t]) / self.selected_df.Life)
            sum_disc_life = n_life if self.delta == 1 else delta_n * (1 - delta_n ** n_life) / (1 - delta_n)
            # Compute V_t for each state
            for i in range(self.ns):
                # Check if the state is possible at all by start year
                if t > 0 and (self.Xmat[i, unavailable_groups[t - 1]] > 0).any():
                    V[i] = np.inf
                    Xt_idx[t, i] = -1  # Use -1 as indicator of null
                    continue
                # State converted to indicator
                Xprev_ind = self.Xmat_ind[i]
                # Choose feasible decision variables by definition
                ind_feasible = (self.Xmat_ind | ~Xprev_ind).all(axis=1)
                # Choose feasible decision variables by start year
                # if unavailable_groups[t].any():  TODO Temp Bug fix, figure out solution later
                if t < len(unavailable_groups) and unavailable_groups[t].any():
                    ind_feasible = ind_feasible & (self.Xmat[:, unavailable_groups[t]] == 0).all(axis=1)
                # Exclude infeasible decision variable by priority
                ind_feasible = ind_feasible & ind_priority[i]
                # print([self.Xmat[i],ind_priority[i].sum()])
                # Exclude infeasible decision variable by budget
                Xnew_ind = self.Xmat_ind[ind_feasible] & ~Xprev_ind
                costs = Xnew_ind @ self.selected_df.Cost
                ind_cost = (costs <= self.budget[t])
                if not ind_cost.any():
                    V[i] = np.inf
                    Xt_idx[t, i] = -1
                    continue

                # Index of feasible decision variables in self.Xmat (self.Xmat_ind)
                idx_feasible = all_indices[ind_feasible][ind_cost]
                # Compute V_t values
                obj_vals = excess_payment[idx_feasible, self.timeline[t] - self.timeline[0]]
                if t < self.T - 1:
                    for y in range(time_diff[t] - 1, -1, -1):
                        obj_vals = excess_payment[idx_feasible, y_past + y] + self.delta * obj_vals
                obj_vals = obj_vals + disc * Vnext[idx_feasible]
                if not target_only:
                    costs_inc = self.Xmat_ind[idx_feasible] @ (Cost_Inc * sum_disc_life)  # Cost for replacement
                    obj_vals = obj_vals + costs[ind_cost] + costs_inc - sum_disc * self.annual_bill_saving[idx_feasible]
                idx_min = obj_vals.argmin()
                Xt_idx[t, i] = idx_feasible[idx_min]
                V[i] = obj_vals[idx_min]
                if t == 0:  # Note for t=0, only V(0) needs assessment
                    break
            Vnext = V

        # Forward recursion
        Xstar_idx = np.zeros(self.T, dtype=np.int64)
        for t in range(self.T):
            Xstar_idx[t] = Xt_idx[t, 0] if t == 0 else Xt_idx[t, Xstar_idx[t - 1]]

        # Output
        self.Xoptimal = self.Xmat[Xstar_idx]
        self.Xoptimal_ind = self.Xmat_ind[Xstar_idx]
        self.total_cost = V[0]
        return self._forward(self.Xoptimal_ind, self.df, scenario)
    
    def _forward(self, scenario_selection, measure_df, scenario='Consumption'):
        """
        Perform forward calculation of energy reductions given a configuration of scenario installations.
        """
        lookup = self.lookups[scenario]
        # from remote_pdb import RemotePdb; RemotePdb('0.0.0.0', 6666).set_trace()

        # Gather the overall reduction, and the reduction for Electricity and Gas
        scenario_df = pd.DataFrame(
            {scenario: (scenario_selection @ measure_df[lookup['data']].values.reshape([-1, 1])).reshape(-1),
             'Electricity': (scenario_selection @ measure_df[lookup['electricity']].values.reshape([-1, 1])).reshape(-1),
             'Gas': (scenario_selection @ measure_df[lookup['gas']].values.reshape([-1, 1])).reshape(-1),
             'Year': self.timeline}
        ).merge(self.timeline_df, on='Year', how='right').fillna(method='ffill').set_index('Year')

        # Collect what the new reduced emissions/consumption will be
        output_df = pd.DataFrame()
        output_df['levels_reduced_to'] = self.baseline[lookup['optimize']] - scenario_df[scenario]
        output_df['electricity_reduced_to'] = self.baseline[lookup['baseline_electricity']] - scenario_df['Electricity']
        output_df['gas_reduced_to'] = self.baseline[lookup['baseline_gas']] - scenario_df['Gas']
        setattr(self, lookup['level'], output_df)
        
        return {'solution': self.Xoptimal, 'objective': self.total_cost}

    def optimize(self, scenario='Consumption', target_num=16, discard_thres=1e-3, max_iter=None, scenario_selection=None, scenario_costs_savings=None, measure_df=None):
        lookup = self.lookups[scenario]

        if max_iter is None:
            max_iter = target_num

        # Compute base scenario (install measures with maximal reducing power subject to budget constraint)
        df_base = self.df.loc[getattr(self.df.groupby('Group', group_keys=False), lookup['data']).idxmax().values]
        df_base = df_base.sort_values(by=lookup['data'])

        time_diff = np.diff(self.timeline)

        Xbase = np.zeros([self.T, df_base.shape[0]], dtype=int)  # base configuration
        obj_base = 0
        reducing_power = 0
        Cost_Inc = df_base.Cost_Incremental.fillna(df_base.Cost)
        self._preselect(target_num, scenario, discard_thres)
        delta_n = self.delta ** self.selected_df.Life

        for t in range(self.T):
            y_past = time_diff[:t].sum() if t > 0 else 0
            cost = 0
            # Compute discounting for future annual bill savings (to time t)
            if self.delta == 1:
                sum_disc = self.timeline[-1] - self.timeline[t]
            else:
                sum_disc = (1 - self.delta ** (self.timeline[-1] - self.timeline[t])) / (1 - self.delta)

            # Compute discounting for discounting all future incremental costs between t and T
            n_life = np.floor((self.timeline[-1] - self.timeline[t]) / self.df.Life)
            sum_disc_life = n_life if self.delta == 1 else delta_n * (1 - delta_n ** n_life) / (1 - delta_n)

            # Check measures to be added
            for i in df_base.index:
                group_idx = df_base.Group[i]
                year_oob = ('Start_Year' in df_base.columns and df_base.loc[i, 'Start_Year'] > self.timeline[t])
                if Xbase[t, group_idx] > 0 or year_oob:
                    continue

                prereq_idx = self.priority.index[self.priority.loc[:, group_idx].notnull()]
                if len(prereq_idx) == 0 or (Xbase[t, prereq_idx].astype(bool)).any():
                    # Choose new measure to install
                    if cost + df_base.Cost[i] > self.budget[t]:
                        continue

                    cost = cost + df_base.Cost[i]
                    Xbase[[s >= t for s in range(self.T)], group_idx] = df_base.Index[i]

                    # Compute total cost (objective value)
                    reducing_power += getattr(df_base, lookup['data'])[i]
                    time_delta = self.timeline[t] - self.timeline[0]
                    excess_penalty = np.maximum(getattr(self.baseline, lookup['optimize']).iloc[time_delta] -
                                                reducing_power - getattr(self, lookup['target']).iloc[time_delta], 0)
                    excess_penalty *= self.penalty
                    if not np.isinf(excess_penalty) and t < (self.T - 1):
                        for y in range(time_diff[t] - 1, -1, -1):
                            if np.isinf(excess_penalty):
                                break
                            excess = np.maximum(getattr(self.baseline, lookup['optimize']).iloc[y_past + y] -
                                                reducing_power - getattr(self, lookup['target']).iloc[y_past + y], 0)
                            excess_penalty = self.delta * excess_penalty + excess * self.penalty

                    if np.isinf(excess_penalty):
                        obj_base = np.inf
                        break

                    measure_cost = df_base.Cost[i] + sum_disc_life[i] * Cost_Inc[i] - \
                                   sum_disc * df_base.Annual_Saving[i] + excess_penalty
                    obj_base += self.delta ** (self.timeline[t] - self.timeline[0]) * measure_cost

        # Begin optimization
        for i in range(max_iter):
            self._prep()
            if scenario_selection:
                self.df = measure_df
                self._forward(scenario_selection, measure_df, scenario)
                self.Xoptimal_ind = np.array(scenario_selection)
                sol = [
                    {'Year': self.timeline[0], 'New Measure': self.df.Identifier[self.Xoptimal_ind[0]].tolist()}
                ] + [
                    {'Year': self.timeline[t],
                     'New Measure': self.df.Identifier[self.Xoptimal_ind[t] & ~self.Xoptimal_ind[t - 1]].tolist()}
                    for t in range(1, self.T)
                ]
            else:
                self._optimize(scenario)
                sol = [
                    {'Year': self.timeline[0], 'New Measure': self.selected_df.Identifier[self.Xoptimal_ind[0]].tolist()}
                ] + [
                    {'Year': self.timeline[t],
                     'New Measure': self.selected_df.Identifier[self.Xoptimal_ind[t] & ~self.Xoptimal_ind[t - 1]].tolist()}
                    for t in range(1, self.T)
                ]
            self.solution = pd.DataFrame(sol)
            # If the suggested solution is no inferior to the base case, return as solution found
            if scenario_selection or self.total_cost < obj_base:
                return {'status': 'success', 'message': 'Solution found'}
            # If the suggested optimized solution is strictly worse than the base case, replace one candidate measure
            # with an un-preselected one and redo optimization
           
            measure_unchosen = list(set(self.selected_df.Identifier) -
                                    set([x for y in list(self.solution['New Measure']) for x in y]))
            
            if len(measure_unchosen) == 0:
                return {'status': 'success', 'message': 'Solution found'}
            
            measure_ids = [ID in measure_unchosen for ID in self.selected_df.Identifier]
            measure_to_reduce = self.selected_df.iloc[self.selected_df.loc[measure_ids][lookup['data']].idxmin()]
            measure_to_reduce = measure_to_reduce.Identifier

            # Modify set of selected measures
            measure_unselected = list(set(self.df.Identifier) - set(self.selected_df.Identifier))
            if measure_unselected:
                measure_ids = [ID in measure_unselected for ID in self.df.Identifier]
                measure_to_add = self.df.iloc[self.df.loc[measure_ids][lookup['data']].idxmax()].Identifier
                self.selected_df = self.selected_df[self.selected_df.Identifier != measure_to_reduce]

                selected_measure_df = self.df.loc[self.df.Identifier == measure_to_add].iloc[0, :].copy(deep=True)
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
        self.Xoptimal_ind = np.zeros([self.T, self.df.shape[0]], dtype=bool)
        for t in range(self.T):
            for i in range(Xbase.shape[1]):
                if Xbase[t, i] > 0:
                    self.Xoptimal_ind[t, (self.df.Group == i).values & (self.df.Index == Xbase[t, i]).values] = True

        self.total_cost = obj_base
        sol = [
            {'Year': self.timeline[0], 'New Measure': self.df.Identifier[self.Xoptimal_ind[0]].tolist()}
        ] + [
            {'Year': self.timeline[t],
             'New Measure': self.df.Identifier[self.Xoptimal_ind[t] & ~self.Xoptimal_ind[t - 1]].tolist()
             } for t in range(1, self.T)
        ]
        self.solution = pd.DataFrame(sol)

        # Format output DF
        scenario_df = pd.DataFrame(
            {scenario: (self.Xoptimal_ind @ self.df[lookup['data']].values.reshape([-1, 1])).reshape(-1),
             'Electricity': (self.Xoptimal_ind @ self.df[lookup['electricity']].values.reshape([-1, 1])).reshape(-1),
             'Gas': (self.Xoptimal_ind @ self.df[lookup['gas']].values.reshape([-1, 1])).reshape(-1),
             'Year': self.timeline}
        ).merge(self.timeline_df, on='Year', how='right').fillna(method='ffill').set_index('Year')

        output_df = pd.DataFrame()
        output_df['levels_reduced_to'] = self.baseline[lookup['optimize']] - scenario_df[scenario]
        output_df['electricity_reduced_to'] = self.baseline[lookup['baseline_electricity']] - scenario_df['Electricity']
        output_df['gas_reduced_to'] = self.baseline[lookup['baseline_gas']] - scenario_df['Gas']
        setattr(self, lookup['level'], output_df)

        return {'status': 'success', 'message': 'Max iteration number reached, returning base case solution'}

    def print_solution(self):
        # Retrieve the measures installed at each time point
        return getattr(self, 'solution', None)

    def get_solution(self):
        # Retrieve the measures installed at each time point
        return getattr(self, 'solution', None)

    def get_level(self, scenario):
        # Retrieve new Consumption/Emission level based on baseline
        # Retrieve the associated reduction from the measures data by Electricity and Natural Gas
        # Retrieve the cost of implementing the measures at each interval
        lookup = self.lookups[scenario]
        solution_df = self.get_solution()
        measures = solution_df['New Measure'].to_list()

        data = {'Year': self.timeline,
                'electricity_reduced_by': [0] * len(self.timeline),
                'gas_reduced_by': [0] * len(self.timeline),
                'electricity_cycle_reduction': [0] * len(self.timeline),
                'gas_cycle_reduction': [0] * len(self.timeline),
                'cost': [0] * len(self.timeline)}
        total_elec, total_gas = 0, 0
        for year_index, set_of_measures in enumerate(measures):
            set_of_measures_df = self.df[self.df['Identifier'].isin(set_of_measures)]

            data['electricity_cycle_reduction'][year_index] = set_of_measures_df[lookup['electricity']].sum()
            data['gas_cycle_reduction'][year_index] = set_of_measures_df[lookup['gas']].sum()
            data['cost'][year_index] = set_of_measures_df['Cost'].sum()

            data['electricity_reduced_by'][year_index] = set_of_measures_df[lookup['electricity']].sum() + total_elec
            data['gas_reduced_by'][year_index] = set_of_measures_df[lookup['gas']].sum() + total_gas
            total_elec += set_of_measures_df[lookup['electricity']].sum()
            total_gas += set_of_measures_df[lookup['gas']].sum()

        output_df = pd.DataFrame(data).merge(self.timeline_df, on='Year', how='right').fillna(method='ffill')
        output_df = output_df.set_index('Year').merge(getattr(self, lookup['level']), left_index=True, right_index=True)
        output_df['levels_reduced_by'] = output_df['electricity_reduced_by'] + output_df['gas_reduced_by']
        return output_df


# ================================== Auxillary functions ===========================
def pick_cost_efficiency(group_df, scenario='Consumption'):
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
    col = (tmp.Electricity_Saving + tmp.Gas_Saving) if scenario == 'Consumption' else tmp.Total_CO2
    col *= tmp.Life

    for i in range(1, tmp.shape[0]):
        if col[tmp.index[i]] >= col[tmp.index[index - 1]]:
            record[index] = tmp.index[i]
            index = index + 1
    return group_df.loc[record[:index]]
