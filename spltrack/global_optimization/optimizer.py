import logging
from typing import Dict

import numpy as np
import gurobipy as gp
from gurobipy import GRB


class GlobalOptimizer:
    def __init__(
        self,
        num_tracks: int = 10,
        time_limit: int = 120,
        inactive_patience: int = 1,
    ):
        self.num_tracks = num_tracks
        self.time_limit = time_limit
        self.inactive_patience = inactive_patience

    @classmethod
    def from_config(cls, cfg):
        go_cfg = cfg.GLOBAL_OPTIMIZER
        track_cfg = cfg.TRACKER
        return cls(
            go_cfg.NUM_TRACKS,
            go_cfg.TIME_LIMIT,
            track_cfg.INACTIVE_PATIENCE,
        )

    def _build_optimization_problem(
        self,
        tracklets,
        tracklet_to_track_costs,
        tracklet_to_tracklet_costs,
        incompatible_tracklet_track_pairs,
        incompatible_tracklet_tracklet_pairs,
    ) -> gp.Model:
        num_tracklets = len(tracklets)

        # Create a new model
        model = gp.Model("qp")
        model.setParam("MIPFocus", 1)
        model.setParam("TIME_LIMIT", self.time_limit)

        x = model.addVars(num_tracklets, self.num_tracks, vtype=GRB.BINARY)

        # Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
        obj = gp.QuadExpr()
        for c in tracklet_to_track_costs:
            target_tracklet = c[0]
            target_track = c[1]
            cost = c[2]
            obj.add(x[target_tracklet, target_track], cost)

        for c in tracklet_to_tracklet_costs:
            target_tracklet_1 = c[0]
            target_tracklet_2 = c[1]
            cost = c[2]
            for target_track in range(self.num_tracks):
                obj.add(
                    x[target_tracklet_1, target_track]
                    * x[target_tracklet_2, target_track],
                    cost,
                )

        model.setObjective(obj)

        # max one track per tracklet
        for i in range(num_tracklets):
            model.addConstr(gp.quicksum([x[i, k] for k in range(self.num_tracks)]) <= 1)

        # add constraints for incompatible tracklet-track pairs
        for i, j in incompatible_tracklet_track_pairs:
            model.addConstr(x[i, j] == 0)

        # add constraints for incompatible tracklet-tracklet pairs
        # i.e. these tracklet pairs cannot be assigned to the same track
        for i, j in incompatible_tracklet_tracklet_pairs:
            model.addConstrs(x[i, k] * x[j, k] == 0 for k in range(self.num_tracks))

        return x, model

    def optimize(
        self,
        tracklets,
        tracklet_to_track_costs,
        tracklet_to_tracklet_costs,
        incompatible_tracklet_track_pairs=[],
        incompatible_tracklet_tracklet_pairs=[],
    ) -> Dict[int, int]:

        x, model = self._build_optimization_problem(
            tracklets,
            tracklet_to_track_costs,
            tracklet_to_tracklet_costs,
            incompatible_tracklet_track_pairs,
            incompatible_tracklet_tracklet_pairs,
        )

        model.optimize()

        statuses = {
            GRB.LOADED: "LOADED",
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.CUTOFF: "CUTOFF",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INPROGRESS: "INPROGRESS",
            GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
            GRB.WORK_LIMIT: "WORK_LIMIT",
        }
        logging.info(f"Optimization status: {statuses[model.Status]}")

        result_msg = ""
        for k, v in x.items():
            if bool(v.X):
                result_msg += "Tracklet " + str(k[0]) + " => Track " + str(k[1]) + "\n"
        logging.info(result_msg)

        tracklet_to_track_dict = {}
        for i in range(len(tracklets)):
            matched = False
            for k in range(self.num_tracks):
                if bool(x[i, k].X):
                    tracklet_to_track_dict[i] = k
                    matched = True
            if not matched:
                tracklet_to_track_dict[i] = -1

        return tracklet_to_track_dict
