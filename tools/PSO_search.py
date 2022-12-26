import numpy as np
from tools.gridsearch import *
import pickle


class PSO_grid:
    def __init__(
        self,
        args,
    ):
        cfg = get_default_cfg()
        cfg.merge_from_file(str(args.config_file))
        self.cfg = cfg
        output_dir = args.output_dir
        self.gridsearch_iteration_fn = functools.partial(
            compute_metrics,
            base_cfg=cfg,
            dataset_dir=args.dataset_dir,
            root_output_dir=output_dir,
        )
        self.params_to_consider = [
            "SELF_LOCALIZATION_COST_WEIGHT",
            "TEAM_DETECTION_COST_WEIGHT",
            "TRACKLET_LIFETIME_COST_WEIGHT",
            "FALLEN_EVENTS_COST_WEIGHT",
            "TRACKLET_TO_TRACKLET_COST_WEIGHT",
            "UNARY_COSTS_OFFSET",
            "PAIRWISE_COSTS_OFFSET",
        ]
        self.total_values = []
        self.args = args

    def convert_param(self, x):
        diction = {
            k: float(v) for k, v in zip(self.params_to_consider, x) if "OFFSET" not in k
        }
        if self.args.use_offsets_pso:
            diction["UNARY_COSTS_OFFSET"] = float(x[-2])
            diction["PAIRWISE_COSTS_OFFSET"] = float(x[-1])
        else:
            diction["UNARY_COSTS_OFFSET"] = self.args.offset_uni
            diction["PAIRWISE_COSTS_OFFSET"] = self.args.offset_pair
        return diction

    def convert_params(self, x):
        params = [self.convert_param(i) for i in x]
        return params

    def calculate_values(self, x):
        params = self.convert_params(x)
        total_values = []
        total_values = Parallel(n_jobs=self.args.jobs)(
            delayed(self.gridsearch_iteration_fn)(params=param) for param in params
        )
        for idx, value in enumerate(total_values):
            self.total_values.append((params[idx], value))
        values = -np.array([v["average_mpia"] for v in total_values])
        print(values)
        return values.reshape(-1, 1)


# hyperparameters
def _parse_args_new():
    """
    The different arguments for the parser
    """
    parser = get_default_argument_parser()
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=64,
        help="Number of jobs to run in parallel",
    )
    parser.add_argument(
        "--offset_uni", type=float, default=-0.16666, help="Offset for uninary terms"
    )
    parser.add_argument(
        "--offset_pair", type=float, default=-0.05555, help="Pariwise offset"
    )
    parser.add_argument(
        "--num_feat",
        type=int,
        default=5,
        help="Number of dimensions each particle has",
    )
    parser.add_argument(
        "--num_units",
        type=int,
        default=20,
        help="Number of particles for the run",
    )
    parser.add_argument(
        "--use_offsets_pso",
        action="store_true",
        help="Optimize the offsets as a part of the PSO",
    )
    parser.add_argument(
        "--saved_position",
        type=str,
        default="",
        help="Does not select the best prev score",
    )
    parser.add_argument(
        "--max_value",
        type=float,
        default=1,
        help="Maximum value of the elements initially",
    )
    parser.add_argument(
        "--min_value",
        type=float,
        default=0,
        help="Minimum value of the elements initially",
    )
    parser.add_argument(
        "--max_range",
        type=float,
        default=np.inf,
        help="Maximum range of the elements initially",
    )
    parser.add_argument(
        "--min_range",
        type=float,
        default=-np.inf,
        help="Minimum range of the elements initially",
    )
    parser.add_argument("--c1", type=float, default=2, help="Positional Dependence")
    parser.add_argument("--c2", type=float, default=2, help="Global Dependence")
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of iterations the algorithm will run",
    )
    parser.add_argument(
        "--func",
        type=str,
        default="other",
        choices=[
            "other",
        ],
    )
    parser.add_argument("--output_pickle", type=str, required=True)
    return parser.parse_args()


def main():
    # initialisation

    try:
        args = _parse_args_new()
        num_units = args.num_units
        num_feat = args.num_feat
        if args.use_offsets_pso:
            num_feat += 2
        args.max_range, args.min_range = (1, 0)
        start_pso = PSO_grid(args)
        function = lambda x: start_pso.calculate_values(x)
        position = np.random.random((num_units, num_feat))
        if args.use_offsets_pso:
            position[:, : num_feat - 2] = (
                position[:, : num_feat - 2] * (args.max_value - args.min_value)
                + args.min_value
            )
            position[:, : num_feat - 2] = position[:, : num_feat - 2] / (
                position[:, : num_feat - 2].sum(axis=1, keepdims=1)
            )
            position[:, -2:] = -position[:, -2:]
        else:
            position = position * (args.max_value - args.min_value) + args.min_value
            position = position / (position.sum(axis=1, keepdims=1))
        velocity = np.random.random((num_units, num_feat))
        if args.saved_position:
            with open(args.saved_position, "rb") as f:
                saved_positions = np.load(f)
            num_particles_saved, num_feat_saved = saved_positions.shape
            assert (
                num_feat_saved == num_feat
            ), "The saved number of features and the number of features in configuration do not match"
            num_particles_saved = min(num_particles_saved, num_units)
            position[:num_particles_saved, :] = saved_positions
        best_individual = np.copy(position)
        valuesnow = function(position)
        best_pos = position[np.argmin(valuesnow), :].reshape(1, num_feat)
        indibestnow = np.copy(valuesnow)

        # Iteration to reach the stable point
        for i in tqdm(range(args.iters)):
            r1 = np.random.random()
            r2 = np.random.random()
            velocity = (
                r1 * velocity
                - args.c1 * r1 * (position - best_individual)
                - args.c2 * r2 * (position - best_pos)
            )
            position = position + velocity
            if args.use_offsets_pso:
                position[:, : num_feat - 2] = (
                    position[:, : num_feat - 2] * (args.max_value - args.min_value)
                    + args.min_value
                )
                position[:, : num_feat - 2] = np.clip(
                    position[:, : num_feat - 2], args.min_range, args.max_range
                )
                position[:, : num_feat - 2] = position[:, : num_feat - 2] / (
                    position[:, : num_feat - 2].sum(axis=1, keepdims=1)
                )
                position[:, -2:] = np.minimum(position[:, -2:], 0)
            else:
                position = position / (position.sum(axis=1, keepdims=1) + 1e-10)
                position = np.clip(position, args.min_range, args.max_range)
                position = position / (position.sum(axis=1, keepdims=1) + 1e-10)
            valuesnow = function(position)
            temp = valuesnow < indibestnow
            # pdb.set_trace()
            temp = temp.reshape(num_units, 1)
            best_individual = best_individual + temp * (position - best_individual)
            indibestnow = indibestnow + temp * (valuesnow - indibestnow)
            best_pos = best_individual[np.argmin(indibestnow), :].reshape(1, num_feat)
            # If it reaches a stable point which is not the maxima, it will cause an unequilibruim
            if np.isclose(np.max(valuesnow), np.min(valuesnow), rtol=1e-1, atol=1e-5):
                break
        print(best_pos)
        print(-np.min(indibestnow))
    except KeyboardInterrupt:
        print("Gridsearch interrupted by user!")
        pass
    with open(args.output_pickle, "wb") as f:
        pickle.dump(start_pso.total_values, f)


if __name__ == "__main__":
    main()
