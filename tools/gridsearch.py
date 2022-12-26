import logging
import functools
import string
import random
import pickle
import shutil
from pathlib import Path

from tqdm import tqdm

import numpy as np
import yaml

from spltrack.config import get_default_cfg
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

from tools.evaluate_orc import (
    evaluate,
    get_argument_parser as get_default_argument_parser,
)


def generate_run_id(chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(24))


def compute_metrics(
    base_cfg,
    dataset_dir,
    root_output_dir,
    params,
):
    # set cost computer params
    cfg = base_cfg.clone()
    for param, value in params.items():
        cfg.COST_COMPUTER[param] = value

    cfg.freeze()

    run_name = generate_run_id()
    output_dir_path = root_output_dir / run_name

    # make sure that the output directory does not exist already
    while output_dir_path.exists():
        run_name = generate_run_id()
        output_dir_path = root_output_dir / run_name

    output_dir_path.mkdir(parents=True)
    with open(output_dir_path / "config.yaml", "w") as f:
        f.write(cfg.dump())

    tracking_metrics = evaluate(cfg, dataset_dir, output_dir_path, quiet=True)
    shutil.make_archive(
        base_name=str(output_dir_path),
        format="zip",
        root_dir=str(root_output_dir),
        base_dir=str(output_dir_path.relative_to(root_output_dir)),
    )
    shutil.rmtree(output_dir_path)
    return tracking_metrics


def calc_grid(args):

    cfg = get_default_cfg()
    cfg.merge_from_file(str(args.config_file))

    output_dir = args.output_dir

    # Load default parameters
    print("Generating parameters grid")

    param_grid = dict()
    for param_name, value in cfg.COST_COMPUTER.items():
        if (
            "WEIGHT" in param_name or "OFFSET" in param_name
        ) and "NUMBER_DETECTION" not in param_name:
            param_grid[param_name] = [value]
    valid_param_keys = list(param_grid.keys())

    with open(args.gridsearch_config, "r") as f:
        gridsearch_config = yaml.load(f, Loader=yaml.SafeLoader)

    # Now add the parameters from the config
    for param_name, param_config in gridsearch_config.items():
        lower_bound = param_config["LOWER_BOUND"]
        upper_bound = param_config["UPPER_BOUND"]
        num_values = param_config["NUM_VALUES"]
        param_grid[param_name] = [
            float(v)
            for v in np.linspace(lower_bound, upper_bound, num_values, dtype=float)
        ]

    print("Removing duplicate configurations")
    valid_param_lists = []
    for params in tqdm(ParameterGrid(param_grid)):
        weight_params_rescaled = {
            name: int(value * 100) for name, value in params.items() if "WEIGHT" in name
        }
        rescaled_params_sum = sum(weight_params_rescaled.values())
        if rescaled_params_sum == 0:
            continue
        for param_name, rescaled_val in weight_params_rescaled.items():
            params[param_name] = float(rescaled_val / rescaled_params_sum)
        valid_param_lists.append([params[k] for k in valid_param_keys])
    unique_valid_param_lists = []
    for l in valid_param_lists:
        if l in unique_valid_param_lists:
            continue
        unique_valid_param_lists.append(l)
    valid_param_dicts = []
    for uvpl in unique_valid_param_lists:
        d = dict()
        for idx, k in enumerate(valid_param_keys):
            d[k] = uvpl[idx]
        valid_param_dicts.append(d)
    print(f"Number of configurations: {len(valid_param_dicts)}")

    if args.shuffle:
        random.shuffle(valid_param_dicts)

    gridsearch_iteration_fn = functools.partial(
        compute_metrics,
        base_cfg=cfg,
        dataset_dir=args.dataset_dir,
        root_output_dir=output_dir,
    )
    print("Starting gridsearch")

    total_values = []
    try:
        total_values = Parallel(n_jobs=args.jobs)(
            delayed(gridsearch_iteration_fn)(params=params)
            for params in valid_param_dicts
        )
        print("Gridsearch finished")
    except KeyboardInterrupt:
        print("Gridsearch interrupted by user!")
        pass

    print("Saving results")
    total_values = [(valid_param_dicts[idx], x) for idx, x in enumerate(total_values)]

    with open("total_values.pkl", "wb") as f:
        pickle.dump(total_values, f)


def _parse_args():

    parser = get_default_argument_parser()

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        help="Number of jobs to run in parallel",
    )

    parser.add_argument(
        "-g",
        "--gridsearch-config",
        type=lambda p: Path(p).resolve(strict=True),
        required=True,
        help="Gridsearch config file",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the order in which the parameters are evaluated.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    args = _parse_args()
    calc_grid(args)
