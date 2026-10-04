"""Shared stages for CLI commands and the one-command smoke test."""
from pathlib import Path
import yaml

from src.datasets import PlaceholderRobotDataset, load_dataset
from src.evaluation import Evaluator
from src.policies import make_policy
from src.selection import make_selector
from src.training import Trainer
from src.utils.config import save_config
from src.utils.logging import read_json, write_json
from src.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]


def paths(config: dict) -> tuple[Path, Path]:
    name = config["experiment_name"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("experiment_name must be one path component")
    return ROOT / "data" / "processed" / name, ROOT / "results" / name


def prepare_dataset(config: dict) -> None:
    data_dir, _ = paths(config)
    d = config["dataset"]
    for role, name, count, offset in (("real", d["real_dataset"], d["real_size"], 0),
                                      ("synthetic", d["synthetic_dataset"], d["synthetic_size"], 1),
                                      ("evaluation", d["evaluation_dataset"], d["evaluation_size"], 2)):
        dataset = load_dataset(name, count=count, task_id=d["task_id"], seed=config["seed"] + offset,
                               observation_dim=d["observation_dim"], action_dim=d["action_dim"])
        write_json(data_dir / f"{role}.json", [dataset[i] for i in range(len(dataset))])


def _dataset(path: Path) -> PlaceholderRobotDataset:
    return PlaceholderRobotDataset(read_json(path))


def create_subset(config: dict) -> None:
    data_dir, _ = paths(config)
    synthetic = _dataset(data_dir / "synthetic.json").filter_by_task(config["dataset"]["task_id"])
    ids = make_selector(config["selection"]["method"], config["seed"]).select(synthetic, config["selection"]["budget"])
    write_json(data_dir / "selected_ids.json", ids)


def train_policy(config: dict) -> dict:
    data_dir, result_dir = paths(config)
    if result_dir.exists():
        raise FileExistsError(f"Result already exists: {result_dir}")
    set_seed(config["seed"])
    real = _dataset(data_dir / "real.json").filter_by_task(config["dataset"]["task_id"])
    synthetic = _dataset(data_dir / "synthetic.json").filter_by_task(config["dataset"]["task_id"])
    ids = read_json(data_dir / "selected_ids.json")
    if len(ids) != config["selection"]["budget"] or len(ids) != len(set(ids)):
        raise ValueError("Selected IDs do not match the configured budget")
    merged = PlaceholderRobotDataset([real[i] for i in range(len(real))] + [synthetic.get_trajectory(i) for i in ids])
    d, t = config["dataset"], config["training"]
    policy = make_policy(config["policy"], d["observation_dim"], d["action_dim"], t["learning_rate"])
    result_dir.mkdir(parents=True)
    save_config(config, result_dir / "config.yaml")
    write_json(result_dir / "selected_ids.json", ids)
    metrics = Trainer(policy, batch_size=t["batch_size"], epochs=t["epochs"], seed=config["seed"]).train(
        merged, result_dir / "checkpoint.pt", result_dir / "train.log")
    metrics.update({"seed": config["seed"], "selection_method": config["selection"]["method"],
                    "num_real_trajectories": len(real), "num_synthetic_trajectories": len(ids)})
    write_json(result_dir / "metrics.json", metrics)
    return metrics


def evaluate_policy(config: dict) -> dict:
    data_dir, result_dir = paths(config)
    set_seed(config["seed"])
    saved_config = result_dir / "config.yaml"
    if not saved_config.exists() or saved_config.read_text() != yaml.safe_dump(config, sort_keys=False):
        raise ValueError("Evaluation config differs from training config")
    d, t = config["dataset"], config["training"]
    policy = make_policy(config["policy"], d["observation_dim"], d["action_dim"], t["learning_rate"])
    policy.load(result_dir / "checkpoint.pt")
    evaluation = _dataset(data_dir / "evaluation.json").filter_by_task(d["task_id"])
    e = config["evaluation"]
    metrics = read_json(result_dir / "metrics.json")
    metrics.update(Evaluator(e["num_episodes"], e["toy_success_mse_threshold"]).evaluate(policy, evaluation))
    write_json(result_dir / "metrics.json", metrics)
    return metrics


def run_smoke_test(config: dict) -> dict:
    _, result_dir = paths(config)
    if result_dir.exists():
        raise FileExistsError(f"Result already exists: {result_dir}")
    prepare_dataset(config)
    create_subset(config)
    train_policy(config)
    return evaluate_policy(config)
