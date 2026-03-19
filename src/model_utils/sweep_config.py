import argparse
from distutils.util import strtobool


def register_hyperparameter_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_length", type=float, default=None)

    for feat in ("a", "c", "e"):
        p.add_argument(f"--{feat}_loss_weight", type=float, default=None)
        p.add_argument(f"--{feat}_schedule_param", type=float, default=None)

    p.add_argument("--n_hidden", type=int, default=None)
    p.add_argument("--n_hidden_edge_feats", type=int, default=None)
    p.add_argument("--n_recycles", type=int, default=None)
    p.add_argument("--n_molecule_updates", type=int, default=None)
    p.add_argument("--convs_per_update", type=int, default=None)
    p.add_argument("--separate_mol_updaters", type=str, default=None)
    p.add_argument("--message_norm", type=str, default=None)
    p.add_argument("--self_conditioning", type=str, default=None)
    p.add_argument("--stochasticity", type=float, default=None)
    p.add_argument("--high_confidence_threshold", type=float, default=None)
    p.add_argument("--time_scaled_loss", type=str, default=None)
    p.add_argument("--exclude_charges", type=str, default=None)
    return p


def merge_config_and_args(config: dict, args: argparse.Namespace) -> dict:
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    if args.lr is not None:
        config["lr_scheduler"]["base_lr"] = args.lr

    if args.warmup_length is not None:
        config["lr_scheduler"]["warmup_length"] = args.warmup_length

    for feat in ("a", "c", "e"):
        loss_weight = getattr(args, f"{feat}_loss_weight")
        if loss_weight is not None:
            config["mol_fm"]["total_loss_weights"][feat] = loss_weight

        schedule_param = getattr(args, f"{feat}_schedule_param")
        if schedule_param is not None:
            config["interpolant_scheduler"]["params"][feat] = schedule_param

    for arg in ("separate_mol_updaters", "self_conditioning"):
        value = getattr(args, arg)
        if value is not None:
            config["vector_field"][arg] = bool(strtobool(value))

    for arg in (
        "n_hidden",
        "n_hidden_edge_feats",
        "n_recycles",
        "n_molecule_updates",
        "convs_per_update",
        "stochasticity",
        "high_confidence_threshold",
    ):
        value = getattr(args, arg)
        if value is not None:
            config["vector_field"][arg] = value

    if args.message_norm is not None:
        message_norm = args.message_norm
        if message_norm.isnumeric():
            message_norm = float(message_norm)
        config["vector_field"]["message_norm"] = message_norm

    if args.time_scaled_loss is not None:
        config["mol_fm"]["time_scaled_loss"] = bool(strtobool(args.time_scaled_loss))

    if args.exclude_charges is not None:
        config["mol_fm"]["exclude_charges"] = bool(strtobool(args.exclude_charges))

    return config
