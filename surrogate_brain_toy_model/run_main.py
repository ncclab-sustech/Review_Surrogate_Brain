"""
run_training.py  (Python ≥3.8, PyTorch ≥2.0)

Example usage:
    python run_training.py --patients 123 060 --device 6 --epochs 500
"""

import argparse, functools, random, datetime
import numpy as np
from pathlib import Path
import torch
from run_model import (
    build_loaders, build_model,
    EarlyStopping, train_model, evaluate_model,
    reg_l1_weight, reg_l2_weight, reg_tv, reg_smooth,
    make_reg_spectral
)
from functools import partial


# ───────────────────────── Utility Functions ──────────────────────────
def set_seed(seed: int = 42):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_ckpt_path(root: Path,
                   model_type: str,
                   patient: str,
                   tau: int,
                   dim_z: int,
                   regs: dict):
    
    date_tag = datetime.datetime.now().strftime("%Y%m%d")
    reg_tag  = "-".join([k for k, (_, lam) in regs.items() if lam > 0])
    fname    = f"{model_type}_HUP{patient}_tau{tau}_hidden{dim_z}_{reg_tag}.pth"
    path     = root / date_tag
    path.mkdir(parents=True, exist_ok=True)
    return path / fname



# ─────────────────────────── Main Workflow ───────────────────────────
def main(args):
    set_seed(args.seed)

    # Parse regularization configuration
    reg_config = {}
    for reg_spec in args.regs:
        if ':' in reg_spec:
            name, weight = reg_spec.split(':', 1)
            try:
                weight = float(weight)
                reg_config[name] = weight
            except ValueError:
                print(f"⚠️ Ignored invalid weight: {reg_spec}")
        else:
            reg_config[reg_spec] = 1.0  # Default weight = 1.0

    # Training and evaluation loop
    for patient in args.patients:
        # Load training and testing datasets
        train_data = np.load(
            f'./dataset/HUP{patient}_ictal_train_freq_200.npy'
        )
        test_data  = np.load(
            f'./dataset/HUP{patient}_ictal_test_freq_200.npy'
        )

        # Build DataLoaders for training/validation/testing
        train_loader, val_loader, test_loader = build_loaders(
            train_data=train_data,
            test_data=test_data,
            seq_len_total=375,
            stride=375,
            batch_size=args.batch_size,
            seed=args.seed
        )

        # Available models (can be customized per patient if needed)
        # model_list = ['RNN', 'lrRNN', 'EIRNN', 'dendPLRNN'] 

        for model_type in args.models:
            for dim_z in args.dim_z:
                for tau in args.taus:
                    # Configure regularizers
                    regularizers = {}
                    if "l1_weight" in reg_config:
                        regularizers["l1_weight"] = (reg_l1_weight, reg_config["l1_weight"])
                    if "l2_weight" in reg_config:
                        regularizers["l2_weight"] = (reg_l2_weight, reg_config["l2_weight"])
                    # Optional regularizers (currently commented out)
                    # if "smooth" in reg_config:
                    #     smooth_reg = partial(reg_smooth, tau=1)
                    #     regularizers["smooth"] = (smooth_reg, reg_config["smooth"])
                    # if "tv" in reg_config:
                    #     tv_reg = partial(reg_tv, tau=1)
                    #     regularizers["tv"] = (tv_reg, reg_config["tv"])
                    # if "spec_sc" in reg_config:
                    #     spec_reg = make_reg_spectral(n_fft=256, use_sc=True)
                    #     regularizers["spec_sc"] = (spec_reg, reg_config["spec_sc"])

                    # ── Core Hyperparameters ─────────────────────────
                    dim_x  = train_data.shape[1]
                    device        = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
                    lr            = 1e-3
                    num_epochs    = args.epochs
                    save_path = make_ckpt_path(
                        Path(args.save_root), model_type, patient, tau, dim_z, regularizers
                    )

                    # ── Build Model & Optimizer ─────────────────────
                    model = build_model(model_type, dim_x, dim_z, tau, device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.75, patience=5
                    )
                    early_stop = EarlyStopping(patience=30, min_delta=1e-4)
                    criterion  = torch.nn.MSELoss()

                    # ── Training ─────────────────────────────────────
                    if args.run_train:
                        train_model(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            early_stop=early_stop,
                            num_epochs=num_epochs,
                            save_path=save_path,
                            regularizers=regularizers,
                            device=device,
                            dim_z=dim_z,
                            tau=tau,
                            model_type=model_type
                        )

                    # ── Evaluation (reload best checkpoint) ─────────
                    reg_tag = "-".join([k for k, (_, lam) in regularizers.items() if lam > 0])
                    if args.run_eval:
                        eval_df = evaluate_model(
                            model, test_loader, save_path,
                            tau=tau,
                            device=device,
                            name=patient,
                            model_type=model_type,
                            reg_tag=reg_tag,
                            dim_z=dim_z
                        )



# ───────────────────────── Command Line Interface ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="+", default=["123"],
                        help="List of patient IDs, e.g., 123 060 064 …")
    parser.add_argument("--taus", nargs="+", type=int,
                        default=[5,10,15,20,25,30,35,40,45,50],
                        help="List of prediction horizons (in timesteps)")
    parser.add_argument("--dim_z", nargs="+", type=int,
                        default=[8,16,32,64,128,256,512],
                        help="List of latent space dimensions to try")
    parser.add_argument("--models", nargs="+", default=["RNN", "lrRNN", "EIRNN", "dendPLRNN"],
                    choices=["RNN", "lrRNN", "EIRNN", "dendPLRNN"],
                    help="List of model types to train")
    parser.add_argument("--regs", nargs="*", default=[],
                    help="Regularization configuration, format: name[:weight]. "
                         "Supported: l1_weight, l2_weight. "
                         "Examples: 'l2_weight:1e-7' or 'smooth' (default weight=1.0)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device ID")
    parser.add_argument("--save_root", default="./results/", type=str)
    parser.add_argument("--run_train", action="store_true")
    parser.add_argument("--run_eval",  action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # If neither --run_train nor --run_eval is specified, run both by default
    if not (args.run_train or args.run_eval):
        args.run_train = args.run_eval = True

    main(args)
