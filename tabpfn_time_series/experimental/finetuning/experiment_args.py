from argparse import ArgumentParser, Namespace


def common_parse_args(return_parser: bool = False) -> Namespace | ArgumentParser:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Fine-tune TabPFN Time Series models")

    # Configuration
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment configuration to use",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="vanilla",
        help="Name of the base model configuration to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Name of the method configuration to use (overrides experiment's method)",
    )

    # Runtime settings
    parser.add_argument(
        "--device", type=str, help="Device to use for training (cuda or cpu)"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset size",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    # Logging settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tabpfn-ts-ft",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="dummy-run-name",
        help="Custom name for the W&B run",
    )
    parser.add_argument("--tags", type=str, nargs="+", help="Tags for the W&B run")

    # Training settings
    parser.add_argument(
        "--max_epochs", type=int, help="Maximum number of epochs to train"
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "-s",
        "--accumulate_grad_steps",
        type=int,
        help="Number of gradient accumulation steps",
    )

    parser.add_argument(
        "--no_log_model",
        action="store_true",
        help="Disable logging the model to Weights & Biases",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    # Dataset settings
    parser.add_argument(
        "--split_train_to_val",
        action="store_true",
        default=None,  # to not overwrite the experiment config when not provided
        help="Split datasets into train and test sets using only training datasets",
    )
    parser.add_argument(
        "--use_train_as_val",
        action="store_true",
        default=None,  # to not overwrite the experiment config when not provided
        help="Use training datasets for validation instead of test datasets",
    )

    parser.add_argument(
        "--also_validate_on_train",
        action="store_true",
        help="Also validate on training datasets",
    )

    if return_parser:
        return parser
    else:
        return parser.parse_args()
