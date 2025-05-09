""" Setup utils for hydra and lightning"""
# pylint: skip-file
# TODO: file skipped because it's not ours

from contextlib import suppress

with suppress(ModuleNotFoundError):
    import logging
    import warnings
    from collections.abc import Sequence

    import pytorch_lightning as pl
    import rich.syntax
    import rich.tree
    from omegaconf import DictConfig, OmegaConf
    from pytorch_lightning.utilities import rank_zero_only

    def get_logger(name: str = __name__) -> logging.Logger:
        """Initializes multi-GPU-friendly python command line logger."""

        logger = logging.getLogger(name)

        # this ensures all logging levels get marked with the rank zero decorator
        # otherwise logs would get multiplied for each
        # GPU process in multi-GPU setup
        for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
        ):
            setattr(logger, level, rank_zero_only(getattr(logger, level)))

        return logger

    log = get_logger(__name__)

    def extras(config: DictConfig) -> None:
        """Applies optional utilities, controlled by config flags.

        Utilities:
        - Ignoring python warnings
        - Rich config printing
        """

        # disable python warnings if <config.ignore_warnings=True>
        if config.get("ignore_warnings"):
            log.info("Disabling python warnings! <config.ignore_warnings=True>")
            warnings.filterwarnings("ignore")

        # pretty print config tree using Rich library if <config.print_config=True>
        if config.get("print_config"):
            log.info("Printing config tree with Rich! <config.print_config=True>")
            print_config(config, resolve=True)

    @rank_zero_only
    def print_config(
        config: DictConfig,
        print_order: Sequence[str] = (
            "datamodule",
            "model",
            "callbacks",
            "logger",
            "trainer",
        ),
        resolve: bool = True,
    ) -> None:
        """Prints content of DictConfig using Rich library and its tree structure.

        Args:
            config (DictConfig): Configuration composed by Hydra.
            print_order (Sequence[str], optional): Determines in what order
                                                   config components are printed.
            resolve (bool, optional): Whether to resolve
                                      reference fields of DictConfig.
        """

        style = "dim"
        tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

        queue = []

        for p_field in print_order:
            queue.append(p_field) if p_field in config else log.info(
                "Field '%s' not found in config", p_field
            )

        for c_field in config:
            assert isinstance(c_field, str)
            if c_field not in queue:
                queue.append(c_field)

        for q_field in queue:
            branch = tree.add(q_field, style=style, guide_style=style)

            config_group = config[p_field]
            if isinstance(config_group, DictConfig):
                branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
            else:
                branch_content = str(config_group)

            branch.add(rich.syntax.Syntax(branch_content, "yaml"))

        rich.print(tree)

        with open("config_tree.log", "w", encoding="utf-8") as file:
            rich.print(tree, file=file)

    @rank_zero_only
    def log_hyperparameters(
        config: DictConfig, model: pl.LightningModule, trainer: pl.Trainer
    ) -> None:
        """Controls which config parts are saved by Lightning loggers.

        Additionaly saves:
        - number of model parameters
        """

        hparams = {}

        # choose which parts of hydra config will be saved to loggers
        hparams["model"] = config["model"]

        # save number of model parameters
        hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
        hparams["model/params/trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        hparams["model/params/non_trainable"] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

        hparams["datamodule"] = config["datamodule"]
        hparams["trainer"] = config["trainer"]

        if "seed" in config:
            hparams["seed"] = config["seed"]
        if "callbacks" in config:
            hparams["callbacks"] = config["callbacks"]

        # send hparams to all loggers
        if trainer.logger:
            trainer.logger.log_hyperparams(hparams)  # type: ignore

    def finish(logger: list[pl.loggers.Logger]) -> None:
        """Makes sure everything closed properly."""

        # without this sweeps with wandb logger might crash!
        for lgr in logger:
            if isinstance(lgr, pl.loggers.wandb.WandbLogger):
                import wandb

                wandb.finish()
