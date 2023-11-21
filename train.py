import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/",
            config_name="train.yaml",
            version_base=None)
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from mmdc_downstream.training_pipeline import train

    from mmdc_downstream.utils import extras
    #
    # # Applies optional utilities
    extras(config)
    # Train model
    return train(config)


if __name__ == "__main__":
    main()
