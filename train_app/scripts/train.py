from __future__ import annotations

import copy
import os
import shutil
from enum import Enum
from glob import glob
from urllib.parse import urlparse

import fire
import pytorch_lightning as pl
import torch
import wandb
import yaml
from dotty_dict import dotty
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

import train_app.dataset as dataset  # noqa
import train_app.models as models  # noqa

import train_app.utils as utils
from train_app.models.base import ModelBase

# from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy

os.system(f"export PYTHONPATH={os.path.dirname(os.path.abspath(__file__)) + os.sep + '..'+ os.sep + '..'}")


class Trainer:
    """Training procedure start with config yaml

    Parameters
    ----------
    project : str
        Name of the project.
    data : str | dict
        File path of the config YAML or a dictionary that includes the configurations in the same format as expected in YAML file.
    logger : Trainer.LoggerType or str, optional
        Class of the logger will be used in training, by default LoggerType.TENSORBOARD.
    visualize : bool, optional
        Whether to visualize training data locally during training or not, by default False.
    start_process : bool, optional
        Whether to start training after initialization or not, by default True.

    Returns
    -------
    Trainer
        Trainer object for start training with config yaml

    """

    class LoggerType(Enum):
        WANDB = "wandb"
        MLFLOW = "mlflow"
        TENSORBOARD = "tensorboard"

        @staticmethod
        def from_str(m_type: str):
            if m_type.lower() == Trainer.LoggerType.WANDB.value:
                return Trainer.LoggerType.WANDB
            elif m_type.lower() == Trainer.LoggerType.MLFLOW.value:
                return Trainer.LoggerType.MLFLOW
            else:
                return Trainer.LoggerType.TENSORBOARD

    class TrainType(Enum):
        TRAIN = "train"
        VALID = "valid"

    def __init__(
        self,
        project: str,
        data: str | dict,
        logger: Trainer.LoggerType | str = LoggerType.TENSORBOARD,
        visualize: bool = False,
        start_process: bool = True,
    ) -> None:
        self.project = project
        self.logger_type = logger if type(logger) == Trainer.LoggerType else Trainer.LoggerType.from_str(str(logger))
        self.is_visualize = visualize
        self.train_type: Trainer.TrainType = Trainer.TrainType.TRAIN if "train_type" not in dir(self) else self.train_type
        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.module_dir = os.path.join(self.file_path, ".." + os.sep + ".." + os.sep + "train_app")
        self.loggers = []  # type: list[object]
        self.config_file_name = os.path.basename(data) if isinstance(data, str) else "auto_generated_config.yml"
        self.config_data = data if isinstance(data, dict) else utils.read_yaml(data)

        pl.seed_everything(1)

        # prepare run directory
        self.__run_path()
        # set config to defaults
        self.config = self.__default_config()
        # update defaults from given yaml
        utils.deep_update(self.config, self.config_data)
        # correct the config with loss manager
        ModelBase.correct_config(self.config)

        self.dataset_conf = self.config["dataset"]
        self.hypes = self.config["hypes"]
        self.hardware = self.config["hardware"]

        # set callbacks
        self.callbacks = self.__generate_callbacks(self.config["callbacks"])
        # Create datasets
        self.__data_loader()
        # correct the config with start model
        self.__model()
        # update the config with wandb logger
        self.__logger()
        utils.print_config(self.config)

        # self.strategy = DDPSpawnStrategy(find_unused_parameters=False)
        
        self.pl_trainer = pl.Trainer(
            max_epochs=self.hypes["epochs"],
            callbacks=self.callbacks,
            logger=self.loggers,
            # deterministic keeps the trainings same, slows down the training.
            # deterministic=False,
            strategy="ddp",
            devices=self.hardware["devices"],
            accelerator=self.hardware["accelerator"],
            # num_nodes=2
            # auto_select_gpus=True,
            log_every_n_steps=30,
        )

        if start_process:
            self.process()
        # TODO: check fire shows help after command run succesfully.

    def process(self):
        """Train and evaluate the model using the specified data loaders.
        This method performs the training and evaluation steps using the `pl_trainer` with the provided model,
        train data loader, and validation data loader. It fits the model to the training data and evaluates it
        on the validation set. If a test data loader is available, it also performs testing on the test data.
        Finally, it prints the best model path and the corresponding best model score obtained during training.

        Returns
        -------
            None

        """
        self.pl_trainer.fit(self.model, self.train_dataloader, self.val_data_loader)

        print(f"best model path: {self.pl_trainer.checkpoint_callback.best_model_path}")
        print(f"best model score: {self.pl_trainer.checkpoint_callback.best_model_score}")

        if self.test_dataloader:
            self.pl_trainer.test(dataloaders=self.test_dataloader, ckpt_path=self.pl_trainer.checkpoint_callback.best_model_path)

        # TODO: print last bests

    def __del__(self):
        if self.loggers:
            for logger in self.loggers:
                if type(logger) == WandbLogger:
                    wandb.finish()
                if type(logger) == MLFlowLogger:
                    logger.finalize(status="success")

    def __data_loader(self):
        """Private method to create and configure data loaders for training, validation, and testing.
        This method is responsible for creating and configuring data loaders based on the dataset configuration and training type.
        It uses the `generate_dataset` function from the `utils` module to generate the appropriate dataset for each task type.
        The data loaders are created using `torch.utils.data.DataLoader` with the specified batch size, number of workers,
        shuffle option, persistent workers, and collate function (if provided).

        Note
        ----
        This method assumes that the dataset configuration (`self.dataset_conf`) and training type (`self.train_type`) are properly
        set before calling this method.

        Returns
        -------
        None

        """

        def create_dataloader(task_type):
            _dataset, _ = utils.generate_dataset(self.dataset_conf, task_type)
            data_loader = torch.utils.data.DataLoader(
                _dataset,
                batch_size=self.hypes["batch_size"],
                num_workers=self.dataset_conf[task_type]["num_workers"],
                shuffle=task_type == "train",
                collate_fn=eval(self.dataset_conf[task_type]["collate_fn"]) if "collate_fn" in self.dataset_conf[task_type] else None,
            )
            return data_loader, _dataset

        self.train_dataloader = self.val_data_loader = self.test_dataloader = None

        if self.train_type == Trainer.TrainType.TRAIN:
            self.train_dataloader, self.train_dataset = create_dataloader("train")

        if "valid" in self.dataset_conf.keys():
            self.val_data_loader, self.val_dataset = create_dataloader("valid")

        if "test" in self.dataset_conf.keys():
            self.test_dataloader, self.test_dataset = create_dataloader("test")

    def __run_path(self):
        """Private method to prepare the run path and save necessary files.
        This method prepares the run path by creating a directory based on the train type value (`self.train_type.value`).
        If the train type is "VALID", the directory is created without including weights. Otherwise, weights are included in the directory.
        The method also copies the data file specified by `self.data_path` to the run path directory.
        The run path is stored in the `self.run_path` attribute.

        Note
        ----
        This method assumes that the project is properly set (`self.project`) and the data path (`self.data_path`) is valid.

        Returns
        -------
            None

        """
        path = utils.prepare_directory(
            f"run/{self.train_type.value}/", self.project, with_weights=False if self.train_type == Trainer.TrainType.VALID else True
        )
        with open(os.path.join(path, self.config_file_name), "w") as f:
            f.write(yaml.dump(self.config_data, indent=4))
        # TODO: change with logger
        print(f"{self.train_type.value} output will save: " + path)
        self.run_path = path

    def __default_config(self):
        """Private method to generate the default configuration dictionary.
        This method generates a default configuration dictionary that includes the default values for different settings.

        Returns
        -------
        dict
            The default configuration dictionary.

        """

        config = {}  # type: dict
        config["model"] = {}  # type: dict
        config["dataset"] = {}  # type: dict
        config["callbacks"] = {"ModelCheckpoint": {"dirpath": self.run_path + "weights/"}}
        config["hypes"] = {"batch_size": 1, "epochs": 1, "lr": 0.0001}
        config["hardware"] = {"accelerator": "gpu", "devices": -1}

        return config

    def __model(self):
        """Private method to initialize the model.
        This method initializes the model based on the configuration specified in `self.config["model"]`.
        It first creates a deep copy of the model configuration to avoid modifying the original configuration.
        If the "args" key is not present in the model configuration, it creates an empty dictionary for it.
        The "config" and "visualize" keys are added to the "args" dictionary, representing the overall configuration and visualization flag.
        If the "start_model" key is present in the model configuration or the train type is "valid", it loads the model from the specified path using
        the `utils.load_model` function.
        Otherwise, it generates a new model using the `utils.generate_model` function.

        Returns
        -------
        None

        """
        model_conf = copy.deepcopy(self.config["model"])
        if "args" not in model_conf:
            model_conf["args"] = {}  # type: dict
        model_conf["args"]["config"] = self.config
        model_conf["args"]["visualize"] = self.is_visualize
        if hasattr(self.train_dataset, "postprocess"):
            model_conf["args"]["postprocess"] = self.train_dataset.postprocess

        start_model_conf = model_conf["start_model"] if "start_model" in model_conf else None
        if type == "valid" or start_model_conf:
            # TODO: connect another wandb to get model from another server
            self.model, model_conf = utils.load_model(model_path=start_model_conf, model_config=model_conf, force_config=True)
        else:
            self.model, model_class = utils.generate_model(model_conf)

    def __logger(self):
        """Private method to initialize the loggers.
        This method initializes the loggers based on the configuration specified in `self.config["loggers"]`.
        If the "wandb" logger is specified in the configuration, it creates a WandbLogger and syncs the configs with it.
        It copies the code to the Wandb directory and removes the "__pycache__" directories.
        The WandbLogger is added to the list of loggers.
        If the "mlflow" logger is specified in the configuration, it creates an MLFlowLogger and logs the artifacts.
        The MLFlowLogger is added to the list of loggers.
        If the "tensorboard" logger is specified in the configuration, it creates a TensorBoardLogger and adds it to the list of loggers.

        Returns
        -------
        None

        """
        hypes = self.config["hypes"]

        def clean_save_path(path):
            shutil.rmtree(path, ignore_errors=True)
            shutil.copytree(self.module_dir, path)
            # remove __pycache__ directories
            shutil.rmtree(f"{path}/__pycache__", ignore_errors=True)
            [shutil.rmtree(folder, ignore_errors=True) for folder in glob(f"{path}/**/__pycache__")]

        if "wandb" in self.config["loggers"]:
            self.wandb_logger = WandbLogger(project=self.project, log_model="all", config=hypes)
            if pl.utilities.rank_zero_only.rank == 0:
                # sync the configs
                config_tmp = dotty(self.config)
                wandb_config = dict(self.wandb_logger.experiment.config)
                for key in wandb_config:
                    if key in config_tmp:
                        config_tmp[key] = wandb_config[key]
                    if "." not in key:
                        hypes[key] = wandb_config[key]
                self.config = config_tmp.to_dict()
                self.wandb_logger.experiment.config.update(hypes)

                # copy the code to the wandb and remove __pycache__ if exists
                wandb_save_dir = "code/train_app"  # TODO : add os.sep or path.join

                wandb_save_path = os.path.join(wandb.run.dir, wandb_save_dir)
                clean_save_path(wandb_save_path)
                # copy configs
                with open(os.path.join(wandb.run.dir, "args_config.yml"), "w") as f:
                    f.write(yaml.dump(self.config_data, indent=4))
                with open(os.path.join(wandb.run.dir, "app_config.yml"), "w") as f:
                    f.write(yaml.dump(self.config, indent=4))

            self.loggers.append(self.wandb_logger)

        if "mlflow" in self.config["loggers"]:
            mlflow_config = self.config["loggers"]["mlflow"]
            os.environ["LOGNAME"] = mlflow_config["user"]
            mlflow_save_dir = "code/train_app"
            self.mlflow_logger = MLFlowLogger(
                experiment_name=self.project,
                tracking_uri=mlflow_config["tracking_uri"],
                log_model="all",
            )
            if pl.utilities.rank_zero_only.rank == 0:
                self.mlflow_logger.experiment.log_artifact(
                    self.mlflow_logger.run_id, local_path=os.path.join(self.run_path, self.config_file_name), artifact_path="code/config"
                )
                self.mlflow_logger.experiment.log_artifact(self.mlflow_logger.run_id, local_path="train_app", artifact_path="code")

                p = urlparse(self.mlflow_logger.experiment.get_run(self.mlflow_logger.run_id).info.artifact_uri)
                mlflow_artifact_path = os.path.abspath(os.path.join(p.netloc, p.path))
                mlflow_save_path = os.path.join(mlflow_artifact_path, mlflow_save_dir)
                clean_save_path(mlflow_save_path)

            self.loggers.append(self.mlflow_logger)

        if "tensorboard" in self.config["loggers"]:
            self.tensorboard_logger = TensorBoardLogger("tensorboard", name=self.project)
            self.loggers.append(self.tensorboard_logger)

    def __generate_callbacks(self, callbackconfig):
        """Generates the callbacks specified in the given configurations. The callbacks are generated from
        Pytorch Lightning's callbacks module.

        Parameters
        ----------
        callbackconfig : dict
            _description_

        Returns
        -------
        list
            List of callback objects.
        """
        return utils.generate_callbacks(callbackconfig=callbackconfig)


class Validation(Trainer):
    @utils.copy_doc(Trainer.__init__)
    def __init__(self, *args, **kwargs) -> None:
        self.train_type = Trainer.TrainType.VALID
        super().__init__(*args, **kwargs)

    def process(self):
        self.pl_trainer.validate(self.model, self.val_data_loader)


fire_train_type_selector = {"train": Trainer, "valid": Validation}

if __name__ == "__main__":
    fire.Fire(fire_train_type_selector)
