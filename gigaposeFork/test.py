import os
import multiprocessing as mp

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from torch.utils.data import DataLoader
from src.utils.logging import get_logger
from src.utils.logging import start_disable_output, stop_disable_output
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__)



@hydra.main(version_base=None, config_path="configs", config_name="test")
def run_test(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    normalize_paths(cfg)
    if os.name == "nt" and cfg.machine.num_workers != 0:
        logger.warning("Setting machine.num_workers=0 for Windows DataLoader startup")
        cfg.machine.num_workers = 0

    logger.info("Initializing trainer")
    cfg_trainer = cfg.machine.trainer
    if cfg_trainer.get("logger") and "TensorBoardLogger" in cfg_trainer.logger._target_:
        tensorboard_dir = f"{cfg.save_dir}/{cfg_trainer.logger.name}"
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Tensorboard logger initialized at {tensorboard_dir}")
    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.disable_output:
        log = start_disable_output(os.path.join(cfg.save_dir, "test.log"))

    trainer = instantiate(cfg_trainer)
    logger.info("Trainer initialized!")

    model = instantiate(cfg.model)
    logger.info("Model initialized!")

    cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    cfg.data.test.dataloader.batch_size = cfg.machine.batch_size
    cfg.data.test.dataloader.test_setting = cfg.test_setting
    test_dataset = instantiate(cfg.data.test.dataloader)
    test_dataloader = DataLoader(
        test_dataset.scene_dataset,
        batch_size=1,  # a single image may have multiples instances
        num_workers=cfg.machine.num_workers,
        collate_fn=test_dataset.collate_fn,
    )

    # set template dataset as a part of the model
    cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    cfg.data.test.dataloader._target_ = "src.dataloader.template.TemplateSet"
    template_dataset = instantiate(cfg.data.test.dataloader)

    model.template_datasets = {cfg.test_dataset_name: template_dataset}
    model.test_dataset_name = cfg.test_dataset_name
    model.max_num_dets_per_forward = cfg.max_num_dets_per_forward
    model.run_id = cfg.run_id or "cpu_rgb"
    logger.info("Dataloaders initialized!")

    trainer.test(
        model, dataloaders=test_dataloader, ckpt_path=cfg.model.checkpoint_path
    )

    if cfg.disable_output:
        stop_disable_output(log)
    logger.info("Done!")


def normalize_paths(cfg: DictConfig):
    cfg.machine.root_dir = to_absolute_path(str(cfg.machine.root_dir))
    cfg.save_dir = to_absolute_path(str(cfg.save_dir))
    cfg.model.checkpoint_path = to_absolute_path(str(cfg.model.checkpoint_path))
    cfg.data.test.dataloader.root_dir = to_absolute_path(
        str(cfg.data.test.dataloader.root_dir)
    )
    cfg.data.test.dataloader.template_config.dir = to_absolute_path(
        str(cfg.data.test.dataloader.template_config.dir)
    )


if __name__ == "__main__":
    mp.freeze_support()
    run_test()
