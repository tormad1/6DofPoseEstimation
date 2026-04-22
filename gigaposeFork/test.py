import os
import multiprocessing as mp
from itertools import islice

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.logging import get_logger
from src.utils.logging import start_disable_output, stop_disable_output
from src.runtime import GigaPoseRuntime
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__)



@hydra.main(version_base=None, config_path="configs", config_name="test")
def run_test(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    if os.name == "nt" and cfg.machine.num_workers != 0:
        logger.warning("Setting machine.num_workers=0 for Windows DataLoader startup")
        cfg.machine.num_workers = 0

    runtime, test_dataset = GigaPoseRuntime.from_config(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.disable_output:
        log = start_disable_output(os.path.join(cfg.save_dir, "test.log"))

    dataloader = runtime.make_dataloader(
        test_dataset,
        num_workers=cfg.machine.num_workers,
    )
    limit_test_batches = OmegaConf.select(cfg, "machine.trainer.limit_test_batches")
    if limit_test_batches is not None:
        dataloader = islice(dataloader, int(limit_test_batches))

    logger.info("Runtime initialized!")
    runtime.run_dataloader(dataloader, save_predictions=True)

    if cfg.disable_output:
        stop_disable_output(log)
    logger.info("Done!")


if __name__ == "__main__":
    mp.freeze_support()
    run_test()
