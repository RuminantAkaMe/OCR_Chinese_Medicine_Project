



"""
PaddleOCR training entrypoint (tools/train.py)

This module is the top-level training script used by PaddleOCR. It wires
together dataset loaders, model construction, optimizer and learning-rate
scheduler, training loop, evaluation, and optional early stopping.

High-level behaviour:
 - Reads configuration provided by `tools.program.preprocess`.
 - Initializes distributed training (if enabled).
 - Builds train/validation dataloaders using `ppocr.data.build_dataloader`.
 - Constructs the model (`ppocr.modeling.architectures.build_model`) and
     wraps it for DataParallel when running distributed training.
 - Creates the optimizer and lr-scheduler via `ppocr.optimizer.build_optimizer`.
 - Instantiates an EarlyStopping helper when `Global.use_early_stopping` is
     enabled in the config. The EarlyStopping object is placed back into
     `config["Global"]["early_stopping_callback"]` so the training loop
     (implemented in `tools.program.train`) can access and call it during
     evaluation checkpoints.
 - After training finishes (normally or via early stopping), if
     `restore_best_weights` was enabled the script restores the best model
     weights saved by the EarlyStopping instance.

EarlyStopping specifics (how the class in this file works):
 - Constructed with: patience, min_delta, mode ("max" for accuracy, "min"
     for loss), restore_best_weights, save_best_only, and save_dir.
 - The callback keeps track of the best score seen so far. If a new score
     improves beyond `best_score + min_delta` (for mode=="max"), it is
     considered an improvement and the counter resets.
 - If no improvement happens for `patience` consecutive calls, the
     `early_stop` flag becomes True and training should halt.
 - When `restore_best_weights` is True, the callback stores the
     model.state_dict() for the best score and provides `restore_weights(model)`
     which will copy those best weights back into the model.
 - When `save_best_only` is True, the callback will write the best
     model to disk under `save_dir` with a filename prefix of
     `best_accuracy_epoch_` plus the epoch number.

Important config keys referenced by this script (from the loaded YAML):
 - Global.distributed: whether to initialize Paddle distributed env
 - Global.use_early_stopping, early_stopping_patience,
     early_stopping_min_delta, early_stopping_mode,
     restore_best_weights, save_best_model_only, save_model_dir
 - Global.epoch_num: total epochs used to build optimizers/schedulers
 - Train/Eval dataset and loader settings used to build dataloaders

Note: This file intentionally adds the EarlyStopping instance into the
config so the training routine implemented in `tools.program.train` can
access and call it at evaluation checkpoints. No other major behavioral
change is introduced by this docstring insertion.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import yaml
import paddle
import paddle.distributed as dist

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
from ppocr.modeling.architectures import apply_to_static
import tools.program as program
import tools.naive_sync_bn as naive_sync_bn

dist.get_world_size()


import time


class EarlyStopping:
    def __init__(
        self,
        patience=10,
        min_delta=0.001,
        mode="max",
        restore_best_weights=True,
        save_best_only=False,
        save_dir=None,
    ):
        """
        Early stopping callback that saves only the best model

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy (higher is better), 'min' for loss
            restore_best_weights: Whether to restore best model weights
            save_best_only: Whether to save only the best model
            save_dir: Directory to save the best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.save_best_only = save_best_only
        self.save_dir = save_dir
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = 0

    def __call__(self, score, model=None, epoch=None, optimizer=None, logger=None):
        is_best = False

        if self.best_score is None:
            self.best_score = score
            is_best = True
            if model and self.restore_best_weights:
                self.best_weights = model.state_dict()
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            is_best = True
            if model and self.restore_best_weights:
                self.best_weights = model.state_dict()
            if epoch is not None:
                self.best_epoch = epoch
        else:
            self.counter += 1

        # Save best model only
        if is_best and self.save_best_only and model and self.save_dir:
            self._save_best_model(model, optimizer, epoch, logger)

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score):
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

    def _save_best_model(self, model, optimizer, epoch, logger):
        """Save the best model"""
        import paddle
        import os

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # Remove previous best model files
        for file in os.listdir(self.save_dir):
            if file.startswith("best_accuracy"):
                os.remove(os.path.join(self.save_dir, file))

        # Save new best model
        best_model_path = os.path.join(self.save_dir, f"best_accuracy_epoch_{epoch}")

        save_dict = {"model": model.state_dict()}
        if optimizer:
            save_dict["optimizer"] = optimizer.state_dict()
        save_dict["epoch"] = epoch
        save_dict["best_accuracy"] = self.best_score

        paddle.save(save_dict, best_model_path + ".pdparams")

        if logger:
            logger.info(
                f"New best model saved: {best_model_path}.pdparams (accuracy: {self.best_score:.4f})"
            )

    def restore_weights(self, model):
        if self.best_weights and model:
            model.set_state_dict(self.best_weights)


def main(config, device, logger, vdl_writer, seed):
    # init dist environment
    if config["Global"]["distributed"]:
        dist.init_parallel_env()

    global_config = config["Global"]

    # Add early stopping to config instead of passing as parameter
    if global_config.get("use_early_stopping", False):
        early_stopping = EarlyStopping(
            patience=global_config.get("early_stopping_patience", 10),
            min_delta=global_config.get("early_stopping_min_delta", 0.001),
            mode=global_config.get("early_stopping_mode", "max"),
            restore_best_weights=global_config.get("restore_best_weights", True),
            save_best_only=global_config.get("save_best_model_only", True),
            save_dir=global_config.get("save_model_dir", "./output"),
        )
        
        # Store early stopping in config for program.train to access
        config["Global"]["early_stopping_callback"] = early_stopping
        
        logger.info(
            "Early stopping enabled: patience={}, min_delta={}, mode={}, save_best_only={}".format(
                early_stopping.patience,
                early_stopping.min_delta,
                early_stopping.mode,
                early_stopping.save_best_only,
            )
        )

    # build dataloader
    set_signal_handlers()
    train_dataloader = build_dataloader(config, "Train", device, logger, seed)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n"
            + "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            + "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config["Eval"]:
        valid_dataloader = build_dataloader(config, "Eval", device, logger, seed)
    else:
        valid_dataloader = None
    step_pre_epoch = len(train_dataloader)

    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list["CTCLabelDecode"] = char_num
                    # update SARLoss params
                    if (
                        list(config["Loss"]["loss_config_list"][-1].keys())[0]
                        == "DistillationSARLoss"
                    ):
                        config["Loss"]["loss_config_list"][-1]["DistillationSARLoss"][
                            "ignore_index"
                        ] = (char_num + 1)
                        out_channels_list["SARLabelDecode"] = char_num + 2
                    elif any(
                        "DistillationNRTRLoss" in d
                        for d in config["Loss"]["loss_config_list"]
                    ):
                        out_channels_list["NRTRLabelDecode"] = char_num + 3

                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num
            # update SARLoss params
            if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":
                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                        "ignore_index": char_num + 1
                    }
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (
                        char_num + 1
                    )
                out_channels_list["SARLabelDecode"] = char_num + 2
            elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":
                out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

        if config["PostProcess"]["name"] == "SARLabelDecode":  # for SAR model
            config["Loss"]["ignore_index"] = char_num - 1

    model = build_model(config["Architecture"])

    use_sync_bn = config["Global"].get("use_sync_bn", False)
    if use_sync_bn:
        if config["Global"].get("use_npu", False) or config["Global"].get(
            "use_xpu", False
        ):
            naive_sync_bn.convert_syncbn(model)
        else:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("convert_sync_batchnorm")

    model = apply_to_static(model, config, logger)

    # build loss
    loss_class = build_loss(config["Loss"])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        model=model,
    )

    # build metric
    eval_class = build_metric(config["Metric"])

    logger.info("train dataloader has {} iters".format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info("valid dataloader has {} iters".format(len(valid_dataloader)))

    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", "O2")
    amp_dtype = config["Global"].get("amp_dtype", "float16")
    amp_custom_black_list = config["Global"].get("amp_custom_black_list", [])
    amp_custom_white_list = config["Global"].get("amp_custom_white_list", [])
    if os.path.exists(
        os.path.join(config["Global"]["save_model_dir"], "train_result.json")
    ):
        try:
            os.remove(
                os.path.join(config["Global"]["save_model_dir"], "train_result.json")
            )
        except:
            pass
    if use_amp:
        AMP_RELATED_FLAGS_SETTING = {}
        if paddle.is_compiled_with_cuda():
            AMP_RELATED_FLAGS_SETTING.update(
                {
                    "FLAGS_cudnn_batchnorm_spatial_persistent": 1,
                    "FLAGS_gemm_use_half_precision_compute_type": 0,
                }
            )
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        scale_loss = config["Global"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["Global"].get(
            "use_dynamic_loss_scaling", False
        )
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        )
        if amp_level == "O2":
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=amp_level,
                master_weight=True,
                dtype=amp_dtype,
            )
    else:
        scaler = None

    # load pretrain model
    pre_best_model_dict = load_model(
        config, model, optimizer, config["Architecture"]["model_type"]
    )

    if config["Global"]["distributed"]:
        find_unused_parameters = config["Global"].get("find_unused_parameters", False)
        model = paddle.DataParallel(
            model, find_unused_parameters=find_unused_parameters
        )

    # start train with original signature
    program.train(
        config,
        train_dataloader,
        valid_dataloader,
        device,
        model,
        loss_class,
        optimizer,
        lr_scheduler,
        post_process_class,
        eval_class,
        pre_best_model_dict,
        logger,
        step_pre_epoch,
        vdl_writer,
        scaler,
        amp_level,
        amp_custom_black_list,
        amp_custom_white_list,
        amp_dtype,
    )

    # Handle early stopping results
    early_stopping = config["Global"].get("early_stopping_callback")
    if (
        early_stopping
        and early_stopping.early_stop
        and early_stopping.restore_best_weights
    ):
        logger.info("Training stopped early. Restoring best model weights...")
        early_stopping.restore_weights(model)
        logger.info(
            f"Best accuracy achieved: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}"
        )


def test_reader(config, device, logger):
    loader = build_dataloader(config, "Train", device, logger)
    import time

    starttime = time.time()
    count = 0
    try:
        for data in loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info(
                    "reader: {}, {}, {}".format(count, len(data[0]), batch_time)
                )
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)
    main(config, device, logger, vdl_writer, seed)
    # test_reader(config, device, logger)
