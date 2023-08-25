# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import collections
import contextlib
import functools
import inspect
import logging
import os
import re
import time
import traceback
from collections import OrderedDict, defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from fairseq import moe_checkpoint_utils
from fairseq.data import data_utils
from fairseq.data.multilingual.multilingual_utils import get_lang_tok
from fairseq.dataclass.configs import CheckpointConfig, FairseqConfig
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
from fairseq.distributed import utils as dist_utils
from fairseq.distributed.fully_sharded_data_parallel import FSDP, has_FSDP
from fairseq.file_io import PathManager, torch_load_cpu
from fairseq.models import FairseqDecoder, FairseqEncoder
from fairseq.utils import safe_getattr

logger = logging.getLogger(__name__)


def save_checkpoint(
    cfg: CheckpointConfig,
    trainer,
    epoch_itr,
    val_loss,
    training_finished=False,
    async_callback_fn=None,
):
    from fairseq import meters

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if cfg.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if cfg.no_save:
        return

    trainer.consolidate_optimizer()  # TODO(SS): we dont need if no_save_optimizer_state

    if not trainer.should_save_checkpoint_on_current_rank:
        return

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    logger.info(f"Preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(a, b):
        return a >= b if cfg.maximize_best_checkpoint_metric else a <= b

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
        end_of_epoch and not cfg.no_epoch_checkpoints and epoch % cfg.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
        not end_of_epoch
        and cfg.save_interval_updates > 0
        and updates % cfg.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = (
        val_loss is not None
        and (
            not hasattr(save_checkpoint, "best")
            or is_better(val_loss, save_checkpoint.best)
        )
        and not cfg.no_best_checkpoints
    )
    if (
        val_loss is not None
        and cfg.keep_best_checkpoints > 0
        and not cfg.no_best_checkpoints
    ):
        worst_best = getattr(save_checkpoint, "best", None)
        chkpts = checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if len(chkpts) > 0:
            p = chkpts[-1] if cfg.maximize_best_checkpoint_metric else chkpts[0]
            worst_best = float(p.rsplit("_")[-1].replace("{}.pt".format(suffix), ""))
        # add random digits to resolve ties
        with data_utils.numpy_seed(epoch, updates, val_loss):
            rand_sfx = np.random.randint(0, cfg.keep_best_checkpoints)

        checkpoint_conds[
            "checkpoint.best_{}_{:.2f}{}{}.pt".format(
                cfg.best_checkpoint_metric, val_loss, rand_sfx, suffix
            )
        ] = worst_best is None or is_better(val_loss, worst_best)
    checkpoint_conds[
        "checkpoint_last{}.pt".format(suffix)
    ] = not cfg.no_last_checkpoints

    extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})

    checkpoints = [
        os.path.join(cfg.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        if PathManager.islink(checkpoints[0]):
            PathManager.rm(checkpoints[0])
        if trainer.is_moe and trainer.is_data_parallel_master:
            shared = re.sub("rank-[0-9]+", "shared", checkpoints[0])
            if PathManager.islink(shared):
                PathManager.rm(shared)

        trainer.save_checkpoint(
            checkpoints[0],
            extra_state,
            training_finished=training_finished,
            async_callback_fn=async_callback_fn,
        )

        if cfg.synchronize_checkpoints_before_copy:
            torch.distributed.barrier()

        def copy_or_symlink(src, dest):
            if cfg.symlink_best_and_last_checkpoints:
                PathManager.symlink(src, dest)
            elif cfg.write_checkpoints_asynchronously:
                pass  # TODO[ioPath]: Need to implement a delayed asynchronous file copying/moving feature.
            else:
                assert PathManager.copy(
                    src, dest, overwrite=True
                ), f"Failed to copy {src} to {dest}"

        for cp in checkpoints[1:]:
            copy_or_symlink(src=checkpoints[0], dest=cp)
            if (trainer.is_moe or trainer.is_base_moe) and (
                trainer.is_data_parallel_master
                or (trainer.is_fsdp and trainer.use_sharded_state)
            ):
                copy_or_symlink(
                    src=re.sub("rank-[0-9]+", "shared", checkpoints[0]),
                    dest=re.sub("rank-[0-9]+", "shared", cp),
                )

        write_timer.stop()
        logger.info(
            "Saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )

    delete_old_checkpoint_files(
        cfg,
        end_of_epoch,
        trainer.is_moe or trainer.is_base_moe,
        suffix,
        trainer.is_data_parallel_master,
    )


def delete_old_checkpoint_files(
    cfg: DictConfig,
    end_of_epoch: bool,
    is_moe: bool,
    suffix: str,
    is_data_parallel_master: bool,
):
    if not end_of_epoch and cfg.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        if cfg.keep_interval_updates_pattern == -1:
            checkpoints = checkpoint_paths(
                cfg.save_dir, pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix)
            )
        else:
            checkpoints = checkpoint_paths(
                cfg.save_dir,
                pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix),
                keep_match=True,
            )
            checkpoints = [
                x[0]
                for x in checkpoints
                if x[1] % cfg.keep_interval_updates_pattern != 0
            ]

        for old_chk in checkpoints[cfg.keep_interval_updates :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

        suffixes = [suffix]
        if is_moe and is_data_parallel_master:
            suffixes.append("-shared")

        # remove old checkpoints; checkpoints are sorted in descending order
        for one_suffix in suffixes:
            checkpoints = checkpoint_paths(
                cfg.save_dir, pattern=r"checkpoint_\d+_(\d+){}\.pt".format(one_suffix)
            )
            for old_chk in checkpoints[cfg.keep_interval_updates :]:
                if os.path.lexists(old_chk):
                    os.remove(old_chk)
    if cfg.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            cfg.save_dir, pattern=r"checkpoint(\d+){}\.pt".format(suffix)
        )
        for old_chk in checkpoints[cfg.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)

    if cfg.keep_best_checkpoints > 0:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if not cfg.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[cfg.keep_best_checkpoints :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)
            elif PathManager.exists(old_chk):
                PathManager.rm(old_chk)


def load_checkpoint(cfg: CheckpointConfig, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """

    reset_optimizer = cfg.reset_optimizer
    reset_lr_scheduler = cfg.reset_lr_scheduler
    optimizer_overrides = ast.literal_eval(cfg.optimizer_overrides)
    reset_meters = cfg.reset_meters
    reset_dataloader = cfg.reset_dataloader
    replication_count = cfg.get("replication_count", 1)

    if cfg.finetune_from_model is not None and (
        reset_optimizer or reset_lr_scheduler or reset_meters or reset_dataloader
    ):
        raise ValueError(
            "--finetune-from-model can not be set together with either --reset-optimizer"
            " or reset_lr_scheduler or reset_meters or reset_dataloader"
        )

    suffix = (
        trainer.checkpoint_suffix
        if not safe_getattr(cfg, "ignore_suffix", None)
        else None
    )
    if (
        cfg.restore_file == "checkpoint_last.pt"
    ):  # default value of restore_file is 'checkpoint_last.pt'
        checkpoint_path = os.path.join(
            cfg.save_dir, "checkpoint_last{}.pt".format(suffix)
        )
        first_launch = not PathManager.exists(checkpoint_path)
        if first_launch and getattr(cfg, "continue_once", None) is not None:
            checkpoint_path = cfg.continue_once
        elif cfg.finetune_from_model is not None and first_launch:
            # if there is no last checkpoint to restore, start the finetune from pretrained model
            # else just use usual logic to load checkpoint, e.g. restart from last checkpoint and etc.
            if PathManager.exists(cfg.finetune_from_model):
                checkpoint_path = cfg.finetune_from_model
                reset_optimizer = True
                reset_lr_scheduler = True
                reset_meters = True
                reset_dataloader = True
                logger.info(
                    f"loading pretrained model from {checkpoint_path}: "
                    "optimizer, lr scheduler, meters, dataloader will be reset"
                )
            else:
                raise ValueError(
                    f"--finetune-from-model {cfg.finetune_from_model} does not exist"
                )
    elif suffix is not None:
        checkpoint_path = cfg.restore_file.replace(".pt", suffix + ".pt")
    else:
        checkpoint_path = cfg.restore_file

    if cfg.restore_file != "checkpoint_last.pt" and cfg.finetune_from_model:
        raise ValueError(
            "--finetune-from-model and --restore-file (non-default value) "
            "can not be specified together: " + str(cfg)
        )

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        reset_optimizer,
        reset_lr_scheduler,
        optimizer_overrides,
        reset_meters=reset_meters,
        replication_count=replication_count,
    )

    if (
        extra_state is not None
        and "best" in extra_state
        and not reset_optimizer
        and not reset_meters
    ):
        save_checkpoint.best = extra_state["best"]

    if extra_state is not None and not reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state["train_iterator"]
        epoch_itr = trainer.get_train_iterator(
            epoch=itr_state["epoch"], load_dataset=True, **passthrough_args
        )
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(
            epoch=1, load_dataset=True, **passthrough_args
        )

    trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch_itr


def is_checkpoint_sharded(checkpoint_files) -> bool:
    "Infer if state is sharded based on whether largest file is more than 10% larger than smallest."
    if not checkpoint_files:
        return False
    sizes = [os.path.getsize(p) for p in checkpoint_files]
    size_ratio = max(sizes) / min(sizes)
    if size_ratio >= 1.1:
        return False
    else:
        return True


def get_paths_to_load(local_path, suffix="rank-", replication_count=1):
    checkpoint_files = glob(re.sub(f"{suffix}[0-9]+", f"{suffix}*", local_path))
    if not is_checkpoint_sharded(checkpoint_files):
        return [local_path]
    checkpoint_files_count = len(checkpoint_files)
    world_size = dist_utils.get_data_parallel_world_size()
    checkpoint_files_count = checkpoint_files_count / replication_count
    fnames = []
    if world_size >= checkpoint_files_count:
        return [local_path]

    assert checkpoint_files_count % world_size == 0

    n_local_files = int(checkpoint_files_count / world_size)
    logger.info(
        f"Loading {checkpoint_files_count} on {world_size} workers: {n_local_files} files per worker."
    )

    rank = dist_utils.get_data_parallel_rank()
    start_rank = n_local_files * rank  #
    for rank_to_load in range(start_rank, start_rank + n_local_files):
        fname = re.sub(
            f"{suffix}[0-9]+",
            f"{suffix}{rank_to_load}",
            local_path,
        )
        fnames.append(fname)
    return fnames


def load_checkpoint_to_cpu(
    path, arg_overrides=None, load_on_all_ranks=False, is_moe=False
) -> dict:
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).

    If doing single-GPU training or if the checkpoint is only being loaded by at
    most one process on each node (current default behavior is for only rank 0
    to read the checkpoint from disk), load_on_all_ranks should be False to
    avoid errors from torch.distributed not having been initialized or
    torch.distributed.barrier() hanging.

    If all processes on each node may be loading the checkpoint
    simultaneously, load_on_all_ranks should be set to True to avoid I/O
    conflicts.

    There's currently no support for > 1 but < all processes loading the
    checkpoint on each node.
    """
    local_path = PathManager.get_local_path(path)
    # The locally cached file returned by get_local_path() may be stale for
    # remote files that are periodically updated/overwritten (ex:
    # checkpoint_last.pt) - so we remove the local copy, sync across processes
    # (if needed), and then download a fresh copy.
    if local_path != path and PathManager.path_requires_pathmanager(path):
        try:
            os.remove(local_path)
        except FileNotFoundError:
            # With potentially multiple processes removing the same file, the
            # file being missing is benign (missing_ok isn't available until
            # Python 3.8).
            pass
        if load_on_all_ranks:
            torch.distributed.barrier()
        local_path = PathManager.get_local_path(path)

    # path to checkpoint...-shared.pt
    shared_path = re.sub("rank-[0-9]+", "shared", local_path)
    # this should be num_training_gpus // num_experts
    replication_count = (
        arg_overrides.get("replication_count", 1) if arg_overrides else 1
    )
    paths_to_load = get_paths_to_load(
        local_path,
        suffix="rank-" if is_moe else "shard",
        replication_count=replication_count,
    )
    if is_moe and os.path.exists(shared_path):
        expert_state = moe_checkpoint_utils.load_expert_state(
            paths_to_load
        )  # Possibly merge experts
        shared_state = torch_load_cpu(shared_path)
        state = moe_checkpoint_utils.merge_expert_and_shared_state(
            expert_state, shared_state
        )
    else:
        if len(paths_to_load) > 1:
            state = _merge_flat_fsdp_shards([torch_load_cpu(f) for f in paths_to_load])
        else:
            state = torch_load_cpu(local_path)
    logger.info("Rank 0: Done reading from disk")

    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)

    if "cfg" in state and state["cfg"] is not None:

        # hack to be able to set Namespace in dict config. this should be removed when we update to newer
        # omegaconf version that supports object flags, or when we migrate all existing models
        from omegaconf import __version__ as oc_version
        from omegaconf import _utils

        if oc_version < "2.2":
            old_primitive = _utils.is_primitive_type
            _utils.is_primitive_type = lambda _: True

            state["cfg"] = OmegaConf.create(state["cfg"])

            _utils.is_primitive_type = old_primitive
            OmegaConf.set_struct(state["cfg"], True)
        else:
            state["cfg"] = OmegaConf.create(state["cfg"], flags={"allow_objects": True})

        if arg_overrides is not None:
            overwrite_args_by_name(state["cfg"], arg_overrides)

    state = _upgrade_state_dict(state)
    return state


def load_model_ensemble(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
    is_moe=False,
    build_model_hook=None,
):
    """Loads an ensemble of models.

    Args:
        filenames (List[str]): checkpoint files to load
        arg_overrides (Dict[str,Any], optional): override model args that
            were used during model training
        task (fairseq.tasks.FairseqTask, optional): task to use for loading
    """
    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble, args, _task = load_model_ensemble_and_task(
        filenames,
        arg_overrides,
        task,
        strict,
        suffix,
        num_shards,
        state,
        is_moe=is_moe,
        build_model_hook=build_model_hook,
    )
    return ensemble, args


def get_maybe_sharded_checkpoint_filename(
    filename: str, suffix: str, shard_idx: int, num_shards: int
) -> str:
    orig_filename = filename
    filename = filename.replace(".pt", suffix + ".pt")
    fsdp_filename = filename[:-3] + f"-shard{shard_idx}.pt"
    model_parallel_filename = orig_filename[:-3] + f"_part{shard_idx}.pt"
    if PathManager.exists(fsdp_filename):
        return fsdp_filename
    elif num_shards > 1:
        return model_parallel_filename
    else:
        return filename


def upgrade_state_for_langs_difference(state, model_config, task):
    """Accounts for the difference in dictionaries due to language tokens
    to allow ensembling between multilingual and bilingual models"""

    lang_count_diff = len(task.langs) - len(model_config.langs)
    assert (
        lang_count_diff >= 0
    ), "Removing langs from ensemble components not yet supported!"

    if model_config.encoder_langtok is not None:
        orig_embed_tokens = state["model"]["encoder.embed_tokens.weight"]
        upgraded_embed_tokens = torch.zeros(
            (orig_embed_tokens.shape[0] + lang_count_diff, orig_embed_tokens.shape[1]),
            dtype=orig_embed_tokens.dtype,
            device=orig_embed_tokens.device,
        )

        first_lang_tok = task.source_dictionary.index(
            get_lang_tok(task.langs[0], "multilingual")
        )
        # language tokens appear at the end of the dictionary
        upgraded_embed_tokens[:first_lang_tok, :] = orig_embed_tokens[
            :first_lang_tok, :
        ]
        for i, lang in enumerate(model_config.langs):
            lang_tok = task.source_dictionary.index(get_lang_tok(lang, "multilingual"))
            upgraded_embed_tokens[lang_tok, :] = orig_embed_tokens[
                first_lang_tok + i, :
            ]

        state["model"]["encoder.embed_tokens.weight"] = upgraded_embed_tokens
        del orig_embed_tokens

    if model_config.decoder_langtok:
        for weight_name in (
            "decoder.embed_tokens.weight",
            "decoder.output_projection.weight",
        ):
            orig_weights = state["model"][weight_name]
            upgraded_weights = torch.zeros(
                (orig_weights.shape[0] + lang_count_diff, orig_weights.shape[1]),
                dtype=orig_weights.dtype,
                device=orig_weights.device,
            )

            first_lang_tok = task.target_dictionary.index(
                get_lang_tok(task.langs[0], "multilingual")
            )
            # language tokens appear at the end of the dictionary
            upgraded_weights[:first_lang_tok, :] = orig_weights[:first_lang_tok, :]
            for i, lang in enumerate(model_config.langs):
                lang_tok = task.target_dictionary.index(
                    get_lang_tok(lang, "multilingual")
                )
                upgraded_weights[lang_tok, :] = orig_weights[first_lang_tok + i, :]

            state["model"][weight_name] = upgraded_weights
            del orig_weights


def load_model_ensemble_and_task(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
    is_moe=False,
    build_model_hook=None,
):

    logger.info("load_model_ensemble_and_task is_moe={}".format(is_moe))

    assert state is None or len(filenames) == 1

    from fairseq import tasks

    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble = []
    cfg = None

    for filename in filenames:
        orig_filename = filename
        model_shard_state = {"shard_weights": [], "shard_metadata": []}
        assert num_shards > 0
        st = time.time()
        for shard_idx in range(num_shards):
            filename = get_maybe_sharded_checkpoint_filename(
                orig_filename, suffix, shard_idx, num_shards
            )

            if not PathManager.exists(filename):
                raise IOError("Model file not found: {}".format(filename))

            if state is None:
                state = load_checkpoint_to_cpu(
                    filename,
                    arg_overrides,
                    is_moe=is_moe,
                )
            if "args" in state and state["args"] is not None:
                cfg = convert_namespace_to_omegaconf(state["args"])
            elif "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            else:
                raise RuntimeError(
                    f"Neither args nor cfg exist in state keys = {state.keys()}"
                )

            if task is None:
                task = tasks.setup_task(cfg.task)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            if build_model_hook is not None:
                model = build_model_hook(cfg, task)
            elif "fsdp_metadata" in state and num_shards > 1:
                model_shard_state["shard_weights"].append(state["model"])
                model_shard_state["shard_metadata"].append(state["fsdp_metadata"])
                # check FSDP import before the code goes too far
                if not has_FSDP:
                    raise ImportError(
                        "Cannot find FullyShardedDataParallel. "
                        "Please install fairscale with: pip install fairscale"
                    )
                if shard_idx == num_shards - 1:
                    consolidated_model_state = FSDP.consolidate_shard_weights(
                        shard_weights=model_shard_state["shard_weights"],
                        shard_metadata=model_shard_state["shard_metadata"],
                    )
                    model = task.build_model(cfg.model)
                    if (
                        "optimizer_history" in state
                        and len(state["optimizer_history"]) > 0
                        and "num_updates" in state["optimizer_history"][-1]
                    ):
                        model.set_num_updates(
                            state["optimizer_history"][-1]["num_updates"]
                        )

                    if (
                        hasattr(cfg.model, "langs")
                        and hasattr(task, "langs")
                        and cfg.model.langs != task.langs
                    ):
                        upgrade_state_for_langs_difference(
                            consolidated_model_state, cfg.model, task
                        )

                    model.load_state_dict(
                        consolidated_model_state, strict=strict, model_cfg=cfg.model
                    )
            else:
                # model parallel checkpoint or unsharded checkpoint
                # support old external tasks

                argspec = inspect.getfullargspec(task.build_model)
                if "from_checkpoint" in argspec.args:
                    model = task.build_model(cfg.model, from_checkpoint=True)
                else:
                    model = task.build_model(cfg.model)
                if (
                    "optimizer_history" in state
                    and len(state["optimizer_history"]) > 0
                    and "num_updates" in state["optimizer_history"][-1]
                ):
                    model.set_num_updates(state["optimizer_history"][-1]["num_updates"])

                if (
                    hasattr(cfg.model, "langs")
                    and hasattr(task, "langs")
                    and cfg.model.langs
                    and task.langs
                    and cfg.model.langs != task.langs
                ):
                    upgrade_state_for_langs_difference(state, cfg.model, task)
                model.load_state_dict(
                    state["model"], strict=strict, model_cfg=cfg.model
                )
                logger.info("Done loading state dict")
            # reset state so it gets loaded for the next model in ensemble
            state = None
            if shard_idx % 10 == 0 and shard_idx > 0:
                elapsed = time.time() - st
                logger.info(
                    f"Loaded {shard_idx} shards in {elapsed:.2f}s, {elapsed / (shard_idx+1):.2f}s/shard"
                )

        # build model for ensemble
        ensemble.append(model)
    return ensemble, cfg, task


def load_model_ensemble_and_task_from_hf_hub(
    model_id,
    cache_dir: Optional[str] = None,
    arg_overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hf_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    library_name = "fairseq"
    cache_dir = cache_dir or (Path.home() / ".cache" / library_name).as_posix()
    cache_dir = snapshot_download(
        model_id, cache_dir=cache_dir, library_name=library_name, **kwargs
    )

    _arg_overrides = arg_overrides or {}
    _arg_overrides["data"] = cache_dir
    models, cfg, task = load_model_ensemble_and_task(
        [p.as_posix() for p in Path(cache_dir).glob("*.pt")],
        arg_overrides=_arg_overrides,
    )
    cfg.task.config_yaml = f"{cache_dir}/{cfg.task.config_yaml}"
    return models, cfg, task


def checkpoint_paths(path, pattern=r"checkpoint(\d+)\.pt", keep_match=False):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = PathManager.ls(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    if keep_match:
        return [(os.path.join(path, x[1]), x[0]) for x in sorted(entries, reverse=True)]
    else:
        return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(
    obj, filename: str, async_write: bool = False, async_callback_fn=None
):
    assert (
        async_callback_fn is None or async_write
    ), "async_callback_fn requires async_write=True (--save-async)"
    if async_write and async_callback_fn is not None:
        callback = functools.partial(async_callback_fn, filename)
    else:
        callback = None
    if async_write:
        with PathManager.opena(filename, "wb", callback_after_file_close=callback) as f:
            _torch_persistent_save(obj, f)
    else:
        if PathManager.supports_rename(filename):
            # do atomic save
            with PathManager.open(filename + ".tmp", "wb") as f:
                _torch_persistent_save(obj, f)
            PathManager.rename(filename + ".tmp", filename)
        else:
            # fallback to non-atomic save
            with PathManager.open(filename, "wb") as f:
                _torch_persistent_save(obj, f)


def _torch_persistent_save(obj, f, num_retries=3):
    if isinstance(f, str):
        with PathManager.open(f, "wb") as h:
            torch_persistent_save(obj, h)
        return
    for i in range(num_retries):
        try:
            return torch.save(obj, f)
        except Exception:
            if i == num_retries - 1:
                logger.error(traceback.format_exc())
                raise


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""

    # add optimizer_history
    if "optimizer_history" not in state:
        state["optimizer_history"] = [
            {"criterion_name": "CrossEntropyCriterion", "best_loss": state["best_loss"]}
        ]
        state["last_optimizer_state"] = state["optimizer"]
        del state["optimizer"]
        del state["best_loss"]
    # move extra_state into sub-dictionary
    if "epoch" in state and "extra_state" not in state:
        state["extra_state"] = {
            "epoch": state["epoch"],
            "batch_offset": state["batch_offset"],
            "val_loss": state["val_loss"],
        }
        del state["epoch"]
        del state["batch_offset"]
        del state["val_loss"]
    # reduce optimizer history's memory usage (only keep the last state)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1]["optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    # record the optimizer class name
    if "optimizer_name" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["optimizer_name"] = "FairseqNAG"
    # move best_loss into lr_scheduler_state
    if "lr_scheduler_state" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["lr_scheduler_state"] = {
            "best": state["optimizer_history"][-1]["best_loss"]
        }
        del state["optimizer_history"][-1]["best_loss"]
    # keep track of number of updates
    if "num_updates" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["num_updates"] = 0
    # use stateful training data iterator
    if "train_iterator" not in state["extra_state"]:
        state["extra_state"]["train_iterator"] = {
            "epoch": state["extra_state"].get("epoch", 0),
            "iterations_in_epoch": state["extra_state"].get("batch_offset", 0),
        }

    # backward compatibility, cfg updates
    if "args" in state and state["args"] is not None:
        # old model checkpoints may not have separate source/target positions
        if hasattr(state["args"], "max_positions") and not hasattr(
            state["args"], "max_source_positions"
        ):
            state["args"].max_source_positions = state["args"].max_positions
            state["args"].max_target_positions = state["args"].max_positions
        # default to translation task
        if not hasattr(state["args"], "task"):
            state["args"].task = "translation"
        # --raw-text and --lazy-load are deprecated
        if getattr(state["args"], "raw_text", False):
            state["args"].dataset_impl = "raw"
        elif getattr(state["args"], "lazy_load", False):
            state["args"].dataset_impl = "lazy"
        # epochs start at 1
        if state["extra_state"]["train_iterator"] is not None:
            state["extra_state"]["train_iterator"]["epoch"] = max(
                state["extra_state"]["train_iterator"].get("epoch", 1), 1
            )
        # --remove-bpe ==> --postprocess
        if hasattr(state["args"], "remove_bpe"):
            state["args"].post_process = state["args"].remove_bpe
        # --min-lr ==> --stop-min-lr
        if hasattr(state["args"], "min_lr"):
            state["args"].stop_min_lr = state["args"].min_lr
            del state["args"].min_lr
        # binary_cross_entropy / kd_binary_cross_entropy => wav2vec criterion
        if hasattr(state["args"], "criterion") and state["args"].criterion in [
            "binary_cross_entropy",
            "kd_binary_cross_entropy",
        ]:
            state["args"].criterion = "wav2vec"
        # remove log_keys if it's None (criteria will supply a default value of [])
        if hasattr(state["args"], "log_keys") and state["args"].log_keys is None:
            delattr(state["args"], "log_keys")
        # speech_pretraining => audio pretraining
        if (
            hasattr(state["args"], "task")
            and state["args"].task == "speech_pretraining"
        ):
            state["args"].task = "audio_pretraining"
        # audio_cpc => wav2vec
        if hasattr(state["args"], "arch") and state["args"].arch == "audio_cpc":
            state["args"].arch = "wav2vec"
        # convert legacy float learning rate to List[float]
        if hasattr(state["args"], "lr") and isinstance(state["args"].lr, float):
            state["args"].lr = [state["args"].lr]
        # convert task data arg to a string instead of List[string]
        if (
            hasattr(state["args"], "data")
            and isinstance(state["args"].data, list)
            and len(state["args"].data) > 0
        ):
            state["args"].data = state["args"].data[0]

        state["cfg"] = convert_namespace_to_omegaconf(state["args"])

    if "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
        with open_dict(cfg):
            # any upgrades for Hydra-based configs
            if (
                "task" in cfg
                and "eval_wer_config" in cfg.task
                and isinstance(cfg.task.eval_wer_config.print_alignment, bool)
            ):
                cfg.task.eval_wer_config.print_alignment = "hard"
            if "generation" in cfg and isinstance(cfg.generation.print_alignment, bool):
                cfg.generation.print_alignment = (
                    "hard" if cfg.generation.print_alignment else None
                )
            if (
                "model" in cfg
                and "w2v_args" in cfg.model
                and cfg.model.w2v_args is not None
                and (
                    hasattr(cfg.model.w2v_args, "task") or "task" in cfg.model.w2v_args
                )
                and hasattr(cfg.model.w2v_args.task, "eval_wer_config")
                and cfg.model.w2v_args.task.eval_wer_config is not None
                and isinstance(
                    cfg.model.w2v_args.task.eval_wer_config.print_alignment, bool
                )
            ):
                cfg.model.w2v_args.task.eval_wer_config.print_alignment = "hard"

    return state


def prune_state_dict(state_dict, model_cfg: Optional[DictConfig]):
    """Prune the given state_dict if desired for LayerDrop
    (https://arxiv.org/abs/1909.11556).

    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.

    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    arch = None
    if model_cfg is not None:
        arch = (
            model_cfg._name
            if isinstance(model_cfg, DictConfig)
            else getattr(model_cfg, "arch", None)
        )

    if not model_cfg or arch is None or arch == "ptt_transformer":
        # args should not be none, but don't crash if it is.
        return state_dict

    encoder_layers_to_keep = getattr(model_cfg, "encoder_layers_to_keep", None)
    decoder_layers_to_keep = getattr(model_cfg, "decoder_layers_to_keep", None)

    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict

    # apply pruning
    logger.info(
        "Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop"
    )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(
            int(layer_string) for layer_string in layers_to_keep.split(",")
        )
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)

        regex = re.compile(r"^{layer}.*\.layers\.(\d+)".format(layer=layer_name))
        return {"substitution_regex": regex, "mapping_dict": mapping_dict}

    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        # if layer has no number in it, it is a supporting layer, such as an
        # embedding
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        # otherwise, layer should be pruned.
        original_layer_number = match.group(1)
        # figure out which mapping dict to replace from
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass[
                "substitution_regex"
            ].search(layer_name):
                new_layer_number = pruning_pass["mapping_dict"][original_layer_number]
                substitution_match = pruning_pass["substitution_regex"].search(
                    layer_name
                )
                new_state_key = (
                    layer_name[: substitution_match.start(1)]
                    + new_layer_number
                    + layer_name[substitution_match.end(1) :]
                )
                new_state_dict[new_state_key] = state_dict[layer_name]

    # Since layers are now pruned, *_layers_to_keep are no longer needed.
    # This is more of "It would make it work fix" rather than a proper fix.
    if isinstance(model_cfg, DictConfig):
        context = open_dict(model_cfg)
    else:
        context = contextlib.ExitStack()
    with context:
        if hasattr(model_cfg, "encoder_layers_to_keep"):
            model_cfg.encoder_layers_to_keep = None
        if hasattr(model_cfg, "decoder_layers_to_keep"):
            model_cfg.decoder_layers_to_keep = None

    return new_state_dict


def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder],
    checkpoint: str,
    strict: bool = True,
    state_key_replacements: Optional[Dict[str, str]] = None,
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            component_state_dict[component_subkey] = state["model"][key]
    if state_key_replacements:
        for name, _ in component.named_parameters():
            if name in state_key_replacements:
                component_state_dict[name] = state["model"][
                    state_key_replacements[name]
                ]
                del state["model"][state_key_replacements[name]]
    component.load_state_dict(component_state_dict, strict=strict)
    return component


def verify_checkpoint_directory(save_dir: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    rank = dist_utils.get_global_rank()
    temp_file_path = os.path.join(save_dir, f"dummy{rank}")
    try:
        with open(temp_file_path, "w"):
            pass
    except OSError as e:
        logger.warning(
            "Unable to access checkpoint save directory: {}".format(save_dir)
        )
        raise e
    else:
        try:
            os.remove(temp_file_path)
        except FileNotFoundError:
            pass


def save_ema_as_checkpoint(src_path, dst_path):
    state = load_ema_from_checkpoint(src_path)
    torch_persistent_save(state, dst_path)


def load_ema_from_checkpoint(fpath):
    """Loads exponential moving averaged (EMA) checkpoint from input and
    returns a model with ema weights.

    Args:
      fpath: A string path of checkpoint to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    new_state = None

    with PathManager.open(fpath, "rb") as f:
        new_state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )

        # EMA model is stored in a separate "extra state"
        model_params = new_state["extra_state"]["ema"]

        for key in list(model_params.keys()):
            p = model_params[key]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if key not in params_dict:
                params_dict[key] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                raise ValueError("Key {} is repeated in EMA model params.".format(key))

        if len(params_dict) == 0:
            raise ValueError(
                f"Input checkpoint path '{fpath}' does not contain "
                "ema model weights, is this model trained with EMA?"
            )

    new_state["model"] = params_dict
    return new_state


OPT_KEY = moe_checkpoint_utils.OPT_KEY


def _merge_flat_fsdp_shards(shards_to_load: List[Dict]) -> Dict:
    """Concatenate tensor entries in a list of local_state_dicts  into one local_state_dict to allow resumption on a different world size."""
    merged_state = {}
    world_size = dist_utils.get_data_parallel_world_size()
    for key in shards_to_load[0].keys():
        merged_state[key] = shards_to_load[0][key]
    if "shard_metadata" not in merged_state:
        # Note: comment this out if you have sharded checkpoints that you think can be loaded
        raise KeyError(f"key shard_metadata not in {merged_state.keys()}")
    # assume only <= shard is padded
    pad_info = _get_pad_info(shards_to_load[-1])
    dtype = torch.float16
    for k in shards_to_load[0]["model"]:
        dtype = shards_to_load[0]["model"][k].dtype
        if "flat_param" in k:
            assert k in pad_info
            pad_info_k = pad_info[k]
            catted = torch.cat([x["model"][k] for x in shards_to_load])
            if world_size == 1 and pad_info_k > 0:
                catted = catted[:-pad_info_k]
            elif world_size > 1 and pad_info_k > 0:
                raise NotImplementedError(
                    f"Param {k} padded with {pad_info_k} extra elements. You must use the consolidate script."
                )
            merged_state["model"][k] = catted

    if "decoder.version" not in merged_state["model"]:
        merged_state["model"]["decoder.version"] = torch.tensor([3.0], dtype=dtype)
    if OPT_KEY in merged_state:
        merged_state[OPT_KEY] = _merge_flat_fsdp_opt_state(shards_to_load)
    return merged_state


def _merge_flat_fsdp_opt_state(shards_to_load: List[Dict]) -> Dict:
    """Logic described here: https://tinyurl.com/2p86zffr"""
    result = shards_to_load[0][OPT_KEY]
    pad_info = _get_pad_info(shards_to_load[-1])
    world_size = dist_utils.get_data_parallel_world_size()
    os2model_key = dict(
        zip(shards_to_load[0][OPT_KEY]["state"].keys(), pad_info.keys())
    )
    for k in shards_to_load[0][OPT_KEY]["state"].keys():
        # 0,1,2,3... if each layer wrapped, else 0
        for k2 in shards_to_load[0][OPT_KEY]["state"][k].keys():
            # exp_avg, exp_avg_sq, step (for adam32 bit)
            states = [x[OPT_KEY]["state"][k][k2] for x in shards_to_load]
            if not torch.is_tensor(states[0]) or is_singleton_tensor(states[0]):
                result["state"][k][k2] = states[0]
            else:
                catted = torch.cat(states)
                pad_info_k = pad_info[os2model_key[k]]
                if world_size == 1 and pad_info_k > 0:  # unpad
                    catted = catted[:-pad_info_k]
                result["state"][k][k2] = catted
    return result


def is_singleton_tensor(x: Any) -> bool:
    """Is x a dimensionless tensor?"""
    return torch.is_tensor(x) and x.dim() == 0


def _get_pad_info(state_dict: Dict) -> Dict[str, int]:
    if "shard_metadata" not in state_dict:
        # Note: comment this out if you have sharded checkpoints that you think can be loaded
        return defaultdict(lambda: 0)
    res = {}
    for m in state_dict["shard_metadata"]["param_metadata"]:
        fsdp_path = m["fsdp_path"]
        for k, v in m["params"].items():
            full_key = f"{fsdp_path}.{k}" if fsdp_path else k
            assert full_key not in res, f"collision: {full_key} already in {res}"
            res[full_key] = v["padding"]
    return res
