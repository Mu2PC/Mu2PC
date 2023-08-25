# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import contextlib
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import checkpoint_utils, tasks, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqDecoder,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_speech.modules import CTCDecoder
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.transformer import (
    Embedding,
    TransformerConfig,
    TransformerDecoder,
    TransformerModelBase,
)
from fairseq.models.transformer.fsdp_wrap_expert import fsdp_wrap_expert
from fairseq.models.wav2vec import Wav2VecEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.layer_norm import LayerNorm
from omegaconf import II, MISSING
logger = logging.getLogger(__name__)

# rewrite name for backward compatibility in `make_generation_fast_`


def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    return Embedding(num_embeddings, embed_dim, padding_idx)



class Conv1dAdaptor(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_layers=3,
        kernel_size=3,
        stride=2,
        layerdrop=0.0,
        layernorm=False,
        proj=False,
    ):
        super().__init__()
        self.proj, self.proj_ln = None, None
        self.post_proj, self.post_proj_ln = None, None
        if proj:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, in_dim * 4), nn.ReLU(), nn.Linear(in_dim * 4, in_dim)
            )
            self.proj_ln = LayerNorm(in_dim)
            self.post_proj = nn.Sequential(
                nn.Linear(out_dim, out_dim * 4),
                nn.ReLU(),
                nn.Linear(out_dim * 4, out_dim),
            )
            self.post_proj_ln = LayerNorm(out_dim)

        self.layers = nn.ModuleList(
            nn.Conv1d(
                in_dim if i == 0 else out_dim,
                out_dim * 2,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            for i in range(n_layers)
        )
        self.stride = stride
        self.layerdrop = layerdrop
        self.layernorm = LayerNorm(in_dim) if layernorm else None

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--adaptor-n-layers", type=int)
        parser.add_argument("--adaptor-kernel-size", type=int)
        parser.add_argument("--adaptor-stride", type=int)
        parser.add_argument("--adaptor-layerdrop", type=float)
        parser.add_argument("--adaptor-layernorm", action="store_true")
        parser.add_argument("--adaptor-proj", action="store_true")

    def forward(self, x, padding_mask: Optional[torch.Tensor]):
        if self.layernorm is not None:
            x = self.layernorm(x)

        if self.proj is not None:
            x = x + 0.5 * self.proj(x)
            x = self.proj_ln(x)

        # T x B x C -> B x C x T
        x = x.transpose(0, 1).transpose(1, 2)
        out_lens = None
        if padding_mask is not None:
            out_lens = (~padding_mask).sum(1).float()

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                x = nn.functional.glu(layer(x), dim=1)
                if padding_mask is not None:
                    out_lens = ((out_lens - 1) / self.stride + 1).floor()
        # B x C x T -> T x B x C
        x = x.transpose(1, 2).transpose(0, 1)

        if self.post_proj is not None:
            x = x + 0.5 * self.post_proj(x)
            x = self.post_proj_ln(x)

        out_padding_mask = None
        if padding_mask is not None:
            out_padding_mask = lengths_to_padding_mask(out_lens.long())
        return x, out_padding_mask


def add_wav2vec_asr_args(parser):
    parser.add_argument("--w2v-path", help="path to wav2vec 2.0 model")
    parser.add_argument(
        "--no-pretrained-weights",
        action="store_true",
        help="if true, does not load pretrained weights",
    )
    parser.add_argument(
        "--dropout-input",
        type=float,
        metavar="D",
        help="dropout to apply to the input (after feat extr)",
    )
    parser.add_argument(
        "--final-dropout",
        type=float,
        metavar="D",
        help="dropout after transformer and before final projection",
    )
    parser.add_argument(
        "--apply-mask", action="store_true", help="apply masking during fine-tuning"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        metavar="D",
        help="dropout probability inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--activation-dropout",
        "--relu-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN inside wav2vec 2.0 model",
    )

    parser.add_argument(
        "--mask-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-prob", type=float, help="probability of replacing a token with mask"
    )

    parser.add_argument(
        "--mask-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--mask-channel-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-channel-prob",
        type=float,
        help="probability of replacing a token with mask",
    )

    parser.add_argument(
        "--mask-channel-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-channel-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-channel-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--freeze-finetune-updates",
        default=0,
        type=int,
        help="dont finetune wav2vec for this many updates",
    )

    parser.add_argument(
        "--feature-grad-mult",
        default=None,
        type=float,
        help="reset feature grad mult in wav2vec 2.0 to this",
    )

    parser.add_argument(
        "--layerdrop",
        default=0.0,
        type=float,
        help="probability of dropping a layer in wav2vec 2.0",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        help="Max input positions to be used in the conformer encoder in wav2vec 2.0",
    )

    parser.add_argument("--encoder-proj", action="store_true")

    parser.add_argument("--w2v-args", default=None)

    parser.add_argument(
        "--remove-weight-norm",
        action="store_true",
        help="if set, then the weight-norm (in one pos_conv layer) is removed from the model",
    )


def need_finetuning(ft_params, param_name):
    if ft_params == "all":
        return True
    ft_params_list = ft_params.split(",")
    for ft_param in ft_params_list:
        if ft_param in param_name:
            return True
    return False


class AVHubertEncoderWithAdaptor(FairseqEncoder):
    def build_adaptor(self, args):
        adaptor = None
        if args.adaptor_n_layers > 0:
            adaptor = Conv1dAdaptor(
                args.decoder_embed_dim,
                args.decoder_embed_dim,
                n_layers=args.adaptor_n_layers,
                kernel_size=args.adaptor_kernel_size,
                stride=args.adaptor_stride,
                layerdrop=args.adaptor_layerdrop,
                layernorm=args.adaptor_layernorm,
                proj=args.adaptor_proj,
            )
        return adaptor

    def __init__(self, encoder, args):
        super().__init__(None)
        self.w2v_encoder = encoder
        self.is_v0_arch = not args.adaptor_proj

        self.w2v_proj_ln = None
        if not self.is_v0_arch and self.w2v_encoder.proj is not None: # False
            self.w2v_proj_ln = LayerNorm(args.decoder_embed_dim)
        self.adaptor = self.build_adaptor(args)

        self.num_updates = 0
        self.freezing_updates = args.w2v_freezing_updates
        self.finetuning_params = args.finetune_w2v_params
        for k, p in self.w2v_encoder.named_parameters():
            p.requires_grad = need_finetuning(self.finetuning_params, k)

    # Original Hubert
    # def forward(self, source, padding_mask, **kwargs):
    #     w2v_args = {
    #         "source": source,
    #         "padding_mask": padding_mask,
    #     }
    #
    #     x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
    #     # B x T x C -> T x B x C
    #     x = x.transpose(0, 1)
    #
    #     return {
    #         "encoder_out": x,  # T x B x C
    #         "encoder_padding_mask": padding_mask,  # B x T
    #         "padding_mask": padding_mask
    #     }



    @classmethod
    def add_args(cls, parser):
        add_wav2vec_asr_args(parser)
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--finetune-w2v-params",
            type=str,
            metavar="STR",
            help="comma-separated param strings to finetune.",
        )
        parser.add_argument("--w2v-freezing-updates", type=int)
        parser.add_argument("--load-pretrained-encoder-from", type=str, metavar="STR")
        Conv1dAdaptor.add_args(parser)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, **kwargs):

        # if self.freezing_updates is not None:
        #     if self.num_updates > self.freezing_updates:
        #         for p in self.w2v_encoder.parameters():
        #             p.requires_grad = True

        '''
         source (dict): audio: [B, 104, T1]  video: [B, 1, T1, 88, 88]
        padding_mask: [B, T1]
        '''

        ft = self.freezing_updates <= self.num_updates   # 前N步fix参数
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_encoder.extract_finetune(source, padding_mask)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        '''
        output:
            encoder_out: [T1, B, 1024]
            padding_mask: [B, T1]
        '''

        if self.w2v_proj_ln is not None: # None
            x = self.w2v_proj_ln(x)

        if self.adaptor is not None:
            x, padding_mask = self.adaptor(x, padding_mask)  # [T, 1, 1024] -> [T//2, B, 1024]

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": []
            if padding_mask is None
            else [padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

def add_decoder_args(parser):
    parser.add_argument(
        "--activation-fn",
        type=str,
        default="relu",
        choices=utils.get_available_activation_fns(),
        help="activation function to use",
    )
    parser.add_argument(
        "--decoder-dropout", type=float, metavar="D", help="dropout probability"
    )
    parser.add_argument(
        "--decoder-attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights",
    )
    parser.add_argument(
        "--decoder-activation-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN.",
    )
    parser.add_argument(
        "--decoder-embed-dim", type=int, metavar="N", help="decoder embedding dimension"
    )
    parser.add_argument(
        "--decoder-ffn-embed-dim",
        type=int,
        metavar="N",
        help="decoder embedding dimension for FFN",
    )
    parser.add_argument(
        "--decoder-layers", type=int, metavar="N", help="num decoder layers"
    )
    parser.add_argument(
        "--decoder-attention-heads",
        type=int,
        metavar="N",
        help="num decoder attention heads",
    )
    parser.add_argument(
        "--decoder-normalize-before",
        action="store_true",
        help="apply layernorm before each decoder block",
    )
    parser.add_argument(
        "--layernorm-embedding", action="store_true", help="add layernorm to embedding"
    )
    parser.add_argument("--decoder-layerdrop", type=float, metavar="D")
    parser.add_argument("--decoder-learned-pos", action="store_true")
    parser.add_argument("--share-decoder-input-output-embed", action="store_true")
    parser.add_argument(
        "--no-scale-embedding",
        action="store_true",
        help="if True, dont scale embeddings",
    )
    parser.add_argument(
        "--load-pretrained-decoder-from",
        type=str,
        metavar="STR",
        help="model to take decoder weights from (for initialization)",
    )
    parser.add_argument(
        "--finetune-decoder-params",
        type=str,
        metavar="STR",
        help="comma-separated param strings to finetune.",
    )


def remove_weight_norm_from_model(model):
    from functools import reduce

    layers_with_wn = []
    for param_name, _ in model.named_parameters():
        if param_name.endswith("_g"):
            # retrieve the module with this param_name
            module_names = param_name.split(".")[
                :-1
            ]  # exclude the actual parameter name
            wn_module = reduce(getattr, module_names, model)
            layers_with_wn.append(wn_module)
    for wn_module in layers_with_wn:
        torch.nn.utils.remove_weight_norm(wn_module)
        logger.warning(f"Weight norm removed from module with {wn_module}\n")


@register_model("av_xm_transformer")
class AVXMTransformerModel(FairseqEncoderDecoderModel):
    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/s2t"
        model_ids = [
            "xm_transformer_600m-es_en-multi_domain",
            "xm_transformer_600m-ru_en-multi_domain",
            "xm_transformer_600m-fr_en-multi_domain",
            "xm_transformer_600m-en_es-multi_domain",
            "xm_transformer_600m-en_ru-multi_domain",
            "xm_transformer_600m-en_fr-multi_domain",
            "xm_transformer_600m-en_zh-multi_domain",
            "xm_transformer_600m-en_ar-multi_domain",
            "xm_transformer_600m-en_tr-multi_domain",
            "xm_transformer_600m-en_vi-multi_domain",
            "xm_transformer-21_en-xls_r_300m",
            "xm_transformer-en_15-xls_r_300m",
            "xm_transformer-21_en-xls_r_1b",
            "xm_transformer-en_15-xls_r_1b",
            "xm_transformer-21_en-xls_r_2b",
            "xm_transformer-en_15-xls_r_2b",
            "xm_transformer-22_16-xls_r_2b",
            "xm_transformer_s2ut_800m-es-en-st-asr-bt_h1_2022",
            "xm_transformer_s2ut_800m-en-es-st_plus_asr",
            "xm_transformer_s2ut_es_en_st_asr_test",
            "xm_transformer_s2ut_en-hk",
            "xm_transformer_s2ut_hk-en",
            "xm_transformer_sm_all-en",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config_yaml="config.yaml",
        task="speech_to_text",
        generation_args=None,
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config_yaml,
            task=task,
            generation_args=generation_args,
            **kwargs,
        )
        return S2THubInterface(x["args"], x["task"], x["models"][0])

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        AVHubertEncoderWithAdaptor.add_args(parser)
        add_decoder_args(parser)
        parser.add_argument("--checkpoint-activations", action="store_true")
        parser.add_argument("--offload-activations", action="store_true")
        parser.add_argument("--min-params-to-wrap", type=int)
        # args for mixture-of-expert layers
        parser.add_argument(
            "--moe-freq",
            type=int,
            metavar="D",
            default=0,
            help="Frequency at which we insert MoE Transformer layers",
        )
        parser.add_argument(
            "--encoder-moe-freq",
            type=int,
            metavar="D",
            default=0,
            help="Frequency at which we insert MoE Transformer encoder layers",
        )
        parser.add_argument(
            "--decoder-moe-freq",
            type=int,
            metavar="D",
            default=0,
            help="Frequency at which we insert MoE Transformer decoder layers",
        )
        parser.add_argument(
            "--moe-expert-count",
            type=int,
            metavar="D",
            default=0,
            help="Number of experts in each MoE Layer",
        )
        parser.add_argument(
            "--moe-gating-use-fp32",
            default=False,
            action="store_true",
            help="Use FP32 computations in MoE top2 gating function",
        )
        parser.add_argument(
            "--moe-second-expert-policy",
            type=str,
            default="sampling",
            help="policy for second expert, options: all/sampling/random",
        )
        parser.add_argument(
            "--moe-normalize-gate-prob-before-dropping",
            default=False,
            action="store_true",
            help="whether to normalize gate probs before or after dropping experts for capacity and randomization",
        )
        parser.add_argument(
            "--moe-expert-ffn-dim", type=int, default=0, help="MoE Expert FFN dimension"
        )
        parser.add_argument(
            "--moe-top1-expert",
            default=False,
            action="store_true",
            help="Use top1 gate instead of top2",
        )
        parser.add_argument(
            "--moe-eval-capacity-token-fraction",
            type=float,
            default=0.25,
            help="Fraction of tokens as capacity during validation"
            + "if set to negative, use same as training. range: (0.0, 1.0].",
        )
        parser.add_argument(
            "--moe-normalize-expert-grad",
            type=str,
            default="sqrt_num_experts",
            help="Divide expert gradients by (1) 'num_experts' (2) 'sqrt_num_experts'",
        )
        parser.add_argument(
            "--use-moe-pad-mask",
            default=False,
            action="store_true",
            help="Don't route padding tokens to any expert",
        )
        parser.add_argument(
            "--pass-tokens-transformer-layer",
            default=False,
            action="store_true",
            help="Pass source or masked previous output tokens to the transformer layer",
        )
        # args for pseudo-MoE layers
        parser.add_argument(
            "--alternate-ffn-embed-dim",
            type=int,
            default=0,
            help="FFN embed dim of alternate pseudo-MoE blocks",
        )
        parser.add_argument(
            "--alternate-decoder-ffn-embed-dim",
            type=int,
            default=0,
            help="FFN embed dim of alternate pseudo-MoE blocks",
        )
        parser.add_argument(
            "--base-layers",
            type=int,
            default=0,
        )

    @classmethod
    def maybe_load_pretrained(
        cls, component, checkpoint: Optional[str] = None, is_moe: bool = False
    ):
        if checkpoint is None:
            return component

        _load = checkpoint_utils.load_pretrained_component_from_model
        state_key_replacements = {}

        if is_moe:
            # Hack to initialize all experts with pretrained model weights
            if isinstance(component, FairseqEncoder):
                component_type = "encoder"
            elif isinstance(component, FairseqDecoder):
                component_type = "decoder"
            for name, _ in component.named_parameters():
                if ".moe_layer.experts." in name and len(name.split(".")) == 7:
                    s = name.split(".")
                    from_key = component_type + "." + ".".join(s[:2] + s[-2:])
                    state_key_replacements[name] = from_key

        try:
            return _load(component, checkpoint)
        except RuntimeError as e:
            logger.warning(e)
            return _load(
                component,
                checkpoint,
                strict=False,
                state_key_replacements=state_key_replacements,
            )

    @classmethod
    def build_encoder(cls, args):
        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        } # config里进行overrides


        if args.w2v_args is None: # 如果load pretrained model
            state = checkpoint_utils.load_checkpoint_to_cpu(args.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None) # pretrained checkpoint的config
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            args.w2v_args = w2v_args
        else:
            state = None
            w2v_args = args.w2v_args
            if isinstance(w2v_args, Namespace):
                args.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert args.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = args.data

        # some fix
        if 'input_modality' in w2v_args.task:
            del w2v_args.task['input_modality']

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = AVHubertEncoderWithAdaptor(encoder_, args)
        if state is not None and not args.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_encoder.load_state_dict(state["model"], strict=False)  # encoder读取state dict
            print('successfully load pretrained model!')

        encoder.w2v_encoder.remove_pretraining_modules()

        # if args.remove_weight_norm:
        #     # remove the wn for EMA usage
        #     logger.warning("Removing weight norm from wav2vec encoder")
        #     remove_weight_norm_from_model(encoder)

        return encoder

    @classmethod
    def get_decoder_args_from_checkpoint(cls, ckpt_args):
        assert "model" in ckpt_args, "Model args not found in checkpoint cfg!"
        decoder_args = {}
        for k, v in ckpt_args["model"].__dict__.items():
            if "decoder" in k:
                decoder_args[k] = v

        return decoder_args

    @classmethod
    def override_decoder_args(cls, cli_args, decoder_args_dict):
        for k, v in decoder_args_dict.items():
            if v != getattr(cli_args, k, None):
                logger.warning(
                    f"Overriding decoder arg {k}: from {getattr(cli_args, k, None)} to {v}"
                )
                setattr(cli_args, k, v)

        return cli_args

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        _args = copy.deepcopy(args)
        if args.adaptor_proj or args.encoder_proj:  # not V0 arch
            _args.encoder_embed_dim = _args.decoder_embed_dim
        _args.dropout = args.decoder_dropout
        _args.attention_dropout = args.decoder_attention_dropout
        _args.activation_dropout = args.decoder_activation_dropout
        _args.layerdrop = _args.decoder_layerdrop

        decoder = TransformerDecoderScriptable(
            TransformerConfig.from_namespace(_args),
            task.target_dictionary,
            embed_tokens,
        )
        decoder = cls.maybe_load_pretrained(
            decoder,
            getattr(args, "load_pretrained_decoder_from", None),
            getattr(args, "decoder_moe_freq", 0) > 0,
        )

        for k, p in decoder.named_parameters():
            p.requires_grad = need_finetuning(args.finetune_decoder_params, k)
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)
        # if getattr(args, "load_pretrained_decoder_from", None) is not None:
        #     ckpt = torch.load(getattr(args, "load_pretrained_decoder_from", None))
        #     decoder_args_dict = cls.get_decoder_args_from_checkpoint(ckpt["cfg"])
        #     args = cls.override_decoder_args(args, decoder_args_dict)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    @classmethod
    def build_multitask_decoder(cls, args, tgt_dict, in_dim):
        decoder_args = args.decoder_args
        decoder_args.encoder_embed_dim = in_dim
        if args.decoder_type == "transformer":
            from fairseq.models.speech_to_speech import (
                base_multitask_text_transformer_decoder_arch,
            )

            base_multitask_text_transformer_decoder_arch(decoder_args)  # 2L
            task_decoder = TransformerDecoder(
                decoder_args,
                tgt_dict,
                embed_tokens=TransformerModelBase.build_embedding(
                    decoder_args,
                    tgt_dict,
                    decoder_args.decoder_embed_dim,
                ),
            )
        elif args.decoder_type == "ctc":
            task_decoder = CTCDecoder(
                dictionary=tgt_dict,
                in_dim=in_dim,
            )
        else:
            raise NotImplementedError(
                "currently only support multitask decoder_type 'transformer', 'ctc'"
            )

        return task_decoder

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def forward(self, source, padding_mask, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(     # [B, T0] -> encoder_out: n * [B, T1, 1024]
            source=source, padding_mask=padding_mask, **kwargs
        )
        '''
        encoder_out: dict
            "encoder_out": list: [T1 x B x C]
            "encoder_padding_mask": [B x T1]
            "encoder_embedding": [] 空
            "encoder_states": [] 空
            "src_tokens": [] 空
            "src_lengths": [] 空  
        '''
        decoder_out = self.decoder(     # prev_output_tokens [B, T2], encoder_out -> decoder_out: [B, T2, 1007]
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        '''
        decoder_out: tuple
            [B, T2, 1007]
            encoder_out: dict
        '''
        return decoder_out

    def upgrade_state_dict(self, state_dict):
        for k, _ in state_dict.items():
            if "adaptor.layers" in state_dict:
                new = k.replace("adaptor.layers", "adaptor_layers")
                state_dict[new] = state_dict[k]
                del state_dict[k]


class TransformerDecoderScriptable(TransformerDecoder):
    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder_layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        # Disabling quant noise
        self.quant_noise = None

        self.project_in_dim = (
            nn.Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        for i in range(cfg.decoder_layers):
            moe_freq = max(cfg.decoder_moe_freq, cfg.moe_freq)
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_decoder_layer(
                    cfg, no_encoder_attn, is_moe_layer=is_moe_layer
                )
            )
        self.num_layers = len(self.layers)

        if cfg.decoder_normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            nn.Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        extra = {"encoder_out": encoder_out} if incremental_state is None else None
        return x, extra

    def build_decoder_layer(self, cfg, no_encoder_attn=False, is_moe_layer=False):
        layer = TransformerDecoderLayer(cfg, no_encoder_attn, is_moe_layer=is_moe_layer)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        if cfg.ddp_backend == "fully_sharded":
            if not is_moe_layer:
                layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
            else:
                layer = fsdp_wrap_expert(cfg, layer, min_num_params=min_params_to_wrap)
        return layer


def set_default_w2v_encoder_args(args):
    args.w2v_path = getattr(args, "w2v_path", None)
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0.0)
    args.final_dropout = getattr(args, "final_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    # masking
    args.apply_mask = getattr(args, "apply_mask", False)
    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.75)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)

    # channel masking
    args.mask_channel_length = getattr(args, "mask_channel_length", 64)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)
    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.alternating_freeze_finetune_updates = getattr(
        args, "alternating_freeze_finetune_updates", 0
    )
    args.feature_grad_mult = 1.0
    args.layerdrop = getattr(args, "layerdrop", 0.0)

    args.normalize = getattr(args, "normalize", False)
    args.finetune_w2v_params = getattr(args, "finetune_w2v_params", "all")
    args.w2v_freezing_updates = getattr(args, "w2v_freezing_updates", 20000)



def set_default_adaptor_args(args):
    args.adaptor_n_layers = getattr(args, "adaptor_n_layers", 3)
    args.adaptor_kernel_size = getattr(args, "adaptor_kernel_size", 3)
    args.adaptor_stride = getattr(args, "adaptor_stride", 2)
    args.adaptor_layerdrop = getattr(args, "adaptor_layerdrop", 0.0)
    args.adaptor_layernorm = getattr(args, "adaptor_layernorm", False)
    args.adaptor_proj = getattr(args, "adaptor_proj", False)


def set_default_transformer_decoder_args(args):
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_attention_dropout = getattr(args, "decoder_attention_dropout", 0.0)
    args.decoder_activation_dropout = getattr(args, "decoder_activation_dropout", 0.0)
    args.decoder_dropout = getattr(args, "decoder_dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.finetune_decoder_params = getattr(args, "finetune_decoder_params", "all")


def set_default_general_args(args):
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    args.min_params_to_wrap = getattr(args, "min_params_to_wrap", int(1e8))
    args.max_positions = getattr(args, "max_positions", 3000)


@register_model_architecture(model_name="av_xm_transformer", arch_name="av_xm_transformer")
def base_architecture(args):
    set_default_general_args(args)
    set_default_w2v_encoder_args(args)
    set_default_adaptor_args(args)
    set_default_transformer_decoder_args(args)
