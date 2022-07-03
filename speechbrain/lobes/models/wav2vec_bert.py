import functools
import math
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from fairseq.data.data_utils import compute_mask_indices


# default conv block params
FEATURE_ENC_LAYERS: List[Tuple[int, int, int]] = (
    [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]
)

# based on https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py


class Wav2Vec2(nn.Module):
    """
    ASR Model of wav2vec
    """

    def __init__(
        self,
        feature_enc_layers: List[Tuple[int, int, int]] = FEATURE_ENC_LAYERS,
        conv_bias: bool = False,
        dropout: float = 0.0,
        encoder_embed_dim = 768,
        mask_prob = 0.65, # probability of replacing a token with mask (supposed to use in pretraining)
    ):
        super().__init__()

        # get the dim of output conv layer 1st block
        self.embed = feature_enc_layers[-1][0]

        # add biais on conv block (1st part of module : default False)
        self.conv_bias = conv_bias
        # self.extractor_mode = extractor_mode
        self.dropout = dropout
        self.encoder_embed_dim = encoder_embed_dim

        # definiotion of conv block for first part of model
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=self.dropout,
            conv_bias=self.conv_bias,
        )

        self.quantize_input = True 

        # adding linear layer if not same dimension from dim output ConvFeatureExtractionModel and dimm encoder 
        self.post_extract_proj = (
             nn.Linear(self.embed, self.encoder_embed_dim)
             if self.embed != self.encoder_embed_dim and not self.quantize_input
             else None
         )

        # params for compute_mask_indices
        self.mask_prob = mask_prob # # probability of replacing a feature with 0
        self.mask_selection = 'static' # how to choose mask length 
        self.mask_other =  0.0 # secondary mask argument (used for more complex distributions)
        self.mask_length = 10 
        self.no_mask_overlap =  False # whether to allow masks to overlap
        self.mask_min_space = 1 # min space between spans (if no overlap is enabled)

        # params for compute mask channel indices
        self.mask_channel_prob = 0.0 # probability of replacing a feature with 0
        self.mask_channel_before = False
        self.mask_channel_selection = 'static' # how to choose mask length for channel masking
        self.mask_channel_other = 0 # secondary mask argument (used for more complex distributions)
        self.mask_channel_length = 10 # length of the mask for features (channels)
        self.no_mask_channel_overlap = False # whether to allow channel masks to overlap
        self.mask_channel_min_space = 1 # min space between spans (if no overlap is enabled)

        # definition of dropouts 'layers'
        self.dropout_input = nn.Dropout(0.0) # dropout to apply to the input (after feat extr)
        self.dropout_features = nn.Dropout(0.0) # dropout to apply to the features (after feat extr)

        self.feature_grad_mult =  1.0 # multiply feature extractor var grads by this

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = 100 # number of negative examples from the same sample
        self.cross_sample_negatives = 0 # number of negative examples from the samples
        # self.codebook_negatives = cfg.codebook_negatives
        # self.negatives_from_everywhere = cfg.negatives_from_everywhere

        # self.logit_temp = cfg.logit_temp

        # final_dim = (
        #     cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        # )

        # if cfg.quantize_targets:
        #     vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
        #     self.quantizer = GumbelVectorQuantizer(
        #         dim=self.embed,
        #         num_vars=cfg.latent_vars,
        #         temp=cfg.latent_temp,
        #         groups=cfg.latent_groups,
        #         combine_groups=False,
        #         vq_dim=vq_dim,
        #         time_first=True,
        #         weight_proj_depth=cfg.quantizer_depth,
        #         weight_proj_factor=cfg.quantizer_factor,
        #     )
        #     self.project_q = nn.Linear(vq_dim, final_dim)
        # else:
        #     self.project_q = nn.Linear(self.embed, final_dim)

        # if cfg.quantize_input:
        #     if cfg.same_quantizer and self.quantizer is not None:
        #         vq_dim = final_dim
        #         self.input_quantizer = self.quantizer
        #     else:
        #         vq_dim = (
        #             cfg.latent_dim
        #             if cfg.latent_dim > 0
        #             else cfg.encoder_embed_dim
        #         )
        #         self.input_quantizer = GumbelVectorQuantizer(
        #             dim=self.embed,
        #             num_vars=cfg.latent_vars,
        #             temp=cfg.latent_temp,
        #             groups=cfg.latent_groups,
        #             combine_groups=False,
        #             vq_dim=vq_dim,
        #             time_first=True,
        #             weight_proj_depth=cfg.quantizer_depth,
        #             weight_proj_factor=cfg.quantizer_factor,
        #         )
        #     self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        # self.mask_emb = nn.Parameter(
        #     torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        # )
        # encoder_cls = TransformerEncoder
        # if cfg.layer_type == "conformer" and cfg.pos_enc_type in [
        #     "rel_pos",
        #     "rope",
        # ]:
        #     encoder_cls = ConformerEncoder

        # self.encoder = encoder_cls(cfg)
        self.layer_norm = nn.LayerNorm(self.embed)

        # self.target_glu = None
        # if cfg.target_glu:
        #     self.target_glu = nn.Sequential(
        #         nn.Linear(final_dim, final_dim * 2), nn.GLU()
        #     )

        # self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

    def apply_mask(
        self, x, padding_mask, mask_indices=None, mask_channel_indices=None,
    ):
        B, T, C = x.shape

        # if self.mask_channel_prob > 0 and self.mask_channel_before:
        #     mask_channel_indices = compute_mask_indices(
        #         (B, C),
        #         None,
        #         self.mask_channel_prob,
        #         self.mask_channel_length,
        #         self.mask_channel_selection,
        #         self.mask_channel_other,
        #         no_overlap=self.no_mask_channel_overlap,
        #         min_space=self.mask_channel_min_space,
        #     )
        #     mask_channel_indices = (
        #         torch.from_numpy(mask_channel_indices)
        #         .to(x.device)
        #         .unsqueeze(1)
        #         .expand(-1, T, -1)
        #     )
        #     x[mask_channel_indices] = 0

        # if self.mask_prob > 0:
        #     if mask_indices is None:
        #         mask_indices = compute_mask_indices(
        #             (B, T),
        #             padding_mask,
        #             self.mask_prob,
        #             self.mask_length,
        #             mask_selection = 'static',
        #             mask_other = 0.0,
        #             min_masks=2,
        #             no_overlap=self.no_mask_overlap,
        #             min_space=self.mask_min_space,
        #             require_same_masks= True,
        #             mask_dropout=0.0,
        #         )
        #         mask_indices = torch.from_numpy(mask_indices).to(x.device)
        #     x = index_put(x, mask_indices, self.mask_emb)
        # else:
        #     mask_indices = None

        # if self.mask_channel_prob > 0 and not self.mask_channel_before:
        #     if mask_channel_indices is None:
        #         mask_channel_indices = compute_mask_indices(
        #             (B, C),
        #             None,
        #             self.mask_channel_prob,
        #             self.mask_channel_length,
        #             self.mask_channel_selection,
        #             self.mask_channel_other,
        #             no_overlap=self.no_mask_channel_overlap,
        #             min_space=self.mask_channel_min_space,
        #         )
        #         mask_channel_indices = (
        #             torch.from_numpy(mask_channel_indices)
        #             .to(x.device)
        #             .unsqueeze(1)
        #             .expand(-1, T, -1)
        #         )
        #     x = index_put(x, mask_channel_indices, 0)

        # return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):
        """
        generate false data from input data given in random uniform order. Permits to accelerate the learning
        """

        # if no number of negative sample selected, return default y output
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape  # batch_size, number_of_tokens, dim_token
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:

                # creation of tensor range from 0 to max with step 1 repeated on last dimenstion [0 ,1 ,2 , .. max ] => [0,0,1,1,2,2,...,max,max]
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                # random integers between 0 and high -1
                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        # get indexes of negatives selected
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / self.logit_temp
        logits = logits.type_as(x)

        if is_xla_tensor(logits) or neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                fillval = -float(2 ** 30)
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            # put to -inf for logits with index True of neg_is_pos if exist
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):
        ...
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(
                input_lengths
            )

            padding_mask = torch.zeros(
                features.shape[:2],
                dtype=features.dtype,
                device=features.device,
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(
                        padding_mask.shape[0], device=padding_mask.device
                    ),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (
                1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
            ).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, layer_results = self.encoder(
            x, padding_mask=padding_mask, layer=layer
        )

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        if self.quantizer:
            if self.negatives_from_everywhere:
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    padding_count=padding_count,
                )
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        if not is_xla_tensor(x):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False, layer=None):
        res = self.forward(
            source, padding_mask, mask=mask, features_only=True, layer=layer
        )
        return res

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self, last_layer=None):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]], # [[dim, k , stride], ... ]
        dropout: float = 0.0,
        conv_bias: bool = False
        ):
        super().__init__()

        # creation of conv blocks
        in_d = 1 # dim input 
        self.conv_layers = nn.ModuleList() # list of conv blocks
        for i, conv_layer_params in enumerate(conv_layers):
            (dim, k, stride) = conv_layer_params
            conv_layer = nn.Conv1d(in_d, dim, k, stride=stride, bias=conv_bias) # conv layer
            nn.init.kaiming_normal_(conv_layer.weight) # intialisation of conv layer weights
            if i == 0: # group norm for 1st conv block
                conv_block =nn.Sequential(conv_layer,nn.Dropout(p=dropout),Fp32GroupNorm(dim, dim, affine=True),nn.GELU())
            else: # clasical conv block for others
                conv_block = nn.Sequential(conv_layer, nn.Dropout(p=dropout), nn.GELU())
            self.conv_layers.append(conv_block)
            in_d = dim # new input dim for next block


    def forward(self, x):

        x = x.unsqueeze(1) # Returns a new tensor with a dimension of size one inserted at the specified position.

        for conv in self.conv_layers:
            x = conv(x)

        return x


def make_conv_pos(e, k, g):
    pos_conv = nn.Conv1d(e, e, kernel_size=k, padding=k // 2, groups=g,)
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    # TODO Need to implement SamePad
    # pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args):
        if self.layer_type == "transformer":
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=self.encoder_ffn_embed_dim,
                num_attention_heads=self.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                activation_dropout=self.activation_dropout,
                activation_fn=self.activation_fn,
                layer_norm_first=self.layer_norm_first,
            )
        elif self.layer_type == "conformer":
            ...
            # layer = ConformerWav2Vec2EncoderLayer(
            #     embed_dim=self.embedding_dim,
            #     ffn_embed_dim=args.encoder_ffn_embed_dim,
            #     attention_heads=args.encoder_attention_heads,
            #     dropout=args.dropout,
            #     depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
            #     activation_fn="swish",
            #     attn_type=args.attn_type,
            #     use_fp16=args.fp16,
            #     pos_enc_type="abs",
            # )
        if self.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(
        self,
        dropout,
        layer_type,
        encoder_embed_dim,
        required_seq_len_multiple,
        conv_pos_groups,
        conv_pos,
        layer_norm_first,
        encoder_layerdrop,
        pos_conv_depth: int = 1,
    ):
        super().__init__()
        self.dropout = dropout
        self.embedding_dim = encoder_embed_dim
        self.required_seq_len_multiple = required_seq_len_multiple
        self.layer_type = layer_type
        self.layer_norm_first = layer_norm_first
        self.layerdrop = encoder_layerdrop

        if pos_conv_depth > 1:
            num_layers = pos_conv_depth
            k = max(3, conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e, e, kernel_size=k, padding=k // 2, groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            nn.LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim, k, conv_pos_groups, num_layers
            )

        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim, conv_pos, conv_pos_groups,
            )

        self.layers = nn.ModuleList(
            [self.build_encoder_layer() for _ in range(self.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        # TODO Check what this doess
        # self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self, x, padding_mask=None, tgt_layer=None, min_layer=0,
    ):

        # TODO need to implement index_put
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask,
                self.required_seq_len_multiple,
                dim=-1,
                value=True,
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = (
                np.random.random() if self.layerdrop > 0 else 1
            )
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length],
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class ConformerEncoder(TransformerEncoder):
    def build_encoder_layer(self, args):
        ...
        # layer = ConformerWav2Vec2EncoderLayer(
        #     embed_dim=self.embedding_dim,
        #     ffn_embed_dim=args.encoder_ffn_embed_dim,
        #     attention_heads=args.encoder_attention_heads,
        #     dropout=args.dropout,
        #     depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
        #     activation_fn="swish",
        #     attn_type=args.attn_type,
        #     pos_enc_type=args.pos_enc_type,
        #     use_fp16=args.fp16,  # only used for rope
        # )
        # layer = fsdp_wrap(layer)
        # if args.checkpoint_activations:
        #     layer = checkpoint_wrapper(layer)
        # return layer

    def __init__(self, args):
        super().__init__(args)
        # self.args = args
        # self.dropout = args.dropout
        # self.embedding_dim = args.encoder_embed_dim
        # self.pos_enc_type = args.pos_enc_type
        # max_source_positions = self.max_positions()

        # if self.pos_enc_type == "rel_pos":
        #     self.embed_positions = RelPositionalEncoding(
        #         max_source_positions, self.embedding_dim
        #     )
        # elif self.pos_enc_type == "rope":
        #     self.embed_positions = None
        # else:
        #     raise Exception("Unsupported positional encoding type")

        # self.layers = nn.ModuleList(
        #     [
        #         self.build_encoder_layer(args)
        #         for _ in range(args.encoder_layers)
        #     ]
        # )
        # self.layer_norm_first = args.layer_norm_first
        # self.layer_norm = LayerNorm(self.embedding_dim)
        # self.layerdrop = args.encoder_layerdrop

        # self.apply(init_bert_params)

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # B X T X C here
        position_emb = None
        if self.pos_enc_type == "rel_pos":
            position_emb = self.embed_positions(x)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    position_emb=position_emb,
                )
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False
     ) -> None:

        super().__init__()
        # # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        
        

        self.self_attn = nn.MultiheadAttention(self.embedding_dim,
             num_attention_heads,
             dropout=attention_dropout)
        # # old deinfition of multiheadattention (from fairseq)
        # self.self_attn = MultiheadAttention(
        #    self.embedding_dim,
        #     num_attention_heads,
        #     dropout=attention_dropout,
        #     self_attention=True,
        # )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)


class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax
        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(
            torch.FloatTensor(1, num_groups * num_vars, var_dim)
        )
        nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(
                    nn.Linear(input_dim, output_dim), activation
                )

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars ** self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
            .index_select(0, indices)
            .view(self.num_vars ** self.groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
            n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = (
            self.vars.squeeze(0)
            .index_select(0, indices.flatten())
            .view(b, n, -1)
        )
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars ** exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):

        result = {"num_vars": self.num_vars * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(
                x.float(), tau=self.curr_temp, hard=True
            ).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        result["x"] = x

        return result


def checkpoint_wrapper(m, offload_to_cpu=False):
    """
    A friendlier wrapper for performing activation checkpointing.
    Compared to the PyTorch version, this version:
    - wraps an nn.Module, so that all subsequent calls will use checkpointing
    - handles keyword arguments in the forward
    - handles non-Tensor outputs from the forward
    Usage::
        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))
    """
    # should I check whether original_forward has already been set?
    assert not hasattr(
        m, "precheckpoint_forward"
    ), "checkpoint function has already been applied?"
    m.precheckpoint_forward = m.forward
    m.forward = functools.partial(
        _checkpointed_forward,
        m.precheckpoint_forward,  # original_forward
        offload_to_cpu,
    )
    return m


def unwrap_checkpoint(m: torch.nn.Module):
    """
    unwrap a module and its children from checkpoint_wrapper
    """
    for module in m.modules():
        if hasattr(module, "precheckpoint_forward"):
            module.forward = module.precheckpoint_forward
            del module.precheckpoint_forward
        if hasattr(module, "old_deepcopy_method"):
            module.__deepcopy__ = module.old_deepcopy_method
            del module.old_deepcopy_method
    return m


def _checkpointed_forward(original_forward, offload_to_cpu, *args, **kwargs):
    # Autograd Functions in PyTorch work best with positional args, since
    # the backward must return gradients (or None) for every input argument.
    # We can flatten keyword arguments to make this easier.
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict = {"offload": offload_to_cpu}
    output = CheckpointFunction.apply(
        original_forward, parent_ctx_dict, kwarg_keys, *flat_args
    )
    if isinstance(output, torch.Tensor):
        return output
    else:
        packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)
        return output


def pack_kwargs(*args, **kwargs) -> Tuple[List[str], List[Any]]:
    """
    Usage::
        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == [1, 2]
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys = []
    flat_args = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return kwarg_keys, flat_args


def unpack_kwargs(
    kwarg_keys: List[str], flat_args: List[Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs


def split_non_tensors(
    mixed: Union[torch.Tensor, Tuple[Any]]
) -> Tuple[Tuple[torch.Tensor], Dict[str, List[Any]]]:
    """
    Usage::
        x = torch.Tensor([1])
        y = torch.Tensor([2])
        tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
        recon = unpack_non_tensors(tensors, packed_non_tensors)
        assert recon == (x, y, None, 3)
    """
    if isinstance(mixed, torch.Tensor):
        return (mixed,), None
    tensors = []
    packed_non_tensors = {"is_tensor": [], "objects": []}
    for o in mixed:
        if isinstance(o, torch.Tensor):
            packed_non_tensors["is_tensor"].append(True)
            tensors.append(o)
        else:
            packed_non_tensors["is_tensor"].append(False)
            packed_non_tensors["objects"].append(o)
    return tuple(tensors), packed_non_tensors


def unpack_non_tensors(
    tensors: Tuple[torch.Tensor], packed_non_tensors: Dict[str, List[Any]],
) -> Tuple[Any]:
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict)
    mixed = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list)
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.
    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling ``unpack_non_tensors``.
    """

    @staticmethod
    def forward(ctx, run_function, parent_ctx_dict, kwarg_keys, *args):
        if (
            torch.is_grad_enabled()
        ):  # grad may be disabled, e.g., during validation
            checkpoint.check_backward_validity(args)

        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = get_rng_state()

        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        if parent_ctx_dict["offload"]:
            ctx.fwd_device = tuple(x.device for x in tensor_inputs)
            ctx.grad_requirements = tuple(
                x.requires_grad for x in tensor_inputs
            )
            tensor_inputs = tuple(
                x.to(torch.device("cpu"), non_blocking=True)
                for x in tensor_inputs
            )

        else:
            ctx.fwd_device, ctx.grad_requirements = None, None

        ctx.save_for_backward(*tensor_inputs)
        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs

        with torch.no_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)

        if isinstance(outputs, torch.Tensor):
            return outputs
        else:
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict[
                "packed_non_tensor_outputs"
            ] = packed_non_tensor_outputs
            return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        tensor_inputs: Tuple = ctx.saved_tensors
        tensor_inputs = checkpoint.detach_variable(tensor_inputs)
        if ctx.fwd_device is not None:
            tensor_inputs = [
                t.to(ctx.fwd_device[i], non_blocking=True)
                for i, t in enumerate(tensor_inputs)
            ]
            for i, need_grad in enumerate(ctx.grad_requirements):
                tensor_inputs[i].requires_grad = need_grad
        inputs = unpack_non_tensors(tensor_inputs, ctx.packed_non_tensor_inputs)

        # Store the current states.
        bwd_rng_state = get_rng_state()

        # Set the states to what it used to be before the forward pass.
        set_rng_state(ctx.fwd_rng_state)

        with torch.enable_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(
                ctx.kwarg_keys, inputs
            )
            outputs = ctx.run_function(*unpacked_args, **unpacked_kwargs)
            tensor_outputs, _ = split_non_tensors(outputs)
        # Set the states back to what it was at the start of this function.
        set_rng_state(bwd_rng_state)

        # Run backward() with only Tensors that require grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(tensor_outputs)):
            if tensor_outputs[i].requires_grad:
                outputs_with_grad.append(tensor_outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of the outputs have requires_grad=True, "
                "this checkpoint() is not necessary"
            )

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in inputs
        )
        return (None, None, None) + grads













def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""
    from fairseq.modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))



def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )

def relu_squared(x: torch.Tensor):
    return F.relu(x).pow(2)

def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)















#from typing import Optional
#from fairseq.modules import (
#
#    ESPNETMultiHeadedAttention,
#    RelPositionMultiHeadedAttention,
#    RotaryPositionMultiHeadedAttention,
#)
#from fairseq.utils import get_activation_fn


class ConvolutionModule(torch.nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(
        self,
        embed_dim,
        channels,
        depthwise_kernel_size,
        dropout,
        activation_fn="swish",
        bias=False,
        export=False,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            channels: Number of channels in depthwise conv layers
            depthwise_kernel_size: Depthwise conv layer kernel size
            dropout: dropout value
            activation_fn: Activation function to use after depthwise convolution kernel
            bias: If bias should be added to conv layers
            export: If layernorm should be exported to jit
        """
        super(ConvolutionModule, self).__init__()
        assert (
            depthwise_kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = torch.nn.Conv1d(
            embed_dim,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.batch_norm = torch.nn.BatchNorm1d(channels)
        self.activation = get_activation_fn(activation_fn)(channels)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        x = self.layer_norm(x)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in conformer"""

    def __init__(
        self,
        input_feat,
        hidden_units,
        dropout1,
        dropout2,
        activation_fn="swish",
        bias=True,
    ):
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            dropout1: dropout value for layer1
            dropout2: dropout value for layer2
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """

        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn,LayerNorm(input_feat)
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = torch.nn.Dropout(dropout1)
        self.dropout2 = torch.nn.Dropout(dropout2)
        self.activation = get_activation_fn(activation_fn)(hidden_units)

    def forward(self, x):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)


class ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100. We currently don't support relative positional encoding in MHA"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout,
        use_fp16,
        depthwise_conv_kernel_size=31,
        activation_fn="swish",
        attn_type=None,
        pos_enc_type="abs",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            ffn_embed_dim: FFN layer dimension
            attention_heads: Number of attention heads in MHA
            dropout: dropout value
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
            activation_fn: Activation function name to use in convulation block and feed forward block
            attn_type: MHA implementation from ESPNET vs fairseq
            pos_enc_type: Positional encoding type - abs, rope, rel_pos
        """
        self.pos_enc_type = pos_enc_type
        super(ConformerEncoderLayer, self).__init__()

        self.ffn1 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, export=False)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos":
                self.self_attn = RelPositionMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            elif self.pos_enc_type == "rope":
                self.self_attn = RotaryPositionMultiHeadedAttention(
                    embed_dim, attention_heads, dropout=dropout, precision=use_fp16
                )
            elif self.pos_enc_type == "abs":
                self.self_attn = ESPNETMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            else:
                raise Exception(f"Unsupported attention type {self.pos_enc_type}")
        else:
            # Default to fairseq MHA
            self.self_attn = nn.MultiheadAttention(
                embed_dim,
                attention_heads,
                dropout=dropout,
            )

        self.conv_module = ConvolutionModule(
            embed_dim=embed_dim,
            channels=embed_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation_fn=activation_fn,
        )

        self.ffn2 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
            activation_fn=activation_fn,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_padding_mask,
        position_emb = None,
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        if self.pos_enc_type == "rel_pos":
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                pos_emb=position_emb,
                need_weights=False,
            )
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        # TBC to BTC
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        # BTC to TBC
        x = x.transpose(0, 1)
        x = residual + x

        residual = x
        x = self.ffn2(x)

        layer_result = x

        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x, (attn, layer_result)


class ConformerWav2Vec2EncoderLayer(ConformerEncoderLayer):
    """Encoder layer for Wav2vec2 encoder"""

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        position_emb=None,
    ):
        return super().forward(x, self_attn_padding_mask, position_emb)



class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)



def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), re
    
def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"
