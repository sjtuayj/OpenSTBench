# coding=utf-8
# Copyright 2024 Google LLC
# Copyright 2026 OpenSTBench contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MetricX inference helpers.

This module follows the input formatting and scoring semantics described in
the official google-research/metricx README. MetricX is a text metric; it does
not consume audio.
"""

import copy
import dataclasses
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
from torch import nn
import transformers
import transformers.modeling_outputs

from ._model_loading import resolve_pretrained_source


DEFAULT_METRICX_VERSION = "24"
DEFAULT_METRICX_TOKENIZER = "google/mt5-xl"
DEFAULT_METRICX24_MODEL = "google/metricx-24-hybrid-large-v2p6"
DEFAULT_METRICX23_MODEL = "google/metricx-23-large-v2p0"
DEFAULT_METRICX23_QE_MODEL = "google/metricx-23-qe-large-v2p0"
DEFAULT_METRICX24_MAX_INPUT_LENGTH = 1536
DEFAULT_METRICX23_MAX_INPUT_LENGTH = 1024
METRICX_METRIC_NAME = "MetricX"
METRICX_QE_METRIC_NAME = "MetricX_QE"


BaseModelOutput = transformers.modeling_outputs.BaseModelOutput
ModelOutput = transformers.modeling_outputs.ModelOutput
MT5Config = transformers.models.mt5.modeling_mt5.MT5Config
MT5PreTrainedModel = transformers.models.mt5.modeling_mt5.MT5PreTrainedModel
MT5Stack = transformers.models.mt5.modeling_mt5.MT5Stack
_MT5_MODULE = transformers.models.mt5.modeling_mt5
__HEAD_MASK_WARNING_MSG = getattr(
    _MT5_MODULE,
    "__HEAD_MASK_WARNING_MSG",
    "The input argument `head_mask` was split into `head_mask` and `decoder_head_mask`.",
)


@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None


class MT5ForRegression(MT5PreTrainedModel):
    """MT5 model for regression, adapted from google-research/metricx."""

    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.post_init()

        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], MT5ForRegressionOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        batch_size = input_ids.size(0)
        decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)
        decoder_input_ids = decoder_input_ids.to(input_ids.device)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        predictions = lm_logits[:, 0, 250089]
        predictions = torch.clamp(predictions, 0, 25)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            labels = labels.to(predictions.device)
            loss = loss_fct(predictions.view(-1), labels.view(-1))

        return MT5ForRegressionOutput(loss=loss, predictions=predictions)


def default_metricx_model(version: str, *, qe: bool = False) -> str:
    normalized = str(version)
    if normalized == "24":
        return DEFAULT_METRICX24_MODEL
    if normalized == "23":
        return DEFAULT_METRICX23_QE_MODEL if qe else DEFAULT_METRICX23_MODEL
    raise ValueError("metricx_version must be '24' or '23'")


def default_metricx_max_input_length(version: str) -> int:
    normalized = str(version)
    if normalized == "24":
        return DEFAULT_METRICX24_MAX_INPUT_LENGTH
    if normalized == "23":
        return DEFAULT_METRICX23_MAX_INPUT_LENGTH
    raise ValueError("metricx_version must be '24' or '23'")


def build_metricx_records(
    *,
    version: str,
    candidates: List[str],
    sources: Optional[List[str]] = None,
    references: Optional[List[str]] = None,
    qe: bool = False,
) -> List[Dict[str, str]]:
    """Build MetricX records following the official README field rules."""

    normalized = str(version)
    if normalized not in {"23", "24"}:
        raise ValueError("metricx_version must be '24' or '23'")

    records = []
    for idx, candidate in enumerate(candidates):
        if normalized == "24":
            source = sources[idx] if sources is not None else ""
            reference = "" if qe else (references[idx] if references is not None else "")
            records.append(
                {
                    "source": source,
                    "hypothesis": candidate,
                    "reference": reference,
                }
            )
        elif qe:
            if sources is None:
                raise ValueError("MetricX-23 QE requires source text")
            records.append(
                {
                    "source": sources[idx],
                    "hypothesis": candidate,
                }
            )
        else:
            if references is None:
                raise ValueError("MetricX-23 reference scoring requires reference text")
            records.append(
                {
                    "hypothesis": candidate,
                    "reference": references[idx],
                }
            )
    return records


def build_metricx_inputs(records: List[Dict[str, str]], *, version: str, qe: bool = False) -> List[str]:
    """Build the exact input strings used by official MetricX predict scripts."""

    normalized = str(version)
    if normalized == "24":
        if qe:
            return [f"source: {record['source']} candidate: {record['hypothesis']}" for record in records]
        return [
            f"source: {record['source']} candidate: {record['hypothesis']} reference: {record['reference']}"
            for record in records
        ]

    if normalized == "23":
        if qe:
            return [f"candidate: {record['hypothesis']} source: {record['source']}" for record in records]
        return [f"candidate: {record['hypothesis']} reference: {record['reference']}" for record in records]

    raise ValueError("metricx_version must be '24' or '23'")


class MetricXScorer:
    """Lazy MetricX scorer for reference-based and QE text scoring."""

    def __init__(
        self,
        *,
        version: str = DEFAULT_METRICX_VERSION,
        model: Optional[str] = None,
        qe_model: Optional[str] = None,
        tokenizer: str = DEFAULT_METRICX_TOKENIZER,
        max_input_length: Optional[int] = None,
        batch_size: int = 1,
        device: str = "cpu",
    ):
        self.version = str(version)
        if self.version not in {"23", "24"}:
            raise ValueError("metricx_version must be '24' or '23'")

        self.model_name = model or default_metricx_model(self.version, qe=False)
        self.qe_model_name = qe_model or default_metricx_model(self.version, qe=True)
        self.tokenizer_name = tokenizer
        self.max_input_length = max_input_length or default_metricx_max_input_length(self.version)
        self.batch_size = max(1, int(batch_size))
        self.device = torch.device(device)
        self.tokenizer = None
        self.models: Dict[str, MT5ForRegression] = {}

    def _load_tokenizer(self):
        if self.tokenizer is None:
            tokenizer_source, _source_kind = resolve_pretrained_source(self.tokenizer_name)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source)
        return self.tokenizer

    def _load_model(self, *, qe: bool = False):
        model_name = self.qe_model_name if qe and self.version == "23" else self.model_name
        cache_key = model_name
        if cache_key not in self.models:
            model_source, _source_kind = resolve_pretrained_source(model_name)
            if self.version == "24":
                model = MT5ForRegression.from_pretrained(model_source, torch_dtype="auto")
            else:
                model = MT5ForRegression.from_pretrained(model_source)
            model.to(self.device)
            model.eval()
            self.models[cache_key] = model
        return self.models[cache_key]

    def _tokenize_batch(self, inputs: List[str]):
        tokenizer = self._load_tokenizer()
        features = []
        for text in inputs:
            encoded = tokenizer(
                text,
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
            )
            feature = {
                "input_ids": list(encoded["input_ids"]),
                "attention_mask": list(encoded["attention_mask"]),
            }
            feature["input_ids"] = feature["input_ids"][:-1]
            feature["attention_mask"] = feature["attention_mask"][:-1]
            features.append(feature)
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in batch.items()}

    def _predict(self, inputs: List[str], *, qe: bool = False) -> List[float]:
        if not inputs:
            return []

        model = self._load_model(qe=qe)
        scores = []
        for start in range(0, len(inputs), self.batch_size):
            batch_inputs = inputs[start : start + self.batch_size]
            batch = self._tokenize_batch(batch_inputs)
            with torch.no_grad():
                output = model(**batch)
            scores.extend(output.predictions.detach().cpu().float().tolist())
        return scores

    def score_reference(
        self,
        *,
        candidates: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
    ) -> float:
        records = build_metricx_records(
            version=self.version,
            candidates=candidates,
            sources=sources,
            references=references,
            qe=False,
        )
        inputs = build_metricx_inputs(records, version=self.version, qe=False)
        return float(np.mean(self._predict(inputs, qe=False)))

    def score_qe(self, *, candidates: List[str], sources: List[str]) -> float:
        records = build_metricx_records(
            version=self.version,
            candidates=candidates,
            sources=sources,
            references=[""] * len(candidates),
            qe=True,
        )
        inputs = build_metricx_inputs(records, version=self.version, qe=True)
        return float(np.mean(self._predict(inputs, qe=True)))

    def cleanup(self):
        self.models.clear()
        self.tokenizer = None
