from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


REQUIRED_MODEL_FILES = (
    "config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _load_transformers() -> tuple[Any, Any, str]:
    try:
        import transformers
        from transformers import RobertaModel, RobertaTokenizerFast
    except Exception as error:  # pragma: no cover - exercised by dependency audit
        raise RuntimeError(
            "Pinned transformers dependency is unavailable; do not substitute another encoder"
        ) from error
    return RobertaModel, RobertaTokenizerFast, str(transformers.__version__)


def build(config_path: Path, *, batch_size: int = 8) -> dict[str, Any]:
    root = project_root()
    config_path = resolve(root, config_path)
    config = load_yaml(config_path)
    model_path = resolve(root, config["roberta"]["local_path"])
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pinned roberta-base snapshot is incomplete ({missing}); no substitute is authorized"
        )
    model_files = {
        name: {
            "bytes": (model_path / name).stat().st_size,
            "sha256": sha256(model_path / name),
        }
        for name in REQUIRED_MODEL_FILES
    }
    paragraph_path = resolve(root, config["outputs"]["paragraph_table"])
    paragraphs = pd.read_parquet(paragraph_path).sort_values(
        ["paragraph_version_id", "token_index"]
    )
    groups = [
        (str(paragraph_id), group["word_text"].astype(str).tolist())
        for paragraph_id, group in paragraphs.groupby("paragraph_version_id", sort=True)
    ]
    paragraph_ids = [item[0] for item in groups]
    word_lists = [item[1] for item in groups]
    if not groups:
        raise ValueError("No canonical paragraphs")
    max_words = max(map(len, word_lists))
    RobertaModel, RobertaTokenizerFast, transformers_version = _load_transformers()
    tokenizer = RobertaTokenizerFast.from_pretrained(
        model_path, local_files_only=True, add_prefix_space=True
    )
    model = RobertaModel.from_pretrained(
        model_path,
        local_files_only=True,
        use_safetensors=True,
        add_pooling_layer=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_size = int(model.config.hidden_size)
    output = np.zeros((len(groups), max_words, hidden_size), dtype=np.float16)
    subword_lengths: list[int] = []
    unmapped_words: list[dict[str, Any]] = []
    torch.set_num_threads(min(8, torch.get_num_threads()))
    with torch.inference_mode():
        for start in range(0, len(groups), batch_size):
            batch_words = word_lists[start : start + batch_size]
            encoded = tokenizer(
                batch_words,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=False,
                return_attention_mask=True,
            )
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            subword_lengths.extend(int(value) for value in lengths)
            if max(lengths) > int(model.config.max_position_embeddings):
                raise ValueError(
                    "A paragraph exceeds roberta-base positional capacity; truncation is unauthorized"
                )
            hidden = model(**encoded).last_hidden_state.detach().cpu()
            for local_index, words in enumerate(batch_words):
                word_ids = encoded.word_ids(batch_index=local_index)
                for word_index in range(len(words)):
                    positions = [
                        token_index
                        for token_index, mapped in enumerate(word_ids)
                        if mapped == word_index
                    ]
                    if not positions:
                        unmapped_words.append(
                            {
                                "paragraph_version_id": paragraph_ids[start + local_index],
                                "word_index": word_index,
                                "word": words[word_index],
                            }
                        )
                        continue
                    output[start + local_index, word_index] = (
                        hidden[local_index, positions].mean(dim=0).numpy().astype(np.float16)
                    )
    if unmapped_words:
        raise ValueError(f"RoBERTa failed to align canonical words: {unmapped_words[:5]}")
    array_path = resolve(root, config["roberta"]["cache_array"])
    ids_path = resolve(root, config["roberta"]["cache_ids"])
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_path, output, allow_pickle=False)
    ids_path.write_text(json.dumps(paragraph_ids, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "goalexit-roberta-word-cache-v1",
        "status": "pass",
        "repo_id": config["roberta"]["repo_id"],
        "resolved_revision": config["roberta"]["revision"],
        "local_model_path": str(model_path),
        "local_files_only": True,
        "roberta_frozen": True,
        "unused_pooler_created": False,
        "tokenizer": {
            "class": "RobertaTokenizerFast",
            "add_prefix_space": True,
            "is_split_into_words": True,
            "truncation": False,
            "subword_aggregation": "arithmetic mean of last hidden states",
        },
        "model_files": model_files,
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "paragraph_table_path": str(paragraph_path),
        "paragraph_table_sha256": sha256(paragraph_path),
        "paragraphs": len(groups),
        "maximum_words": max_words,
        "maximum_subword_tokens_including_special_tokens": max(subword_lengths),
        "hidden_size": hidden_size,
        "dtype": "float16 storage; float32 fusion input",
        "array_path": str(array_path),
        "array_shape": list(output.shape),
        "array_bytes": array_path.stat().st_size,
        "array_sha256": sha256(array_path),
        "paragraph_ids_path": str(ids_path),
        "paragraph_ids_sha256": sha256(ids_path),
        "unmapped_words": 0,
        "assertions": {
            "exact_pinned_revision": True,
            "no_network_during_encoding": True,
            "no_truncation": True,
            "all_words_aligned": True,
            "encoder_parameters_frozen": True,
        },
    }
    manifest_path = resolve(root, config["outputs"]["roberta_manifest"])
    save_json(manifest_path, manifest)
    print(
        f"status=pass paragraphs={len(groups)} max_subwords={max(subword_lengths)} "
        f"cache_sha256={manifest['array_sha256']}"
    )
    return manifest


def load_cache(array_path: Path, ids_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    values = np.load(array_path, mmap_mode="r")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if values.ndim != 3 or len(ids) != values.shape[0]:
        raise ValueError("Invalid RoBERTa word cache")
    return values, {str(value): index for index, value in enumerate(ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/development_goalexit.yaml"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    build(args.config, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
