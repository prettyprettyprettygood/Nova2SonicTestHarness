"""
Dataset Loader for Scripted Test Input
Loads test datasets from JSONL files or HuggingFace datasets.
Each entry becomes a separate Nova Sonic test session.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetEntry:
    """A single test scenario from a dataset."""
    id: str
    user_messages: List[str]                                # Sent to Nova Sonic as scripted input
    expected_responses: List[str] = field(default_factory=list)  # Ground truth per turn
    expected_tool_calls: List[Optional[str]] = field(default_factory=list)  # Expected tool per turn
    rubric: Optional[str] = None                            # Evaluation criteria / rubric
    category: Optional[str] = None                          # Test category / axis
    config_overrides: Optional[Dict[str, Any]] = None       # Per-entry config overrides


class DatasetLoader:
    """Loads test datasets and normalizes them to DatasetEntry objects."""

    def load(self, path: str) -> List[DatasetEntry]:
        """
        Load a dataset from a local JSONL file or HuggingFace.

        Args:
            path: Local file path or "hf://org/dataset" for HuggingFace

        Returns:
            List of DatasetEntry objects
        """
        if path.startswith("hf://"):
            return self._load_huggingface(path[5:])
        return self._load_jsonl(path)

    def _load_jsonl(self, path: str) -> List[DatasetEntry]:
        """Load from a local JSONL file. Auto-detects turns vs wide format."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        entries = []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = self._parse_entry(data, idx)
                    if entry.user_messages:
                        entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {idx + 1} (invalid JSON): {e}")
                except Exception as e:
                    print(f"Warning: Skipping line {idx + 1}: {e}")

        print(f"Loaded {len(entries)} entries from {path}")
        return entries

    def _load_huggingface(self, dataset_id: str) -> List[DatasetEntry]:
        """Load from a HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for HuggingFace datasets. "
                "Install it with: pip install datasets"
            )

        print(f"Loading HuggingFace dataset: {dataset_id}")
        ds = load_dataset(dataset_id)

        # Use test split if available, else first available split
        if 'test' in ds:
            split = ds['test']
        else:
            split_name = list(ds.keys())[0]
            split = ds[split_name]
            print(f"Using split: {split_name}")

        entries = []
        for idx, row in enumerate(split):
            data = dict(row)
            entry = self._parse_entry(data, idx)
            if entry.user_messages:
                entries.append(entry)

        print(f"Loaded {len(entries)} entries from hf://{dataset_id}")
        return entries

    def _parse_entry(self, data: dict, idx: int) -> DatasetEntry:
        """Parse a single entry, auto-detecting format."""
        if 'turns' in data:
            return self._parse_turns_format(data, idx)
        elif any(k.startswith('user_turn_') and k.endswith('_transcript') for k in data):
            return self._parse_wide_format(data, idx)
        else:
            raise ValueError(f"Cannot detect format for entry {idx}: "
                           f"expected 'turns' array or 'user_turn_N_transcript' columns")

    def _parse_turns_format(self, data: dict, idx: int) -> DatasetEntry:
        """
        Parse turns array format:
        {"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...", "expected_tool_call": "..."}]}
        """
        turns = data.get('turns', [])
        user_messages = []
        expected_responses = []
        expected_tool_calls = []

        for turn in turns:
            role = turn.get('role', '')
            content = turn.get('content', '')

            if role == 'user':
                user_messages.append(content)
            elif role == 'assistant':
                expected_responses.append(content)
                expected_tool_calls.append(turn.get('expected_tool_call'))

        return DatasetEntry(
            id=data.get('id', f'entry_{idx}'),
            user_messages=user_messages,
            expected_responses=expected_responses,
            expected_tool_calls=expected_tool_calls,
            rubric=data.get('rubric'),
            category=data.get('category') or data.get('axis'),
            config_overrides=data.get('config'),
        )

    def _parse_wide_format(self, data: dict, idx: int) -> DatasetEntry:
        """
        Parse wide format (AudioMC-style):
        {"user_turn_1_transcript": "...", "assistant_turn_1_transcript": "...", ...}
        """
        user_messages = []
        expected_responses = []

        for n in range(1, 21):
            user_key = f'user_turn_{n}_transcript'
            assistant_key = f'assistant_turn_{n}_transcript'

            user_msg = data.get(user_key)
            if user_msg is None:
                break
            user_messages.append(str(user_msg))

            assistant_msg = data.get(assistant_key)
            if assistant_msg is not None:
                expected_responses.append(str(assistant_msg))

        return DatasetEntry(
            id=data.get('id', f'entry_{idx}'),
            user_messages=user_messages,
            expected_responses=expected_responses,
            expected_tool_calls=[None] * len(expected_responses),
            rubric=data.get('rubric'),
            category=data.get('category') or data.get('axis'),
            config_overrides=data.get('config'),
        )
