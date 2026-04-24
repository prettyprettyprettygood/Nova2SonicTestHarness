"""
Multi-Session Runner for Batch Testing
Runs multiple test sessions with different configurations.
"""

import asyncio
import copy
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.config_manager import ConfigManager, TestConfig
from runners.dataset_loader import DatasetEntry
from logging_.results_manager import ResultsManager
from main import LiveInteractionSession


class MultiSessionRunner:
    """
    Runs multiple test sessions in batch mode.
    Supports parallel execution and automatic evaluation.
    """

    def __init__(
        self,
        results_manager: Optional[ResultsManager] = None,
        auto_evaluate: bool = True,
        parallel_sessions: int = 1,
        log_judge_prompts: bool = False,
    ):
        """
        Initialize multi-session runner.

        Args:
            results_manager: Results manager instance
            auto_evaluate: Whether to automatically evaluate sessions
            parallel_sessions: Number of sessions to run in parallel
            log_judge_prompts: Whether to log judge prompts and raw responses
        """
        self.results_manager = results_manager or ResultsManager()
        self.auto_evaluate = auto_evaluate
        self.parallel_sessions = max(1, parallel_sessions)
        self.config_manager = ConfigManager()
        self.last_batch_id: Optional[str] = None
        self._log_judge_prompts = log_judge_prompts

    async def run_scenarios(
        self,
        scenario_configs: List[Path],
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run multiple scenario configurations.

        Args:
            scenario_configs: List of paths to scenario configuration files
            config_overrides: Optional dict of config field overrides (test_name, max_turns, mode)

        Returns:
            List of session results
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.last_batch_id = batch_id

        print(f"\n{'='*60}")
        print(f"Multi-Session Runner")
        print(f"Batch ID: {batch_id}")
        print(f"Scenarios: {len(scenario_configs)}")
        print(f"Parallel: {self.parallel_sessions}")
        print(f"Auto-evaluate: {self.auto_evaluate}")
        print(f"{'='*60}\n")

        results = []
        pending_evals: list[tuple[dict, LiveInteractionSession]] = []

        # Phase 1: Run all conversations (evals fire as background tasks)
        for i in range(0, len(scenario_configs), self.parallel_sessions):
            batch = scenario_configs[i:i + self.parallel_sessions]

            print(f"\n🔄 Running batch {i // self.parallel_sessions + 1}/{(len(scenario_configs) + self.parallel_sessions - 1) // self.parallel_sessions}")

            batch_results = await asyncio.gather(*[
                self._run_single_scenario(config_path, batch_id=batch_id, config_overrides=config_overrides)
                for config_path in batch
            ], return_exceptions=True)

            for config_path, raw in zip(batch, batch_results):
                if isinstance(raw, Exception):
                    print(f"❌ Failed: {config_path.name} - {str(raw)}")
                    results.append({
                        "config": str(config_path),
                        "status": "failed",
                        "error": str(raw),
                        "batch_id": batch_id,
                    })
                else:
                    result_dict, session = raw
                    results.append(result_dict)
                    if session is not None:
                        pending_evals.append((result_dict, session))

        # Phase 2: Await all background evaluations
        if pending_evals:
            print(f"\n⏳ Awaiting {len(pending_evals)} evaluation(s)...")
            eval_outcomes = await asyncio.gather(*[
                session.await_evaluation() for _, session in pending_evals
            ], return_exceptions=True)
            for (result_dict, _), eval_out in zip(pending_evals, eval_outcomes):
                if isinstance(eval_out, Exception):
                    print(f"⚠️  Evaluation failed for {result_dict.get('session_id', '?')}: {eval_out}")
                elif eval_out:
                    result_dict["evaluation"] = eval_out

        # Generate batch summary
        cli_args = {
            "total_scenarios": len(scenario_configs),
            "parallel_sessions": self.parallel_sessions,
            "auto_evaluate": self.auto_evaluate,
        }
        self.results_manager.generate_batch_summary(batch_id, results, cli_args=cli_args)

        # Print console summary
        self._print_summary(results, batch_id=batch_id)

        return results

    async def _run_single_scenario(
        self,
        config_path: Path,
        batch_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> tuple[Dict[str, Any], Optional[LiveInteractionSession]]:
        """Run a single scenario.

        Returns (result_dict, session) — the session is returned so the caller
        can await its background evaluation task after all conversations finish.
        """
        print(f"\n📝 Starting: {config_path.name}")

        # Load configuration
        config = self.config_manager.load_config(str(config_path))

        # Apply CLI overrides
        if config_overrides:
            if config_overrides.get('test_name'):
                config.test_name = config_overrides['test_name']
            if config_overrides.get('max_turns') is not None:
                config.max_turns = config_overrides['max_turns']
            if config_overrides.get('mode'):
                config.interaction_mode = config_overrides['mode']
            if config_overrides.get('output_directory'):
                config.output_directory = config_overrides['output_directory']
            if config_overrides.get('log_events'):
                config.log_events = True

        # Apply judge logging flag
        if self._log_judge_prompts:
            config.log_judge_prompts = True

        # Generate a unique session_id to avoid collisions in parallel runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        session_id = f"{config.test_name}_{timestamp}_{short_uuid}"

        # Create session — evaluation fires as background task after conversation
        session = LiveInteractionSession(
            config,
            auto_evaluate=self.auto_evaluate,
            evaluation_criteria=config.evaluation_criteria,
            session_id=session_id,
        )

        # Run session (conversation + cleanup; eval runs in background)
        try:
            await session.run(batch_id=batch_id)

            session_id = session.logger.session_id if session.logger else "unknown"
            log_path = session.logger.session_dir if session.logger else None

            result = {
                "config": str(config_path),
                "config_name": config_path.stem,
                "test_name": config.test_name,
                "session_id": session_id,
                "status": "completed",
                "log_path": str(log_path) if log_path else None,
                "batch_id": batch_id,
            }

            print(f"✅ Completed: {config_path.name}")
            return result, session

        except Exception as e:
            print(f"❌ Error in {config_path.name}: {str(e)}")
            raise

    async def run_dataset(
        self,
        entries: List[DatasetEntry],
        base_config: Optional[TestConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run dataset entries as scripted test sessions.

        Each DatasetEntry becomes a separate Nova Sonic session with
        interaction_mode="scripted" and ground truth passed to evaluation.

        Args:
            entries: List of DatasetEntry objects from DatasetLoader
            base_config: Optional base TestConfig (entries can override fields)

        Returns:
            List of session results
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.last_batch_id = batch_id

        print(f"\n{'='*60}")
        print(f"Dataset Runner")
        print(f"Batch ID: {batch_id}")
        print(f"Entries: {len(entries)}")
        print(f"Parallel: {self.parallel_sessions}")
        print(f"Auto-evaluate: {self.auto_evaluate}")
        print(f"{'='*60}\n")

        results = []
        pending_evals: list[tuple[dict, LiveInteractionSession]] = []

        # Phase 1: Run all conversations
        for i in range(0, len(entries), self.parallel_sessions):
            batch = entries[i:i + self.parallel_sessions]

            print(f"\n🔄 Running batch {i // self.parallel_sessions + 1}/{(len(entries) + self.parallel_sessions - 1) // self.parallel_sessions}")

            batch_results = await asyncio.gather(*[
                self._run_dataset_entry(entry, base_config, batch_id=batch_id)
                for entry in batch
            ], return_exceptions=True)

            for entry, raw in zip(batch, batch_results):
                if isinstance(raw, Exception):
                    print(f"❌ Failed: {entry.id} - {str(raw)}")
                    results.append({
                        "entry_id": entry.id,
                        "category": entry.category,
                        "status": "failed",
                        "error": str(raw),
                        "batch_id": batch_id,
                    })
                else:
                    result_dict, session = raw
                    results.append(result_dict)
                    if session is not None:
                        pending_evals.append((result_dict, session))

        # Phase 2: Await all background evaluations
        if pending_evals:
            print(f"\n⏳ Awaiting {len(pending_evals)} evaluation(s)...")
            eval_outcomes = await asyncio.gather(*[
                session.await_evaluation() for _, session in pending_evals
            ], return_exceptions=True)
            for (result_dict, _), eval_out in zip(pending_evals, eval_outcomes):
                if isinstance(eval_out, Exception):
                    print(f"⚠️  Evaluation failed for {result_dict.get('session_id', '?')}: {eval_out}")
                elif eval_out:
                    result_dict["evaluation"] = eval_out

        # Generate batch summary
        cli_args = {
            "total_entries": len(entries),
            "parallel_sessions": self.parallel_sessions,
            "auto_evaluate": self.auto_evaluate,
        }
        self.results_manager.generate_batch_summary(batch_id, results, cli_args=cli_args)

        # Print console summary
        self._print_summary(results, batch_id=batch_id)

        return results

    async def _run_dataset_entry(
        self,
        entry: DatasetEntry,
        base_config: Optional[TestConfig],
        batch_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Optional[LiveInteractionSession]]:
        """Run a single dataset entry as a scripted session.

        Returns (result_dict, session) so the caller can await background eval.
        """
        print(f"\n📝 Starting entry: {entry.id}" + (f" [{entry.category}]" if entry.category else ""))

        # Build config: start from base or create a default
        if base_config:
            config_dict = base_config.to_dict()
        else:
            config_dict = TestConfig(test_name="dataset_test").to_dict()

        # Force scripted mode with the entry's user messages
        config_dict['interaction_mode'] = 'scripted'
        config_dict['scripted_messages'] = entry.user_messages
        config_dict['max_turns'] = len(entry.user_messages)
        config_dict['test_name'] = entry.id

        # Apply per-entry config overrides
        if entry.config_overrides:
            config_dict.update(entry.config_overrides)

        # Apply judge logging flag
        if self._log_judge_prompts:
            config_dict['log_judge_prompts'] = True

        config = TestConfig.from_dict(config_dict)

        # Generate a unique session_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        session_id = f"{entry.id}_{timestamp}_{short_uuid}"

        # Create session with ground truth
        session = LiveInteractionSession(
            config,
            auto_evaluate=self.auto_evaluate,
            evaluation_criteria=config.evaluation_criteria,
            session_id=session_id,
            expected_responses=entry.expected_responses or None,
            expected_tool_calls=entry.expected_tool_calls or None,
            rubric=entry.rubric,
        )

        try:
            await session.run(batch_id=batch_id)

            session_id = session.logger.session_id if session.logger else "unknown"
            log_path = session.logger.session_dir if session.logger else None

            result = {
                "entry_id": entry.id,
                "category": entry.category,
                "test_name": config.test_name,
                "session_id": session_id,
                "status": "completed",
                "log_path": str(log_path) if log_path else None,
                "batch_id": batch_id,
            }

            print(f"✅ Completed: {entry.id}")
            return result, session

        except Exception as e:
            print(f"❌ Error in {entry.id}: {str(e)}")
            raise

    def _print_summary(self, results: List[Dict[str, Any]], batch_id: Optional[str] = None):
        """Print summary of all runs."""
        print(f"\n{'='*60}")
        if batch_id:
            print(f"Batch Summary: {batch_id}")
        else:
            print(f"Multi-Session Summary")
        print(f"{'='*60}")

        completed = [r for r in results if r.get("status") == "completed"]
        failed = [r for r in results if r.get("status") == "failed"]

        print(f"\nResults:")
        print(f"   Total: {len(results)} | Completed: {len(completed)} | Failed: {len(failed)}")

        if completed and self.auto_evaluate:
            evaluated = [r for r in completed if "evaluation" in r]
            if evaluated:
                passed = sum(1 for r in evaluated if r["evaluation"]["pass_fail"])
                pass_rate = passed / len(evaluated) * 100
                print(f"\nEvaluation:")
                print(f"   Passed: {passed}/{len(evaluated)} ({pass_rate:.1f}%)")

                # Per-metric pass rates
                metric_totals: Dict[str, int] = {}
                metric_passes: Dict[str, int] = {}
                for r in evaluated:
                    for aspect, rating in r["evaluation"].get("aspect_ratings", {}).items():
                        metric_totals[aspect] = metric_totals.get(aspect, 0) + 1
                        if rating == "PASS":
                            metric_passes[aspect] = metric_passes.get(aspect, 0) + 1

                if metric_totals:
                    max_count = max(metric_totals.values()) if metric_totals else 1
                    bar_scale = 30

                    print(f"\n   Metric Pass Rates:")
                    for metric, total in metric_totals.items():
                        passes = metric_passes.get(metric, 0)
                        pct = passes / total * 100 if total > 0 else 0
                        bar_len = int(passes / max_count * bar_scale) if max_count > 0 else 0
                        bar = "\u2588" * bar_len
                        print(f"   {metric:<25}: {bar} {passes}/{total} ({pct:.1f}%)")

        if failed:
            print(f"\nFailed Sessions:")
            for r in failed:
                print(f"   - {Path(r['config']).name}: {r.get('error', 'Unknown error')}")

        if batch_id:
            batch_path = self.results_manager.batches_dir / batch_id / "batch_summary.json"
            print(f"\nBatch results: {batch_path}")

        print(f"{'='*60}\n")


