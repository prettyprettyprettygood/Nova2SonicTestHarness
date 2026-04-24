"""
Results Manager for Live Interaction Mode
Organizes results with clear structure for easy access.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


class ResultsManager:
    """
    Manages results organization for multiple test sessions.
    Creates a clear, accessible structure for all outputs.
    """

    def __init__(self, base_results_dir: str = "results"):
        """
        Initialize results manager.

        Args:
            base_results_dir: Base directory for all results
        """
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)

        # Standard subdirectories (sessions are created by InteractionLogger)
        self.sessions_dir = self.base_results_dir / "sessions"
        self.summaries_dir = self.base_results_dir / "summaries"
        self.batches_dir = self.base_results_dir / "batches"

        for dir_path in [self.sessions_dir, self.summaries_dir, self.batches_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def register_session(
        self,
        session_id: str,
        test_name: str,
        session_dir: Path,
        batch_id: Optional[str] = None
    ):
        """
        Register a completed session in the master index and write reference files.

        The session directory and its contents are created by InteractionLogger.
        This method only updates indexes and writes README files.

        Args:
            session_id: Session identifier
            test_name: Test name
            session_dir: Path to the session directory (created by InteractionLogger)
            batch_id: Optional batch identifier for grouping sessions
        """
        # Write session metadata
        metadata = {
            "session_id": session_id,
            "test_name": test_name,
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
        }
        metadata_file = session_dir / "session_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create quick reference file
        self._create_quick_reference(session_dir, session_id, test_name)

        # Update master index
        self._update_master_index(session_id, test_name, session_dir, batch_id=batch_id)

        print(f"✅ Session registered: {session_dir}")

    def _create_quick_reference(
        self,
        session_root: Path,
        session_id: str,
        test_name: str,
    ):
        """Create a quick reference file with key information."""
        quick_ref = {
            "session_id": session_id,
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "paths": {
                "chat_transcript": "chat/conversation.txt",
                "interaction_log": "logs/interaction_log.json",
                "audio_files": "audio/",
                "configuration": "config/test_config.json",
                "evaluation": "evaluation/"
            }
        }

        # Add summary statistics if available
        log_file = session_root / "logs" / "interaction_log.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                if "summary" in log_data:
                    quick_ref["statistics"] = log_data["summary"]

        quick_ref_file = session_root / "README.json"
        with open(quick_ref_file, 'w') as f:
            json.dump(quick_ref, f, indent=2)

        # Also create markdown README
        self._create_markdown_readme(session_root, quick_ref)

    def _create_markdown_readme(self, session_root: Path, quick_ref: Dict[str, Any]):
        """Create a markdown README for easy viewing."""
        readme_content = f"""# Session: {quick_ref['session_id']}

## Test Information
- **Test Name**: {quick_ref['test_name']}
- **Timestamp**: {quick_ref['timestamp']}

## Files

### Conversation
- [Chat Transcript](chat/conversation.txt) - Human-readable conversation
- [Interaction Log](logs/interaction_log.json) - Complete structured data

### Media
- [Audio Files](audio/) - Recorded audio (if enabled)

### Configuration
- [Test Config](config/test_config.json) - Test configuration used

### Evaluation
- [Evaluation Results](evaluation/) - LLM judge results (if applicable)

## Statistics
"""

        if "statistics" in quick_ref:
            stats = quick_ref["statistics"]
            readme_content += f"""
- **Total Turns**: {stats.get('total_turns', 'N/A')}
- **Tool Calls**: {stats.get('total_tool_calls', 'N/A')}
- **Tool Usage Rate**: {stats.get('tool_usage_rate', 0):.2%}
- **Avg User Message Length**: {stats.get('avg_user_message_length', 0):.1f} chars
- **Avg Sonic Response Length**: {stats.get('avg_sonic_response_length', 0):.1f} chars
"""

        readme_file = session_root / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)

    def _update_master_index(self, session_id: str, test_name: str, session_path: Path, batch_id: Optional[str] = None):
        """Update the master index of all sessions."""
        index_file = self.base_results_dir / "master_index.json"

        # Load existing index
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {"sessions": []}

        # Remove existing entry for this session (avoid duplicates)
        index["sessions"] = [s for s in index["sessions"] if s.get("session_id") != session_id]

        # Add new session
        entry = {
            "session_id": session_id,
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "path": str(session_path.relative_to(self.base_results_dir))
        }
        if batch_id is not None:
            entry["batch_id"] = batch_id
        index["sessions"].append(entry)

        # Save index
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)

        # Also create/update CSV for easy spreadsheet viewing
        self._update_master_csv(index["sessions"])

    def _update_master_csv(self, sessions: List[Dict[str, Any]]):
        """Create/update a CSV file with session information."""
        csv_file = self.base_results_dir / "sessions_index.csv"

        df = pd.DataFrame(sessions)
        df.to_csv(csv_file, index=False)

    def get_session_path(self, session_id: str) -> Optional[Path]:
        """
        Get the path to a session's results.

        Args:
            session_id: Session identifier

        Returns:
            Path to session directory, or None if not found
        """
        session_path = self.sessions_dir / session_id
        if session_path.exists():
            return session_path
        return None

    def list_sessions(self, test_name: Optional[str] = None, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all sessions, optionally filtered by test name or batch ID.

        Args:
            test_name: Optional test name filter
            batch_id: Optional batch ID filter

        Returns:
            List of session information dictionaries
        """
        index_file = self.base_results_dir / "master_index.json"

        if not index_file.exists():
            return []

        with open(index_file, 'r') as f:
            index = json.load(f)

        sessions = index.get("sessions", [])

        if test_name:
            sessions = [s for s in sessions if s.get("test_name") == test_name]

        if batch_id:
            sessions = [s for s in sessions if s.get("batch_id") == batch_id]

        return sessions

    def get_chat_transcript(self, session_id: str) -> Optional[str]:
        """
        Get the chat transcript for a session.

        Args:
            session_id: Session identifier

        Returns:
            Chat transcript content, or None if not found
        """
        session_path = self.get_session_path(session_id)
        if not session_path:
            return None

        transcript_file = session_path / "chat" / "conversation.txt"
        if transcript_file.exists():
            with open(transcript_file, 'r') as f:
                return f.read()

        return None

    def get_interaction_log(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the interaction log for a session.

        Args:
            session_id: Session identifier

        Returns:
            Interaction log data, or None if not found
        """
        session_path = self.get_session_path(session_id)
        if not session_path:
            return None

        log_file = session_path / "logs" / "interaction_log.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)

        return None

    def generate_batch_summary(
        self,
        batch_id: str,
        session_results: List[Dict[str, Any]],
        cli_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a batch summary with aggregate evaluation stats.

        Args:
            batch_id: Batch identifier
            session_results: List of per-session result dicts from run_scenarios
            cli_args: Optional CLI arguments used for this batch run

        Returns:
            The batch summary dict
        """
        batch_dir = self.batches_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        completed = [r for r in session_results if r.get("status") == "completed"]
        failed = [r for r in session_results if r.get("status") == "failed"]
        evaluated = [r for r in completed if "evaluation" in r]

        # Compute pass/fail counts
        pass_count = 0
        fail_count = 0
        for r in evaluated:
            if r["evaluation"].get("pass_fail"):
                pass_count += 1
            else:
                fail_count += 1

        pass_rate = pass_count / len(evaluated) if evaluated else 0.0

        # Mode-specific aggregation
        metric_pass_rates: Dict[str, Dict[str, Any]] = {}

        # Aggregate per-metric pass rates
        for r in evaluated:
            for aspect, rating in r["evaluation"].get("aspect_ratings", {}).items():
                if aspect not in metric_pass_rates:
                    metric_pass_rates[aspect] = {"total": 0, "passed": 0}
                metric_pass_rates[aspect]["total"] += 1
                if rating == "PASS":
                    metric_pass_rates[aspect]["passed"] += 1

        for metric, counts in metric_pass_rates.items():
            counts["rate"] = round(counts["passed"] / counts["total"], 3) if counts["total"] > 0 else 0.0

        # Build sessions list for the summary
        sessions_list = []
        for r in session_results:
            entry = {
                "session_id": r.get("session_id", "unknown"),
                "config_name": r.get("config_name", "unknown"),
                "test_name": r.get("test_name", "unknown"),
                "status": r.get("status", "unknown"),
            }
            if "evaluation" in r:
                entry["evaluation"] = r["evaluation"]
            if "error" in r:
                entry["error"] = r["error"]
            sessions_list.append(entry)

        summary = {
            "batch_id": batch_id,
            "generated_at": datetime.now().isoformat(),
            "cli_args": cli_args or {},
            "totals": {
                "total_sessions": len(session_results),
                "completed": len(completed),
                "failed": len(failed),
                "evaluated": len(evaluated),
            },
            "evaluation_summary": {
                "pass_count": pass_count,
                "fail_count": fail_count,
                "pass_rate": round(pass_rate, 3),
                "metric_pass_rates": metric_pass_rates,
            },
            "sessions": sessions_list,
        }

        # Write batch_summary.json
        summary_file = batch_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Update the batch index
        self._update_batch_index(batch_id, summary)

        print(f"Batch summary written: {summary_file}")
        return summary

    def _update_batch_index(self, batch_id: str, summary: Dict[str, Any]):
        """
        Update the master batch index file.

        Args:
            batch_id: Batch identifier
            summary: The batch summary dict
        """
        index_file = self.batches_dir / "batch_index.json"

        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {"batches": []}

        # Remove existing entry for this batch (avoid duplicates)
        index["batches"] = [b for b in index["batches"] if b.get("batch_id") != batch_id]

        totals = summary.get("totals", {})
        eval_summary = summary.get("evaluation_summary", {})

        index["batches"].append({
            "batch_id": batch_id,
            "timestamp": summary.get("generated_at", datetime.now().isoformat()),
            "total_sessions": totals.get("total_sessions", 0),
            "completed": totals.get("completed", 0),
            "failed": totals.get("failed", 0),
            "pass_rate": eval_summary.get("pass_rate", 0.0),
            "metric_pass_rates": eval_summary.get("metric_pass_rates", {}),
            "path": f"batches/{batch_id}",
        })

        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)

    def generate_summary_report(self, test_name: Optional[str] = None):
        """
        Generate a summary report across multiple sessions.

        Args:
            test_name: Optional test name filter
        """
        sessions = self.list_sessions(test_name)

        if not sessions:
            print("No sessions found")
            return

        # Collect statistics
        total_sessions = len(sessions)
        total_turns = 0
        total_tool_calls = 0
        sessions_with_errors = 0

        for session_info in sessions:
            session_id = session_info["session_id"]
            log_data = self.get_interaction_log(session_id)

            if log_data:
                summary = log_data.get("summary", {})
                total_turns += summary.get("total_turns", 0)
                total_tool_calls += summary.get("total_tool_calls", 0)

                if log_data.get("errors"):
                    sessions_with_errors += 1

        # Create report
        report = {
            "generated_at": datetime.now().isoformat(),
            "filter": {"test_name": test_name} if test_name else None,
            "statistics": {
                "total_sessions": total_sessions,
                "total_turns": total_turns,
                "total_tool_calls": total_tool_calls,
                "sessions_with_errors": sessions_with_errors,
                "avg_turns_per_session": total_turns / total_sessions if total_sessions > 0 else 0,
                "avg_tool_calls_per_session": total_tool_calls / total_sessions if total_sessions > 0 else 0
            },
            "sessions": sessions
        }

        # Save report
        report_name = f"summary_{test_name or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file = self.summaries_dir / report_name
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Summary report generated: {report_file}")
        return report
