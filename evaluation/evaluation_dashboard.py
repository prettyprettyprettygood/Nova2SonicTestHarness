"""
Evaluation Dashboard for Nova Sonic Test Harness.

Visualizes batch results, session details, error patterns, and batch comparisons
for binary (PASS/FAIL) evaluation mode.

Usage:
    streamlit run evaluation_dashboard.py
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import os

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
BATCHES_DIR = RESULTS_DIR / "batches"
SESSIONS_DIR = RESULTS_DIR / "sessions"


def _get_results_dir() -> Path:
    """Get the current results directory from session state, query params, or default."""
    # Priority: session_state > query_param > env var > default
    if "results_dir" in st.session_state and st.session_state.results_dir:
        return Path(st.session_state.results_dir)
    qp = st.query_params.get("results_dir", "")
    if qp:
        return Path(qp)
    return RESULTS_DIR


def _dirs() -> tuple[Path, Path, Path]:
    """Return (results_dir, batches_dir, sessions_dir) based on current config."""
    rd = _get_results_dir()
    return rd, rd / "batches", rd / "sessions"

RATING_COLORS = {
    "FAIL": "#e74c3c",
    "PASS": "#2ecc71",
}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_batch_index(_batches_dir: str = "") -> list[dict]:
    bd = Path(_batches_dir) if _batches_dir else _dirs()[1]
    path = bd / "batch_index.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("batches", [])


@st.cache_data(ttl=60)
def load_batch_summary(batch_id: str, _batches_dir: str = "") -> dict | None:
    bd = Path(_batches_dir) if _batches_dir else _dirs()[1]
    path = bd / batch_id / "batch_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_session_evaluation(session_id: str, _sessions_dir: str = "") -> dict | None:
    sd = Path(_sessions_dir) if _sessions_dir else _dirs()[2]
    path = sd / session_id / "evaluation" / "llm_judge_evaluation.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_interaction_log(session_id: str, _sessions_dir: str = "") -> dict | None:
    sd = Path(_sessions_dir) if _sessions_dir else _dirs()[2]
    path = sd / session_id / "logs" / "interaction_log.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_chat_transcript(session_id: str, _sessions_dir: str = "") -> str | None:
    sd = Path(_sessions_dir) if _sessions_dir else _dirs()[2]
    path = sd / session_id / "chat" / "conversation.txt"
    if not path.exists():
        return None
    return path.read_text()


@st.cache_data(ttl=60)
def load_session_readme(session_id: str, _sessions_dir: str = "") -> dict | None:
    sd = Path(_sessions_dir) if _sessions_dir else _dirs()[2]
    path = sd / session_id / "README.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_all_sessions(_sessions_dir: str = "") -> list[dict]:
    """Scan sessions directory and return metadata for every session."""
    sd = Path(_sessions_dir) if _sessions_dir else _dirs()[2]
    sessions = []
    if not sd.exists():
        return sessions
    for session_dir in sorted(sd.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name

        # Try README.json first (fast), fall back to interaction log
        readme_path = session_dir / "README.json"
        log_path = session_dir / "logs" / "interaction_log.json"

        test_name = ""
        total_turns = 0
        timestamp = ""
        has_eval = False
        eval_rating = ""

        if readme_path.exists():
            try:
                with open(readme_path) as f:
                    readme = json.load(f)
                test_name = readme.get("test_name", "")
                stats = readme.get("statistics", {})
                total_turns = stats.get("total_turns", 0)
                timestamp = readme.get("start_time", "")
            except (json.JSONDecodeError, KeyError):
                pass
        elif log_path.exists():
            try:
                with open(log_path) as f:
                    log = json.load(f)
                test_name = log.get("test_name", "")
                total_turns = log.get("total_turns", 0)
                timestamp = log.get("start_time", "")
            except (json.JSONDecodeError, KeyError):
                pass
        else:
            continue  # no usable data

        # Check for evaluation
        eval_path = session_dir / "evaluation" / "llm_judge_evaluation.json"
        if eval_path.exists():
            has_eval = True
            try:
                with open(eval_path) as f:
                    ev = json.load(f)
                eval_rating = ev.get("results", {}).get("overall_rating", "")
            except (json.JSONDecodeError, KeyError):
                pass

        sessions.append({
            "session_id": session_id,
            "test_name": test_name,
            "timestamp": timestamp,
            "total_turns": total_turns,
            "has_eval": has_eval,
            "eval_rating": eval_rating,
        })
    return sessions




def rating_badge(rating: str) -> str:
    """Return a colored markdown badge for a rating."""
    color = RATING_COLORS.get(rating, "#95a5a6")
    return f'<span style="background-color:{color};color:white;padding:2px 10px;border-radius:12px;font-weight:bold;font-size:0.9em">{rating}</span>'


def format_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(ts)


def navigate(page: str, **params):
    """Set query params and rerun to navigate."""
    st.query_params["page"] = page
    for k, v in params.items():
        st.query_params[k] = v
    st.rerun()


def batch_label(b: dict) -> str:
    """Rich label for a batch: date | N sessions | pass%."""
    ts = format_timestamp(b.get("timestamp", ""))
    n = b.get("total_sessions", 0)
    pr = b.get("pass_rate", 0)
    return f"{b['batch_id']}  ({ts} | {n} sessions | {pr:.0%})"


SESSIONS_PAGE_SIZE = 50


# ---------------------------------------------------------------------------
# Page: Batch Overview
# ---------------------------------------------------------------------------

def page_batch_overview():
    st.header("Batch Overview")

    batches = load_batch_index()
    if not batches:
        st.warning("No batches found. Run some tests first.")
        return

    # Sort by timestamp descending
    batches_sorted = sorted(batches, key=lambda b: b.get("timestamp", ""), reverse=True)

    # KPI row
    total_sessions = sum(b.get("total_sessions", 0) for b in batches)
    pass_rates = [b["pass_rate"] for b in batches if b.get("pass_rate") is not None]
    avg_pass = sum(pass_rates) / len(pass_rates) if pass_rates else 0
    latest = batches_sorted[0] if batches_sorted else {}
    prev = batches_sorted[1] if len(batches_sorted) > 1 else {}
    latest_delta = (latest.get("pass_rate", 0) - prev.get("pass_rate", 0)) if prev else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Batches", len(batches))
    c2.metric("Total Sessions", total_sessions)
    c3.metric("Avg Pass Rate", f"{avg_pass:.1%}")
    c4.metric("Latest Pass Rate", f"{latest.get('pass_rate', 0):.1%}",
              delta=f"{latest_delta:+.1%}" if latest_delta is not None else None)

    # Pass Rate Trend
    st.subheader("Pass Rate Trend")
    trend_data = []
    for b in sorted(batches, key=lambda x: x.get("timestamp", "")):
        trend_data.append({
            "Timestamp": format_timestamp(b.get("timestamp", "")),
            "Pass Rate": b.get("pass_rate", 0),
            "Batch ID": b["batch_id"],
            "Sessions": b.get("total_sessions", 0),
        })
    if trend_data:
        df_trend = pd.DataFrame(trend_data)
        fig = px.line(
            df_trend, x="Timestamp", y="Pass Rate",
            hover_data=["Batch ID", "Sessions"],
            markers=True,
        )
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Batch Table — click a row to navigate
    st.subheader("Batches")
    st.caption("Click a row to view batch details.")
    table_data = []
    for b in batches_sorted:
        summary = load_batch_summary(b["batch_id"])
        table_data.append({
            "Batch ID": b["batch_id"],
            "Timestamp": format_timestamp(b.get("timestamp", "")),
            "Sessions": b.get("total_sessions", 0),
            "Completed": b.get("completed", 0),
            "Failed": b.get("failed", 0),
            "Pass Rate": b.get("pass_rate", 0),
        })

    df_batches = pd.DataFrame(table_data)
    event = st.dataframe(
        df_batches,
        column_config={
            "Pass Rate": st.column_config.ProgressColumn(
                "Pass Rate", format="%.0f%%", min_value=0, max_value=1,
            ),
        },
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="batch_overview_table",
    )

    sel_rows = event.selection.rows if event.selection else []
    if sel_rows:
        selected_bid = table_data[sel_rows[0]]["Batch ID"]
        navigate("batch_detail", batch_id=selected_bid)


# ---------------------------------------------------------------------------
# Page: Batch Detail
# ---------------------------------------------------------------------------

def page_batch_detail():
    batch_id = st.query_params.get("batch_id", "")
    if not batch_id:
        st.warning("No batch selected. Go to Batch Overview to select one.")
        return

    summary = load_batch_summary(batch_id)
    if not summary:
        st.error(f"Batch summary not found for {batch_id}")
        return

    st.header(f"Batch: {batch_id}")

    totals = summary.get("totals", {})
    ev = summary.get("evaluation_summary", {})

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sessions", totals.get("total_sessions", 0))
    c2.metric("Completed", totals.get("completed", 0))
    c3.metric("Failed", totals.get("failed", 0))
    c4.metric("Evaluated", totals.get("evaluated", 0))
    c5.metric("Pass Rate", f"{ev.get('pass_rate', 0):.1%}")

    # Charts
    _render_binary_charts(ev)

    # Sessions table — with search, filters, clickable rows, pagination
    st.subheader("Sessions")
    sessions = summary.get("sessions", [])

    # Search + filters
    search = st.text_input("Search sessions", placeholder="Type to filter by session ID or config name...",
                           key="session_search")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        status_filter = st.selectbox("Status", ["All", "completed", "failed"], index=0)
    with fc2:
        rating_opts = ["All", "PASS", "FAIL"]
        rating_filter = st.selectbox("Rating", rating_opts, index=0)
    with fc3:
        only_failures = st.toggle("Show only failures", value=False)

    filtered = sessions
    if search:
        q = search.lower()
        filtered = [s for s in filtered
                    if q in s.get("session_id", "").lower()
                    or q in s.get("config_name", "").lower()]
    if status_filter != "All":
        filtered = [s for s in filtered if s.get("status") == status_filter]
    if rating_filter != "All":
        filtered = [s for s in filtered if s.get("evaluation", {}).get("overall_rating") == rating_filter]
    if only_failures:
        filtered = [s for s in filtered if not s.get("evaluation", {}).get("pass_fail", True)]

    rows = []
    for s in filtered:
        ev_data = s.get("evaluation", {})
        rows.append({
            "Session ID": s.get("session_id", "unknown"),
            "Config": s.get("config_name", "unknown"),
            "Status": s.get("status", "unknown"),
            "Rating": ev_data.get("overall_rating", s.get("error", "N/A")),
            "Pass": ev_data.get("pass_fail", None),
        })

    if rows:
        # Pagination
        total = len(rows)
        total_pages = max(1, (total + SESSIONS_PAGE_SIZE - 1) // SESSIONS_PAGE_SIZE)
        if total_pages > 1:
            page_num = st.number_input(
                f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=1,
                key="session_page",
            )
        else:
            page_num = 1
        start = (page_num - 1) * SESSIONS_PAGE_SIZE
        page_rows = rows[start : start + SESSIONS_PAGE_SIZE]

        st.caption(f"Showing {start + 1}–{min(start + SESSIONS_PAGE_SIZE, total)} of {total} sessions. Click a row to view details.")
        df_sessions = pd.DataFrame(page_rows)
        event = st.dataframe(
            df_sessions, hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row",
            key="session_table",
        )
        sel_rows = event.selection.rows if event.selection else []
        if sel_rows:
            selected_sid = page_rows[sel_rows[0]]["Session ID"]
            if selected_sid != "unknown":
                navigate("session_detail", session_id=selected_sid, batch_id=batch_id)
    else:
        st.info("No sessions match the current filters.")

    # Export buttons
    if rows:
        st.subheader("Export")
        ec1, ec2 = st.columns(2)
        with ec1:
            csv_data = pd.DataFrame(rows).to_csv(index=False)
            st.download_button("Download CSV", csv_data, f"{batch_id}_sessions.csv", "text/csv")
        with ec2:
            json_data = json.dumps(rows, indent=2)
            st.download_button("Download JSON", json_data, f"{batch_id}_sessions.json", "application/json")




def _render_binary_charts(ev: dict):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Per-Metric Pass Rates")
        mpr = ev.get("metric_pass_rates", {})
        if mpr:
            data = []
            for metric, info in mpr.items():
                data.append({
                    "Metric": metric,
                    "Pass Rate": info.get("rate", 0),
                    "Passed": info.get("passed", 0),
                    "Total": info.get("total", 0),
                })
            df = pd.DataFrame(data)
            fig = px.bar(
                df, y="Metric", x="Pass Rate", orientation="h",
                hover_data=["Passed", "Total"],
                color="Pass Rate",
                color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                range_color=[0, 1],
            )
            fig.update_xaxes(range=[0, 1.05], tickformat=".0%")
            fig.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Pass / Fail")
        pass_count = ev.get("pass_count", 0)
        fail_count = ev.get("fail_count", 0)
        fig = go.Figure(data=[go.Pie(
            labels=["Pass", "Fail"],
            values=[pass_count, fail_count],
            marker_colors=[RATING_COLORS["PASS"], RATING_COLORS["FAIL"]],
            hole=0.4,
        )])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Session Detail
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Page: Sessions Browser
# ---------------------------------------------------------------------------

def page_sessions_browser():
    st.header("Sessions")
    st.caption("Browse all individual sessions. Click a row to view details.")

    all_sessions = load_all_sessions()
    if not all_sessions:
        st.warning("No sessions found in results/sessions/.")
        return

    # Filters
    col_search, col_test, col_rating = st.columns([2, 1, 1])

    with col_search:
        search = st.text_input("Search sessions", placeholder="Filter by session ID...")

    test_names = sorted({s["test_name"] for s in all_sessions if s["test_name"]})
    with col_test:
        selected_test = st.selectbox("Test name", ["All"] + test_names)

    with col_rating:
        selected_rating = st.selectbox("Rating", ["All", "PASS", "FAIL", "Not evaluated"])

    # Apply filters
    filtered = all_sessions
    if search:
        q = search.lower()
        filtered = [s for s in filtered if q in s["session_id"].lower()]
    if selected_test != "All":
        filtered = [s for s in filtered if s["test_name"] == selected_test]
    if selected_rating == "PASS":
        filtered = [s for s in filtered if s["eval_rating"] == "PASS"]
    elif selected_rating == "FAIL":
        filtered = [s for s in filtered if s["eval_rating"] == "FAIL"]
    elif selected_rating == "Not evaluated":
        filtered = [s for s in filtered if not s["has_eval"]]

    # KPIs
    total = len(filtered)
    evaluated = sum(1 for s in filtered if s["has_eval"])
    passed = sum(1 for s in filtered if s["eval_rating"] == "PASS")
    failed = sum(1 for s in filtered if s["eval_rating"] == "FAIL")
    pass_rate = passed / evaluated if evaluated else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", total)
    c2.metric("Evaluated", evaluated)
    c3.metric("Passed", passed)
    c4.metric("Failed", failed)
    c5.metric("Pass Rate", f"{pass_rate:.0%}" if evaluated else "N/A")

    # Pagination
    page_size = 50
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page_num = st.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=1)
    start = (page_num - 1) * page_size
    page_items = filtered[start:start + page_size]

    # Table
    rows = []
    for s in page_items:
        rows.append({
            "Session ID": s["session_id"],
            "Test Name": s["test_name"],
            "Timestamp": format_timestamp(s["timestamp"]) if s["timestamp"] else "",
            "Turns": s["total_turns"],
            "Rating": s["eval_rating"] if s["has_eval"] else "—",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.caption(f"Showing {start + 1}–{min(start + page_size, len(filtered))} of {len(filtered)} sessions.")
        event = st.dataframe(
            df, hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row",
            key="sessions_browser_table",
        )
        sel = event.selection.rows if event.selection else []
        if sel:
            sid = rows[sel[0]]["Session ID"]
            navigate("session_detail", session_id=sid)
    else:
        st.info("No sessions match the current filters.")


# ---------------------------------------------------------------------------
# Page: Session Detail
# ---------------------------------------------------------------------------

def page_session_detail():
    session_id = st.query_params.get("session_id", "")
    batch_id = st.query_params.get("batch_id", "")
    if not session_id:
        st.warning("No session selected.")
        return

    st.header(f"Session: {session_id}")
    if batch_id:
        if st.button(f"← Back to batch {batch_id}"):
            navigate("batch_detail", batch_id=batch_id)
    else:
        if st.button("← Back to Sessions"):
            navigate("sessions_browser")

    readme = load_session_readme(session_id)
    evaluation = load_session_evaluation(session_id)
    interaction = load_interaction_log(session_id)
    transcript = load_chat_transcript(session_id)

    # Stats row
    stats = {}
    if readme:
        stats = readme.get("statistics", {})
    elif interaction:
        stats = interaction.get("summary", {})

    if stats:
        cols = st.columns(7)
        cols[0].metric("Turns", stats.get("total_turns", 0))
        cols[1].metric("Tool Calls", stats.get("total_tool_calls", 0))
        cols[2].metric("Tool Usage Rate", f"{stats.get('tool_usage_rate', 0):.0%}")
        cols[3].metric("Avg User Msg Len", f"{stats.get('avg_user_message_length', 0):.0f}")
        cols[4].metric("Avg Sonic Resp Len", f"{stats.get('avg_sonic_response_length', 0):.0f}")
        cols[5].metric("Errors", stats.get("total_errors", 0))
        cols[6].metric("Session Continuations", stats.get("total_session_continuations", stats.get("session_continuations", 0)))

    # Evaluation
    st.subheader("Evaluation")
    if evaluation:
        _render_session_evaluation(evaluation)
    else:
        st.info("No evaluation data available for this session.")

    # Conversation
    st.subheader("Conversation")
    tab_chat, tab_raw = st.tabs(["Chat View", "Raw Transcript"])

    with tab_chat:
        if interaction:
            turns = interaction.get("turns", [])
            session_events = interaction.get("session_events", [])
            # Build a map of continuation events by after_turn_number
            continuation_map: dict[int, list] = {}
            for evt in session_events:
                tn = evt.get("after_turn_number", 0)
                continuation_map.setdefault(tn, []).append(evt)

            # Get expected responses from config for per-turn comparison
            config = interaction.get("configuration", {})
            expected_responses = config.get("_expected_responses", [])
            expected_tool_calls = config.get("_expected_tool_calls", [])

            for turn in turns:
                turn_num = turn.get("turn_number", 0)

                with st.chat_message("user"):
                    st.write(turn.get("user_message", ""))

                with st.chat_message("assistant"):
                    st.write(turn.get("sonic_response", ""))

                # Audio playback
                audio_file = _dirs()[2] / session_id / "audio" / f"turn_{turn_num}.wav"
                if audio_file.exists():
                    st.audio(str(audio_file), format="audio/wav")

                # Tool calls
                tool_calls = turn.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        with st.expander(f"Tool: {tc.get('tool_name', 'unknown')}"):
                            st.json({
                                "input": tc.get("tool_input", {}),
                                "result": tc.get("tool_result", {}),
                            })

                # Per-turn expected vs actual (dataset-driven runs)
                if expected_responses and turn_num <= len(expected_responses):
                    expected = expected_responses[turn_num - 1]
                    if expected:
                        with st.expander("Expected vs Actual"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Expected:**")
                                st.text(expected)
                            with c2:
                                st.markdown("**Actual:**")
                                st.text(turn.get("sonic_response", ""))

                # Session continuation events after this turn
                for evt in continuation_map.get(turn_num, []):
                    st.info(
                        f"🔄 Session continued after turn {turn_num} "
                        f"(session {evt.get('old_session_number', '?')} → {evt.get('new_session_number', '?')}, "
                        f"elapsed: {evt.get('old_session_duration_seconds', 0):.0f}s, "
                        f"replayed {evt.get('messages_replayed', 0)} messages)"
                    )
        else:
            st.info("No interaction log available.")

    # Full conversation audio
    if interaction:
        conv_wav = _dirs()[2] / session_id / "audio" / "conversation.wav"
        if conv_wav.exists():
            with st.expander("Full Conversation Audio"):
                st.audio(str(conv_wav), format="audio/wav")

    with tab_raw:
        if transcript:
            st.code(transcript, language=None)
        else:
            st.info("No transcript file available.")

    # Config
    if interaction:
        config = interaction.get("configuration", {})
        if config:
            with st.expander("Configuration"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**System Prompt:**")
                    st.text(config.get("sonic_system_prompt", "N/A"))
                    st.markdown(f"**Sonic Model:** `{config.get('sonic_model_id', 'N/A')}`")
                    st.markdown(f"**User Model:** `{config.get('user_model_id', 'N/A')}`")
                with c2:
                    tools = (config.get("sonic_tool_config") or {}).get("tools", [])
                    if tools:
                        st.markdown("**Tools:**")
                        for t in tools:
                            spec = t.get("toolSpec", {})
                            st.markdown(f"- `{spec.get('name', 'unknown')}`: {spec.get('description', '')[:80]}")
                    eval_criteria = config.get("evaluation_criteria", {})
                    if eval_criteria:
                        st.markdown("**Evaluation Criteria:**")
                        st.json(eval_criteria)


def _render_session_evaluation(evaluation: dict):
    results = evaluation.get("results", {})
    _render_binary_session_eval(results)




def _render_binary_session_eval(results: dict):
    overall = results.get("overall_rating", "N/A")
    pass_rate = results.get("pass_rate", 0)
    st.markdown(
        f"**Verdict:** {rating_badge(overall)} &nbsp; **Pass Rate:** {pass_rate:.0%}",
        unsafe_allow_html=True,
    )

    # Per-metric verdicts
    metric_verdicts = results.get("metric_verdicts", {})
    if metric_verdicts:
        st.markdown("**Metric Verdicts:**")
        for metric, info in metric_verdicts.items():
            verdict = info.get("verdict", "N/A")
            reasoning = info.get("reasoning", "")
            icon = "+" if verdict == "PASS" else "-"
            with st.expander(f"{metric}: {verdict}", icon=f":material/{'check_circle' if verdict == 'PASS' else 'cancel'}:"):
                st.write(reasoning)
                rubric_verdicts = info.get("rubric_verdicts", [])
                if rubric_verdicts:
                    st.markdown("**Rubric Questions:**")
                    for rv in rubric_verdicts:
                        q = rv.get("question", "")
                        v = rv.get("verdict", "")
                        r = rv.get("reasoning", "")
                        color = "green" if v == "YES" else "red"
                        st.markdown(f"- :{color}[**{v}**] {q}")
                        if r:
                            st.caption(r)
    else:
        # Fall back to aspect_ratings
        aspect_ratings = results.get("aspect_ratings", {})
        if aspect_ratings:
            cols = st.columns(len(aspect_ratings))
            for i, (aspect, rating) in enumerate(aspect_ratings.items()):
                with cols[i]:
                    st.markdown(f"**{aspect}**")
                    st.markdown(rating_badge(rating), unsafe_allow_html=True)

    # Strengths & weaknesses
    col1, col2 = st.columns(2)
    with col1:
        strengths = results.get("strengths", [])
        if strengths:
            st.markdown("**Strengths:**")
            for s in strengths:
                st.markdown(f"- {s}")
    with col2:
        weaknesses = results.get("weaknesses", [])
        if weaknesses:
            st.markdown("**Weaknesses:**")
            for w in weaknesses:
                st.markdown(f"- {w}")


# ---------------------------------------------------------------------------
# Shared: clickable session list
# ---------------------------------------------------------------------------

def _render_clickable_session_list(rows: list[dict], batch_id: str, key_prefix: str):
    """Render a small dataframe of sessions with click-to-navigate."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    event = st.dataframe(
        df, hide_index=True, use_container_width=True,
        on_select="rerun", selection_mode="single-row",
        key=f"{key_prefix}_session_list",
    )
    sel = event.selection.rows if event.selection else []
    if sel:
        sid = rows[sel[0]]["Session ID"]
        if sid != "unknown":
            navigate("session_detail", session_id=sid, batch_id=batch_id)


# ---------------------------------------------------------------------------
# Page: Error Analysis
# ---------------------------------------------------------------------------

def page_error_analysis():
    st.header("Error Analysis")

    batches = load_batch_index()
    if not batches:
        st.warning("No batches found.")
        return

    batches_sorted = sorted(batches, key=lambda x: x.get("timestamp", ""), reverse=True)
    batch_options = {batch_label(b): b["batch_id"] for b in batches_sorted}
    selected_label = st.selectbox("Select Batch", list(batch_options.keys()))
    if not selected_label:
        return
    selected_batch = batch_options[selected_label]

    summary = load_batch_summary(selected_batch)
    if not summary:
        st.error("Batch summary not found.")
        return

    sessions = summary.get("sessions", [])
    totals = summary.get("totals", {})

    exec_failures = [s for s in sessions if s.get("status") == "failed"]
    eval_failures = [s for s in sessions
                     if s.get("status") == "completed"
                     and not s.get("evaluation", {}).get("pass_fail", True)]
    total_failures = len(exec_failures) + len(eval_failures)
    total_count = totals.get("total_sessions", len(sessions))
    failure_rate = total_failures / total_count if total_count else 0

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Failures", total_failures)
    c2.metric("Failure Rate", f"{failure_rate:.1%}")
    c3.metric("Execution Failures", len(exec_failures))
    c4.metric("Evaluation Failures", len(eval_failures))

    tab_exec, tab_eval = st.tabs(["Execution Failures", "Evaluation Failures"])

    with tab_exec:
        if exec_failures:
            # Group by error message
            error_groups: dict[str, list] = {}
            for s in exec_failures:
                err = s.get("error", "Unknown error")
                error_groups.setdefault(err, []).append(s)

            for err_msg, sessions_in_group in sorted(error_groups.items(), key=lambda x: -len(x[1])):
                st.markdown(f"**{err_msg}** ({len(sessions_in_group)} sessions)")
                with st.expander("Show sessions"):
                    for s in sessions_in_group[:20]:
                        st.text(f"  {s.get('session_id', 'unknown')} ({s.get('config_name', 'unknown')})")
        else:
            st.success("No execution failures in this batch.")

    with tab_eval:
        if eval_failures:
            _render_binary_error_analysis(eval_failures)

            # Failure by config
            st.subheader("Failure by Config")
            config_stats: dict[str, dict] = {}
            for s in sessions:
                cfg = s.get("config_name", "unknown")
                if cfg not in config_stats:
                    config_stats[cfg] = {"total": 0, "failures": 0, "weaknesses": []}
                config_stats[cfg]["total"] += 1
                if not s.get("evaluation", {}).get("pass_fail", True):
                    config_stats[cfg]["failures"] += 1
                    config_stats[cfg]["weaknesses"].extend(
                        s.get("evaluation", {}).get("weaknesses", [])
                    )

            rows = []
            for cfg, info in sorted(config_stats.items(), key=lambda x: -x[1]["failures"]):
                if info["failures"] == 0:
                    continue
                top_weakness = info["weaknesses"][0][:80] + "..." if info["weaknesses"] else "N/A"
                rows.append({
                    "Config": cfg,
                    "Failures": info["failures"],
                    "Total": info["total"],
                    "Failure Rate": info["failures"] / info["total"] if info["total"] else 0,
                    "Top Weakness": top_weakness,
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Drill-down: filter by failing metric → clickable session list
            st.subheader("Drill Down")
            metrics = set()
            for s in eval_failures:
                for m, v in s.get("evaluation", {}).get("aspect_ratings", {}).items():
                    if v == "FAIL":
                        metrics.add(m)
            if metrics:
                selected_metric = st.selectbox("Filter by failing metric", sorted(metrics))
                matching = [
                    s for s in eval_failures
                    if s.get("evaluation", {}).get("aspect_ratings", {}).get(selected_metric) == "FAIL"
                ]
                st.write(f"**{len(matching)} sessions** failed on **{selected_metric}**:")
                drill_rows = [{"Session ID": s.get("session_id", "unknown"),
                               "Config": s.get("config_name", "")} for s in matching[:50]]
                _render_clickable_session_list(drill_rows, selected_batch, "drill_binary")
        else:
            st.success("No evaluation failures in this batch.")




def _render_binary_error_analysis(eval_failures: list[dict]):
    st.subheader("Per-Metric Failure Rates")
    metric_fail_count: Counter = Counter()
    metric_total_count: Counter = Counter()
    metric_reasons: dict[str, list] = {}

    for s in eval_failures:
        ev = s.get("evaluation", {})
        for metric, rating in ev.get("aspect_ratings", {}).items():
            metric_total_count[metric] += 1
            if rating == "FAIL":
                metric_fail_count[metric] += 1
                # Collect reasoning from weaknesses
                for w in ev.get("weaknesses", []):
                    if w.startswith(metric):
                        metric_reasons.setdefault(metric, []).append(
                            (s.get("session_id", "unknown"), w)
                        )

    if metric_fail_count:
        data = []
        for metric in sorted(metric_fail_count.keys(), key=lambda m: -metric_fail_count[m]):
            data.append({
                "Metric": metric,
                "Failures": metric_fail_count[metric],
                "Total Failing Sessions": metric_total_count[metric],
                "Failure Rate": metric_fail_count[metric] / metric_total_count[metric]
                if metric_total_count[metric] else 0,
            })
        df = pd.DataFrame(data)
        fig = px.bar(df, y="Metric", x="Failures", orientation="h",
                     color="Failure Rate",
                     color_continuous_scale=["#f39c12", "#e74c3c"])
        fig.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Show reasons per metric
        for metric, reasons in metric_reasons.items():
            with st.expander(f"{metric} — {len(reasons)} failures"):
                for sid, reason in reasons[:20]:
                    st.markdown(f"- **{sid}**: {reason}")

    # Co-failure correlation
    st.subheader("Co-Failure Correlation")
    if len(eval_failures) > 1:
        fail_combos: Counter = Counter()
        for s in eval_failures:
            failing_metrics = sorted(
                m for m, r in s.get("evaluation", {}).get("aspect_ratings", {}).items()
                if r == "FAIL"
            )
            if len(failing_metrics) > 1:
                fail_combos[tuple(failing_metrics)] += 1

        if fail_combos:
            rows = [{"Metrics": " + ".join(combo), "Count": count}
                    for combo, count in fail_combos.most_common(10)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No co-failures detected (each failure involves only one metric).")


# ---------------------------------------------------------------------------
# Page: Batch Comparison
# ---------------------------------------------------------------------------

def page_batch_comparison():
    st.header("Batch Comparison")

    batches = load_batch_index()
    if not batches:
        st.warning("No batches found.")
        return

    batches_sorted = sorted(batches, key=lambda b: b.get("timestamp", ""), reverse=True)
    label_to_id = {batch_label(b): b["batch_id"] for b in batches_sorted}
    labels = list(label_to_id.keys())

    # Default to last 2-3
    default_count = min(3, len(labels))
    selected_labels = st.multiselect("Select batches to compare", labels,
                                     default=labels[:default_count])
    selected = [label_to_id[l] for l in selected_labels]

    if len(selected) < 2:
        st.info("Select at least 2 batches to compare.")
        return

    summaries = {}
    for bid in selected:
        s = load_batch_summary(bid)
        if s:
            summaries[bid] = s

    if len(summaries) < 2:
        st.warning("Could not load enough batch summaries.")
        return

    # Pass rate comparison
    st.subheader("Pass Rate Comparison")
    pr_data = []
    for bid in selected:
        if bid in summaries:
            ev = summaries[bid].get("evaluation_summary", {})
            pr_data.append({
                "Batch": bid,
                "Pass Rate": ev.get("pass_rate", 0),
                "Sessions": summaries[bid].get("totals", {}).get("total_sessions", 0),
            })
    df_pr = pd.DataFrame(pr_data)
    fig = px.bar(df_pr, x="Batch", y="Pass Rate", color="Batch",
                 hover_data=["Sessions"], text_auto=".1%")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Delta analysis
    st.subheader("Delta Analysis")
    if len(selected) >= 2:
        # Use last two in chronological order
        ordered = sorted(selected, key=lambda b: next(
            (x.get("timestamp", "") for x in batches if x["batch_id"] == b), ""
        ))
        old_bid, new_bid = ordered[-2], ordered[-1]
        old_ev = summaries[old_bid].get("evaluation_summary", {})
        new_ev = summaries[new_bid].get("evaluation_summary", {})

        st.caption(f"Comparing **{old_bid}** → **{new_bid}**")
        old_pr = old_ev.get("pass_rate", 0)
        new_pr = new_ev.get("pass_rate", 0)
        st.metric("Pass Rate", f"{new_pr:.1%}", delta=f"{new_pr - old_pr:+.1%}")

    # Per-Metric Pass Rate Comparison
    if len(selected) >= 2:
        st.subheader("Per-Metric Pass Rate Comparison")
        mpr_data = []
        all_metrics = set()
        for bid in selected:
            mpr = summaries[bid].get("evaluation_summary", {}).get("metric_pass_rates", {})
            for metric, info in mpr.items():
                all_metrics.add(metric)
                mpr_data.append({
                    "Batch": bid,
                    "Metric": metric,
                    "Pass Rate": info.get("rate", 0),
                })
        if mpr_data:
            df_mpr = pd.DataFrame(mpr_data)
            fig = px.bar(
                df_mpr, x="Metric", y="Pass Rate", color="Batch",
                barmode="group", text_auto=".1%",
            )
            fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Per-metric deltas between last two batches
            if len(selected) >= 2:
                ordered_bin = sorted(selected, key=lambda b: next(
                    (x.get("timestamp", "") for x in batches if x["batch_id"] == b), ""
                ))
                old_b, new_b = ordered_bin[-2], ordered_bin[-1]
                old_mpr = summaries[old_b].get("evaluation_summary", {}).get("metric_pass_rates", {})
                new_mpr = summaries[new_b].get("evaluation_summary", {}).get("metric_pass_rates", {})
                cols = st.columns(len(all_metrics))
                for i, metric in enumerate(sorted(all_metrics)):
                    old_rate = old_mpr.get(metric, {}).get("rate", 0)
                    new_rate = new_mpr.get(metric, {}).get("rate", 0)
                    cols[i].metric(metric, f"{new_rate:.1%}", delta=f"{new_rate - old_rate:+.1%}")


# ---------------------------------------------------------------------------
# Page: Conversation Search
# ---------------------------------------------------------------------------

def page_search():
    st.header("Conversation Search")
    st.caption("Search across all sessions for text in user messages or assistant responses.")

    query = st.text_input("Search query", placeholder="Type to search conversations...")
    if not query or len(query) < 2:
        st.info("Enter at least 2 characters to search.")
        return

    q = query.lower()
    results = []

    # Scan all session directories
    sessions_dir = _dirs()[2]
    if sessions_dir.exists():
        for session_dir in sorted(sessions_dir.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue
            log_path = session_dir / "logs" / "interaction_log.json"
            if not log_path.exists():
                continue
            try:
                with open(log_path) as f:
                    log = json.load(f)
                for turn in log.get("turns", []):
                    user_msg = turn.get("user_message", "")
                    sonic_resp = turn.get("sonic_response", "")
                    matched_in = []
                    if q in user_msg.lower():
                        matched_in.append("user")
                    if q in sonic_resp.lower():
                        matched_in.append("assistant")
                    if matched_in:
                        results.append({
                            "Session ID": session_dir.name,
                            "Turn": turn.get("turn_number", 0),
                            "Matched In": ", ".join(matched_in),
                            "User Message": user_msg[:120],
                            "Assistant Response": sonic_resp[:120],
                        })
            except (json.JSONDecodeError, KeyError):
                continue

            # Cap results for performance
            if len(results) >= 200:
                break

    st.write(f"Found {len(results)} matching turns.")
    if results:
        df = pd.DataFrame(results)
        event = st.dataframe(
            df, hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row",
            key="search_results",
        )
        sel = event.selection.rows if event.selection else []
        if sel:
            sid = results[sel[0]]["Session ID"]
            navigate("session_detail", session_id=sid)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Nova Sonic Evaluation Dashboard",
        page_icon=":material/analytics:",
        layout="wide",
    )

    st.title("Nova Sonic Evaluation Dashboard")

    # Results directory configuration
    qp_results_dir = st.query_params.get("results_dir", "")
    default_dir = qp_results_dir or str(RESULTS_DIR)
    with st.sidebar.expander("Settings", icon=":material/settings:"):
        results_input = st.text_input("Results directory", value=default_dir, key="results_dir",
                                      help="Path to the results folder (default: results)")
        if results_input != default_dir:
            st.query_params["results_dir"] = results_input

    # Deep linking via query params
    pages = {
        "batch_overview": "Batch Overview",
        "batch_detail": "Batch Detail",
        "sessions_browser": "Sessions",
        "session_detail": "Session Detail",
        "error_analysis": "Error Analysis",
        "batch_comparison": "Batch Comparison",
        "search": "Search",
    }
    page_keys = list(pages.keys())
    page_labels = list(pages.values())

    qp_page = st.query_params.get("page", "batch_overview")
    if qp_page not in page_keys:
        qp_page = "batch_overview"

    default_idx = page_keys.index(qp_page)

    selected_label = st.sidebar.radio("Navigation", page_labels, index=default_idx)
    selected_page = page_keys[page_labels.index(selected_label)]

    # Sync query param if sidebar changed the page
    if selected_page != qp_page:
        st.query_params["page"] = selected_page
        # Clear detail params when navigating away
        if selected_page == "batch_overview":
            for k in ["batch_id", "session_id"]:
                st.query_params.pop(k, None)

    dispatch = {
        "batch_overview": page_batch_overview,
        "batch_detail": page_batch_detail,
        "sessions_browser": page_sessions_browser,
        "session_detail": page_session_detail,
        "error_analysis": page_error_analysis,
        "batch_comparison": page_batch_comparison,
        "search": page_search,
    }

    dispatch[selected_page]()


if __name__ == "__main__":
    main()
