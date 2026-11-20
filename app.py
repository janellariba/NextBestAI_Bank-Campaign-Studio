"""Run: python dashboard_draft/app.py --mode live   (or --mode demo for simulated LLM responses)."""
import argparse
import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from flask import jsonify, session

import swimlane
from backend import CampaignService

TIMEZONE = ZoneInfo(os.environ.get("NEXTBEST_TIMEZONE", "Asia/Manila"))
POLL_MS = int(os.environ.get("NEXTBEST_POLL_MS", "1500"))
STAGES = ("Eligibility", "Scoring", "Drafting", "Compliance", "Review", "Sent", "Client reply")
STAGE_OF = {"Eligibility Agent": 0, "Propensity Agent": 1, "Email Agent": 2, "Compliance Agent": 3, "Human Reviewer": 4,
            "Release Agent": 5, "Reject Agent": 5, "Record Agent": 5}

# The team's Final Architecture diagram, mapped to what implements each box. Status words match the
# project spreadsheet: Implemented, or Synthetic / Placeholder where the data is not real yet.
ARCHITECTURE = [
    ("Mission Application · Business Operation App", [("Campaign studio (this app)", "app.py", "Implemented")]),
    ("Team Layer", [("Marketing team + AI agent", "review panel, CampaignService", "Implemented")]),
    ("Agentic AI Orchestration Layer", [("Ollama", "qwen3.5:4b", "Implemented"), ("LangGraph", "StateGraph, interrupt, Supervisor", "Implemented"),
                                        ("LangChain", "@tool, text splitter", "Implemented"), ("Python", "notebook + backend", "Implemented")]),
    ("Agentic AI Layer", [("Eligibility Agent", "eligibility_agent", "Implemented"), ("Scoring Agent", "propensity_agent", "Implemented"),
                          ("Email Agent", "email_agent", "Implemented"), ("Compliance Agent", "compliance_agent", "Implemented")]),
    ("Data Layer", [("Marketing Guidelines", "product_kb.db page 4", "Placeholder"), ("Customer Transaction Data", "bank_demo.db", "Synthetic"),
                    ("Bank policies", "product_kb.db pages 1 to 3", "Placeholder")]),
]
RESPONSIBLE_AI = [("Model evaluation", "holdout AUC, recall@k, Evaluation lab"), ("Model explainability", "top drivers + why summary"),
                  ("RAG-grounded email", "BM25 + SBERT, Evidence tab"), ("Human approval of email", "interrupt(), release gate"),
                  ("Audit trail", "campaign_audit, archived on reset")]
ADVERSARIAL_AI = [("Age restriction guardrail", "18 to 60, re-checked at release"), ("LLM as a judge", "judge_email, fails closed"),
                  ("Rule-based compliance", "privacy, rates, guarantees, tone"), ("User Prompt gate", "blocks injection, admin override"),
                  ("Red-team tests", "notebook cell 35, Evaluation lab")]


def architecture_view():
    tone = {"Implemented": "ok", "Synthetic": "warn", "Placeholder": "warn"}
    side = lambda title, items: html.Div([html.H3(title), html.Ul([html.Li([html.Strong(n), html.Div(w, className="muted small")]) for n, w in items])],
                                         className="arch-side")
    layers = [html.Div([html.Div(name, className="arch-name"), html.Div([html.Div([
        html.Strong(item), html.Div(where, className="muted small"), html.Span(status, className="pill " + tone[status])],
        className="arch-item") for item, where, status in items], className="arch-items")], className="arch-layer") for name, items in ARCHITECTURE]
    return html.Div([side("Responsible AI", RESPONSIBLE_AI), html.Div(layers, className="arch-stack"), side("Adversarial AI", ADVERSARIAL_AI)],
                    className="arch")


def table(records, **kwargs):
    return dash_table.DataTable(
        data=records, columns=[{"name": k.replace("_", " ").title(), "id": k} for k in (records[0] if records else {}) if k != "id"],
        page_size=10, sort_action="native", style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "inherit", "fontSize": 13, "textAlign": "left", "padding": "12px", "maxWidth": 260, "whiteSpace": "normal"},
        style_header={"backgroundColor": "#eff3f8", "fontWeight": "bold", "border": "none"},
        style_data={"borderBottom": "1px solid #edf0f6", "borderTop": "none"}, **kwargs)


def safe_records(df):
    if df is None:
        return []
    return json.loads(df.to_json(orient="records"))


def signature(*parts):
    """Stable hash used to skip re-rendering unchanged panels (keeps <details> and tabs open between polls)."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def review_feedback(campaign, state):
    """Describe the current gate/outcome, rather than an earlier button submission."""
    if campaign.get("busy"):
        return "Review actions are temporarily unavailable while " + campaign.get("job", "the campaign is processing").lower() + "."
    if state.get("error"):
        return "Workflow error for this client: " + state["error"]
    if campaign.get("error"):
        return "Workflow error: " + campaign["error"]
    if state.get("final_email") and state.get("approval_status") == "approve":
        if state.get("client_response"):
            return "Approved and sent. Client reply logged: " + state["client_response"]["next_step"].lower() + "."
        return "Approved. The email was released (simulated send) and recorded. Log the client's reply when it arrives."
    if state.get("approval_status") == "reject":
        return "Rejected. This workflow is complete. Start a new campaign to prepare another email for this customer."
    if state.get("status", "").startswith("BLOCKED"):
        return "Approval unavailable: " + state.get("recommendation_reason", state["status"]) + "."
    if not state.get("pending_review"):
        return "Approval becomes available when drafting and compliance checking are complete."
    compliance = state.get("compliance") or {}
    if not compliance.get("passed"):
        return "Approval unavailable: compliance failed. Revise or reject this draft. " + "; ".join(compliance.get("issues", []))
    return "Ready for your decision. Approve, reject, or request another revision."


def local_time(value):
    return datetime.fromisoformat(value).astimezone(TIMEZONE).strftime("%b %d, %H:%M:%S")


def client_status(state):
    """(label, tone) for a status pill. Tone maps to CSS: ok, warn, bad, info, idle."""
    status = state.get("status", "")
    if state.get("error") or status.startswith("ERROR"):
        return "Error", "bad"
    if state.get("pending_review"):
        return ("Awaiting review", "warn") if (state.get("compliance") or {}).get("passed") else ("Needs revision", "bad")
    response = state.get("client_response")
    if response:
        return ("Interested · to approval", "ok") if response["interested"] else ("Not interested", "idle")
    if status.startswith("COMPLETED"):
        return "Sent · awaiting reply", "info"
    if status.startswith("REJECTED"):
        return "Rejected", "idle"
    if status.startswith("BLOCKED"):
        return "Blocked · " + status.split(":", 1)[-1].strip().title(), "bad"
    if not state.get("trace"):
        return "Queued", "idle"
    return "Working", "info"


def stage_dots(state):
    trace = state.get("trace", [])
    stopped = str(state.get("status", "")).startswith(("BLOCKED", "ERROR")) or state.get("error")
    # A blocked client never reached release: mark the stage where it actually stopped, not the audit write.
    reached = max((STAGE_OF[s] for s in trace if s in STAGE_OF and not (stopped and STAGE_OF[s] == 5)), default=-1)
    if stopped and reached < 0:
        reached = 0
    if state.get("pending_review"):
        reached = 4
    if state.get("client_response"):
        reached = 6
    label, tone = client_status(state)
    dots = []
    for i, name in enumerate(STAGES):
        cls = "dot done" if i < reached or (i == reached and tone in ("ok", "idle")) else "dot"
        if i == reached and tone not in ("ok", "idle"):
            cls = "dot " + ("stop" if tone == "bad" else "now")
        dots.append(html.Span(className=cls, title=name))
    return html.Div(dots, className="stage-dots", title=" → ".join(STAGES))


def score_bars(scores, selected, products):
    rows = []
    for p in products:
        value = scores.get(p)
        rows.append(html.Div([
            html.Span(p.title(), className="bar-label"),
            html.Div(html.Div(style={"width": f"{100 * value:.0f}%" if isinstance(value, (int, float)) else "0%"},
                              className="bar-fill" + (" chosen" if p == selected else "")), className="bar-track"),
            html.Span("—" if value is None else f"{value:.2f}", className="bar-value")], className="bar-row"))
    return html.Div(rows, className="score-bars")


def pipeline_table(campaign, profiles, products):
    head = html.Tr([html.Th(h) for h in ("Client", "Progress", "Product scores", "Status", "")])
    body = []
    for client_id in campaign["ids"]:
        s = campaign["states"].get(client_id, {})
        p = profiles.get(client_id, {})
        label, tone = client_status(s)
        body.append(html.Tr([
            html.Td([html.Strong(client_id), html.Div(f"Age {p.get('age', '—')} · ADB {p.get('adb', 0):,.0f}", className="muted small")]),
            html.Td(stage_dots(s)),
            html.Td(score_bars((s.get("propensity") or {}).get("scores", {}), s.get("product"), products)),
            html.Td(html.Span(label, className="pill " + tone)),
            html.Td(html.Button("Review", id={"type": "pick-client", "index": client_id}, className="secondary small-btn",
                                disabled=not s.get("trace")))]))
    return html.Table([html.Thead(head), html.Tbody(body)], className="pipeline")


# ---------------------------------------------------------------------- client journey
# The Business Flow Chart, step by step, with the lane each step belongs to.
JOURNEY = [("Requested", "mkt"), ("Eligibility", "ds"), ("Scoring", "ds"), ("Draft", "sales"), ("Compliance", "sales"),
           ("Your review", "mkt"), ("Sent", "sales"), ("Client reply", "sales"), ("Application", "appr")]
LANES = {"mkt": "Marketing", "ds": "DS Team", "sales": "Sales Team", "appr": "Approval"}


def journey_steps(state, busy=False):
    """Status of each JOURNEY step: done, now, need, stop, todo or outside. Built only from recorded state."""
    trace, status = state.get("trace", []), str(state.get("status", ""))
    steps = ["todo"] * len(JOURNEY)
    reached = -1
    for agent, index in (("User Prompt", 0), ("Eligibility Agent", 1), ("Propensity Agent", 2), ("Email Agent", 3), ("Compliance Agent", 4)):
        if agent in trace:
            reached = max(reached, index)
    for i in range(reached + 1):
        steps[i] = "done"
    if state.get("error") or status.startswith("ERROR"):
        steps[min(reached + 1, len(steps) - 1)] = "stop"
        return steps
    if status.startswith("BLOCKED"):
        steps[max(reached, 1)] = "stop"
        return steps
    comp = state.get("compliance") or {}
    if state.get("pending_review"):
        steps[4] = "done" if comp.get("passed") else "stop"
        steps[5] = "need"
        return steps
    if state.get("approval_status") == "reject" or status.startswith("REJECTED"):
        steps[5] = "stop"
        return steps
    if state.get("final_email"):
        steps[5] = steps[6] = "done"
        reply = state.get("client_response")
        if not reply:
            steps[7] = "need"
        elif reply["interested"]:
            steps[7], steps[8] = "done", "outside"
        else:
            steps[7] = "stop"
        return steps
    if trace or busy:
        steps[min(reached + 1, len(steps) - 1)] = "now"
    return steps


def next_action(state, busy=False):
    """(short label, full sentence, tone) telling the marketer what happens next for this client."""
    status = str(state.get("status", ""))
    comp = state.get("compliance") or {}
    reply = state.get("client_response")
    if state.get("error") or status.startswith("ERROR"):
        return "Error", "The AI step failed. Check the Ollama connection, then run this client again.", "bad"
    if not state.get("trace"):
        return "Queued", "Waiting for the agents to start on this client.", "info"
    if status.startswith("BLOCKED"):
        return "Blocked", "No email: " + (state.get("recommendation_reason") or "the client is outside the contact policy").rstrip(".").lower() + ".", "bad"
    if state.get("pending_review") and not busy:
        if comp.get("passed"):
            return "Review email", "Read the draft, then approve and send, revise, or reject.", "need"
        return "Revise or reject", "The draft failed compliance. Ask for a revision or reject it.", "bad"
    if reply:
        if reply["interested"]:
            return "To approval", "Client is interested. Hand over to application and approval (outside the prototype).", "ok"
        return "Closed", "Client is not interested. Nothing more to do.", "idle"
    if state.get("final_email"):
        return "Log reply", "Email sent. Log the client's reply when it comes in.", "need"
    if state.get("approval_status") == "reject" or status.startswith("REJECTED"):
        return "Closed", "Email rejected. Nothing was sent.", "idle"
    return "Working", "The agents are working on this client.", "info"


def journey_rail(state, busy=False):
    steps = journey_steps(state, busy)
    short, sentence, tone = next_action(state, busy)
    groups, current = [], None
    for (name, lane), step in zip(JOURNEY, steps):
        if lane != current:
            groups.append((lane, []))
            current = lane
        groups[-1][1].append(html.Li([html.Span(className="j-dot"), html.Span(name, className="j-name")],
                                     className="j-step " + step, title=f"{name}: {step}"))
    return html.Div([
        html.Ol([html.Li([html.Ol(items, className="j-steps"), html.Span(LANES[lane], className="j-lane-name")],
                         className="j-lane lane-" + lane) for lane, items in groups], className="journey", **{"aria-label": "Client journey"}),
        html.P([html.Strong("Next step: "), sentence], className="next-step " + tone, role="status")], className="journey-wrap")


def queue_list(campaign, profiles, products, selected):
    rows = []
    for client_id in campaign["ids"]:
        s = campaign["states"].get(client_id, {})
        p = profiles.get(client_id, {})
        short, _, tone = next_action(s, campaign["busy"] and not s.get("pending_review"))
        steps = journey_steps(s)
        done = sum(x in ("done", "outside") for x in steps)
        product = (s.get("product") or "").title()
        rows.append(html.Button([
            html.Span([html.Strong(client_id), html.Span(f"Age {p.get('age', '?')}" + (f" · {product}" if product else ""), className="q-meta")], className="q-id"),
            html.Span(html.Span(style={"width": f"{100 * done / len(steps):.0f}%"}), className="q-progress", title=f"{done} of {len(steps)} steps"),
            html.Span(short, className="pill " + tone)],
            id={"type": "pick-client", "index": client_id}, className="q-row" + (" selected" if client_id == selected else ""),
            disabled=not s.get("trace"), **{"aria-label": f"Open client {client_id}: {short}"}))
    return rows


def queue_summary(states):
    states = list(states)
    counts = [("need", "need you", sum(next_action(s)[2] == "need" for s in states)),
              ("ok", "interested", sum(bool((s.get("client_response") or {}).get("interested")) for s in states)),
              ("info", "sent", sum(bool(s.get("final_email")) for s in states)),
              ("bad", "blocked or failed", sum(next_action(s)[2] == "bad" for s in states))]
    return [html.Div([html.Strong(str(n)), html.Span(label)], className="qs " + tone) for tone, label, n in counts]


def compliance_checklist(compliance):
    if not compliance:
        return html.P("Compliance runs after the draft is generated.", className="muted small")
    judge = compliance.get("judge") or {}
    rules = judge.get("rules", [])
    items = [html.Li([html.Span("✓" if r["passed"] else "✕", className="tick " + ("ok" if r["passed"] else "bad")),
                      html.Span(r["check"]), html.Span(" · " + r["detail"], className="muted") if r.get("detail") else None]) for r in rules]
    if judge.get("source") == "rules":
        items.append(html.Li([html.Span("–", className="tick"), html.Span("LLM judge"), html.Span(" · skipped, a hard rule already failed", className="muted")]))
    else:
        items.append(html.Li([html.Span("✓" if judge.get("passed") else "✕", className="tick " + ("ok" if judge.get("passed") else "bad")),
                              html.Span("LLM judge (grounding, tone)"),
                              html.Span(" · " + "; ".join(judge.get("issues", [])), className="muted") if judge.get("issues") else None]))
    return html.Ul(items, className="checklist")


def email_preview(text, product):
    subject, _, body = str(text).partition("\nBody:")
    subject = subject.removeprefix("Subject:").strip() if body else ""
    return html.Div([
        html.Div([html.Span("To"), html.Strong("Valued Customer"), html.Span("From"), html.Strong("Bank marketing team")], className="mail-meta"),
        html.Div(subject or "No subject yet", className="mail-subject"),
        html.Pre(body.strip() if body else text, className="email")], className="mail")


def render_workflow_details(state, documents, busy=False):
    """Present the notebook's recorded facts without inventing progress or timings."""
    evidence_ids = state.get("evidence", [])
    by_id = {d["chunk_id"]: d for d in documents}
    evidence = []
    for chunk_id in evidence_ids:
        doc = by_id.get(chunk_id)
        evidence.append(html.Article([
            html.Div([html.Strong(chunk_id), html.Span(
                f"{doc['product'].title()} · Page {doc['page']} · placeholder document" if doc else "Source unavailable",
                className="muted")], className="inspection-heading"),
            html.P(doc["text"] if doc else "This recorded source is no longer in the knowledge base.")
        ], className="evidence-card"))
    trace = state.get("trace", [])
    trajectory = html.Ol([
        html.Li([html.Span(f"{i + 1:02d}", className="trace-number"), html.Span(step)],
                className="trace-route" if step.startswith("Supervisor ->") else "trace-agent")
        for i, step in enumerate(trace)
    ], className="agent-timeline") if trace else html.P("Agent steps appear as they complete.", className="muted")
    timings = state.get("node_latency_ms", [])
    total = sum(t["ms"] for t in timings)
    slowest = max(timings, key=lambda t: t["ms"]) if timings else None
    timing_rows = [html.Div([
        html.Div([html.Span(f"{i + 1:02d}  " + t["node"].replace("_", " ").title()),
                  html.Strong(f"{t['ms']:,.1f} ms")], className="inspection-heading"),
        html.Div(html.Div(style={"width": f"{100 * t['ms'] / slowest['ms'] if slowest['ms'] else 0:.2f}%"}),
                 className="timing-track")
    ], className="timing-row") for i, t in enumerate(timings)]
    timing_content = [html.Div([
        html.Div([html.Strong(f"{total / 1000:,.2f} s"), html.Span("Total recorded execution")]),
        html.Div([html.Strong(str(len(timings))), html.Span("Completed node calls")]),
        html.Div([html.Strong(slowest["node"].replace("_", " ").title() if slowest else "—"), html.Span("Slowest recorded call")])
    ], className="stats timing-stats"),
        html.P("Elapsed execution per completed call, including repeated nodes and revisions. Human approval waiting time is excluded. Bars are relative to the slowest call.", className="hint"),
        *(timing_rows or [html.P("Timing appears after the first node completes.", className="muted")])]
    lane_svg = swimlane.live_swimlane(state, busy)
    return html.Div([
        html.H3("Workflow details"),
        dcc.Tabs(id="workflow-details-tabs", value="swimlane", persistence=True, persistence_type="memory", children=[
            dcc.Tab(label="Swimlane", value="swimlane", children=html.Div([
                html.P("Live flow for this client, drawn from the recorded agent trace. Numbered steps are agents; grey pills are supervisor routing; dashed boxes are the tools each agent called.", className="hint"),
                html.Div(html.Img(src=swimlane.data_uri(lane_svg), alt="Workflow swimlane", className="lane-img"), className="lane-scroll")
            ], className="inspection-panel")),
            dcc.Tab(label=f"Evidence · {len(evidence_ids)}", value="evidence", children=html.Div(
                evidence or [html.P("Approved evidence appears after email generation.", className="muted")], className="inspection-panel")),
            dcc.Tab(label=f"Agent trace · {len(trace)}", value="trace", children=html.Div([
                html.P("Recorded execution order, including supervisor routing and revision loops.", className="hint"), trajectory,
                html.Details([html.Summary("State transitions"), html.Ul([html.Li(step) for step in state.get("status_history", [])])])
            ], className="inspection-panel")),
            dcc.Tab(label="Timing", value="timing", children=html.Div(timing_content, className="inspection-panel"))
        ]),
        html.Details([html.Summary("Raw workflow data"), html.Pre(json.dumps({k: state.get(k) for k in
            ("evidence", "trace", "node_latency_ms", "status_history")}, indent=2), className="history")])
    ], className="workflow-inspection")


def render_evaluation(report):
    summary = report.get("summary", {})
    percent = lambda v: f"{v:.1%}" if v is not None else "—"
    cases = report.get("cases", [])
    result = [html.Div([html.Span(report["status"].replace("_", " ").title(), className="badge"),
                       html.Span("  " + report["run_id"][:12] + " · " + local_time(report["started_at"]), className="muted")]),
              html.P(report.get("phase", "Saved metric report"), role="status")]
    if report.get("error"):
        result.append(html.P(report["error"], role="alert"))
    if cases:
        result += [html.Div([
            html.Div([html.Strong(f"{summary.get('passed', 0)}/{summary.get('cases_completed', 0)}"), html.Span("Live cases passed")]),
            html.Div([html.Strong(percent(summary.get("pass_rate"))), html.Span("Live pass rate")]),
            html.Div([html.Strong(percent(summary.get("red_team_pass_rate"))), html.Span("Red-team pass rate")]),
            html.Div([html.Strong(str(summary.get("errors", 0))), html.Span("Execution errors")])], className="stats"),
            table([{k: c[k] for k in ("name", "category", "status", "latency_ms")} for c in cases]),
            html.Details([html.Summary("Actual inputs, outputs, and expected behavior"), *[
                html.Details([html.Summary(c["name"] + " · " + c["status"]),
                              html.Pre(json.dumps({"expected": c["expected"], "actual": c["actual"]}, indent=2, ensure_ascii=False), className="history")]) for c in cases]])]
    ranking = report.get("ranking")
    if ranking:
        result += [html.H3("Ranking metrics · " + ranking["scope"].replace("_", " ")),
                   table([{"metric": k, "value": str(v)} for k, v in ranking["summary"].items()]),
                   table(ranking["per_product"]),
                   table([{"product": p, "historical_ownership_auc": v} for p, v in ranking["ownership_auc"].items()]),
                   html.Details([html.Summary("Customer-level ranking outcomes"), table(ranking["detail"])])]
    if report.get("operations"):
        result += [html.H3("Observed campaign operations"), table([{"metric": k, "value": str(v)} for k, v in report["operations"].items()])]
    controls = report.get("controls")
    if controls:
        result.append(html.Details([html.Summary("Notebook control regressions · " + controls["status"]),
                    table(controls.get("checks", [])), html.Pre(controls.get("execution_log", controls.get("error", "")), className="history")]))
    result.append(html.Details([html.Summary("Run configuration & provenance"), html.Pre(json.dumps({"model": report.get("model"),
        "started_at": report["started_at"], "finished_at": report.get("finished_at"), "duration_ms": report.get("duration_ms"),
        "case_latency": {k: summary.get(k) for k in ("median_case_ms", "p95_case_ms")}, "provenance": report.get("provenance")}, indent=2), className="history")]))
    return result


def kpi(value, label, tone=""):
    return html.Div([html.Strong(str(value)), html.Span(label)], className="kpi " + tone)


def create_app(service):
    app = Dash(__name__, title="NextBestAI · Campaign studio", assets_folder=str(Path(__file__).parent / "assets"),
               meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
               # IBM Plex for the interface and the email letter; system fonts are used if offline.
               external_stylesheets=["https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;500&display=swap"])
    secret_path = Path(__file__).parent / "runtime" / "session_secret.key"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("NEXTBEST_SESSION_SECRET") and not secret_path.exists():
        try:
            with secret_path.open("x", encoding="utf-8") as out:
                out.write(secrets.token_hex(32))
        except FileExistsError:
            pass
    app.server.secret_key = os.environ.get("NEXTBEST_SESSION_SECRET") or secret_path.read_text(encoding="utf-8")
    app.server.config.update(SESSION_COOKIE_NAME="nextbest_session", SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                             PERMANENT_SESSION_LIFETIME=timedelta(days=7))
    products = list(service.ns["PRODUCTS"])

    @app.server.route("/health")
    def health():
        # Lightweight liveness probe for monitoring; never exposes customer data.
        return jsonify(status="ok", mode=service.mode, model=service.ns["MODEL"],
                       busy=any(c["busy"] for c in service.campaigns.values()))

    def owner():
        if "owner" not in session:
            session["owner"] = uuid.uuid4().hex
        session.permanent = True
        return session["owner"]

    client_options = [{"label": f"{r['client_id']} · Age {r['age']} · {r['eligible']}", "value": r["client_id"]} for r in service.customers()]
    live = service.mode == "live"

    nav = [("campaign", "Campaign"), ("evaluation", "Evaluation lab"), ("how", "How it works")]
    app.layout = html.Div([
        dcc.Store(id="campaign", storage_type="session"), dcc.Store(id="audience"), dcc.Store(id="revision-event"), dcc.Store(id="reset-event"),
        dcc.Store(id="evaluation-run"), dcc.Store(id="review-sig"), dcc.Store(id="progress-sig"),
        dcc.Download(id="evaluation-download"),
        dcc.ConfirmDialog(id="reset-confirm", message="Clear campaign history, email drafts, review decisions, checkpoints and evaluation reports? Audit rows are archived first. Customer data and trained models are kept."),
        dcc.Interval(id="poll", interval=POLL_MS),
        # ------------------------------------------------------------------ sidebar
        html.Nav([
            html.Div([html.Span("N", className="brand-mark"), html.Div([html.Strong("NextBestAI"), html.Span("Campaign studio")], className="brand-text")], className="brand"),
            dcc.RadioItems(id="view", value="campaign", className="side-nav", inputClassName="side-nav-input", labelClassName="side-nav-item",
                           options=[{"label": label, "value": value} for value, label in nav]),
            html.Div([
                html.Div([html.Span(className="dot-live " + ("on" if live else "demo")),
                          html.Span(("Live AI · " + service.ns["MODEL"]) if live else "Preview mode, simulated AI")], className="side-status"),
                html.Button("Reset workspace", id="reset-workspace", className="side-reset")], className="side-foot"),
        ], className="sidebar", **{"aria-label": "Main"}),
        # ------------------------------------------------------------------ content
        html.Main([
            html.Div(id="reset-message", role="status", className="toast"),
            html.Div(id="campaign-view", children=[
                html.Header([
                    html.Div([html.H1("Campaign"), html.P("Pick clients, let the agents score and draft, then review every email before it is sent.", className="lede")]),
                    html.Div(id="kpis", className="queue-summary")], className="page-head"),
                html.Section([
                    dcc.RadioItems(id="run-mode", options=[{"label": "Single client", "value": "single"}, {"label": "Batch clients", "value": "batch"}],
                                   value="single", className="segmented", inputClassName="seg-input", labelClassName="seg-item"),
                    html.Div([dcc.Dropdown(id="single-client", options=client_options, placeholder="Choose a client", clearable=True)],
                             id="single-run-panel", className="run-picker"),
                    html.Div([dcc.Dropdown(id="batch-clients", options=client_options, value=list(service.audience_ids), multi=True, placeholder="Choose clients")],
                             id="batch-run-panel", className="run-picker", style={"display": "none"}),
                    html.Button("Run single client", id="start", className="primary"),
                    html.Div([html.Div(id="audience-message", role="status"), html.Div(id="action-message", role="status")], className="run-notes"),
                ], className="run-bar", **{"aria-label": "Run clients"}),
                html.Div([
                    # ------------------------------------------- queue
                    html.Section([
                        html.Div([html.H2("Campaign pipeline"), html.Div(id="progress")], className="panel-head"),
                        html.P("Synthetic demo data. Click a client to open it.", className="fine"),
                        html.Div(id="campaign-table", className="queue"),
                        html.Div(id="campaign-message", role="alert", className="alert-line")], className="panel queue-panel"),
                    # ------------------------------------------- workspace
                    html.Section([
                        html.Div([html.H2("Client workspace", id="workspace-title"),
                                  html.Div(dcc.Dropdown(id="review-client", placeholder="Switch client", persistence=True, persistence_type="session",
                                                        clearable=False), className="client-switch")], className="panel-head"),
                        html.Div(id="journey"),
                        html.Div([
                            html.Div(id="review-detail", className="review-main"),
                            html.Aside([
                                html.Div(id="review-message", role="status", className="notice"),
                                html.Div([html.Button("Approve and send", id="approve", className="primary"),
                                          html.Button("Revise", id="open-revise", className="ghost"),
                                          html.Button("Reject", id="reject", className="danger")], id="decision-row", className="decision-buttons", style={"display": "none"}),
                                html.Div([html.Label("Tell the AI how to edit this email", htmlFor="revision-note"),
                                          html.P("Your note is applied while keeping the selected product and every compliance rule." if live else "Preview mode can only shorten a draft. Start with --mode live for other edits.", className="fine"),
                                          dcc.Textarea(id="revision-note", placeholder="For example: make it shorter and warmer.", maxLength=2000),
                                          html.Button("Generate revised draft", id="revise", className="primary")], id="revision-panel", style={"display": "none"}, className="revision"),
                                html.Div(id="review-submit-message", role="alert", className="alert-line"),
                                html.Div([html.H3("Did the client reply?"),
                                          html.P("Log the client's real answer. Interested clients go to application and approval, which happens outside this prototype.", className="fine"),
                                          dcc.Input(id="response-note", maxLength=500, placeholder="Optional note, e.g. will visit the branch"),
                                          html.Div([html.Button("Client interested", id="respond-yes", className="primary"),
                                                    html.Button("Not interested", id="respond-no", className="ghost")], className="decision-buttons"),
                                          html.Div(id="response-message", role="alert", className="alert-line")],
                                         id="response-panel", style={"display": "none"}, className="response-panel"),
                                html.Div(id="review-side")], className="review-aside")], className="review-grid"),
                        html.Div(id="review-details")], className="panel workspace-panel", **{"aria-label": "Client workspace"}),
                ], className="split"),
            ]),
            # ------------------------------------------------------------------ evaluation
            html.Div(id="evaluation-view", style={"display": "none"}, children=[
                html.Header([html.Div([html.H1("Evaluation lab"), html.P("Test the agents and the ranking model, and keep every report.", className="lede")])], className="page-head"),
                html.Section([html.H2("Agent and red-team tests"),
                    html.P("Runs real model calls against an isolated copy of the data. Needs --mode live.", className="fine"),
                    html.Div([html.Button("Run agent and red-team tests", id="run-controls", className="primary"),
                              html.Button("Download report", id="download-evaluation", className="ghost")], className="decision-buttons"),
                    html.Div(id="evaluation-submit", role="status", className="alert-line"),
                    dcc.Dropdown(id="evaluation-history", placeholder="Open a saved report"),
                    dcc.Loading(html.Div(id="control-results"))], className="panel"),
                html.Section([html.H2("Product ranking metrics"),
                    html.P("Compares product rankings with acquisition outcomes. Holdout clients were never used for training.", className="fine"),
                    dcc.RadioItems(id="metric-scope", options=[{"label": "Unseen holdout", "value": "holdout"}, {"label": "Campaign clients", "value": "campaign"}],
                                   value="holdout", className="segmented", inputClassName="seg-input", labelClassName="seg-item"),
                    html.Button("Run metric test", id="metrics", className="primary"), dcc.Loading(html.Div(id="metric-results"))], className="panel"),
            ]),
            # ------------------------------------------------------------------ how it works
            html.Div(id="how-view", style={"display": "none"}, children=[
                html.Header([html.Div([html.H1("How it works"), html.P("Your business flow and architecture, drawn from what the code actually does.", className="lede")])], className="page-head"),
                html.Section([html.H2("Business flow"),
                    html.P("Your Business Flow Chart lanes with the agent or screen that now does each step. Application processing stays outside the prototype.", className="fine"),
                    html.Div(html.Img(src=app.get_asset_url("business_flow.svg"), alt="Business flow swimlane", className="lane-img lane-fit"), className="lane-scroll")], className="panel"),
                html.Section([html.H2("Final architecture"),
                    html.P("Each box of the architecture diagram and the code behind it. Synthetic and placeholder data must be replaced before production.", className="fine"),
                    architecture_view()], className="panel"),
                html.Section([html.H2("Agent workflow"),
                    html.P("Each client runs as its own LangGraph thread. The supervisor only routes to steps the rules allow, so the LLM cannot skip the age gate, compliance or your approval.", className="fine"),
                    html.Div(html.Img(src=app.get_asset_url("swimlane.svg"), alt="Agent workflow swimlane", className="lane-img lane-fit"), className="lane-scroll"),
                    html.Div([
                        html.Div([html.Strong("Hard gates"), html.P("Request grammar, age 18 to 60, a compliance pass and your approval are checked again before anything is sent.")]),
                        html.Div([html.Strong("Two-layer compliance"), html.P("Fixed rules for privacy, rates, guarantees and tone run first. The LLM judge only sees drafts that pass, and fails closed.")]),
                        html.Div([html.Strong("Human in the loop"), html.P("Every eligible client pauses for your review. Revise sends your note back to the Email Agent, up to 3 times.")]),
                        html.Div([html.Strong("Full audit"), html.P("Trace, decisions, evidence and timings are written to campaign_audit. Reset archives rows instead of deleting them.")]),
                    ], className="explain-grid")], className="panel"),
            ]),
            html.Footer("NextBestAI campaign studio. Emails are simulated sends; no message leaves this computer.")
        ], className="content")
    ], className="shell")

    @app.callback(Output("campaign-view", "style"), Output("evaluation-view", "style"), Output("how-view", "style"), Input("view", "value"))
    def switch_view(view):
        hide = {"display": "none"}
        return ({} if view == "campaign" else hide, {} if view == "evaluation" else hide, {} if view == "how" else hide)

    @app.callback(Output("reset-confirm", "displayed"), Input("reset-workspace", "n_clicks"), prevent_initial_call=True)
    def confirm_reset(_):
        return True

    @app.callback(Output("campaign", "data", allow_duplicate=True), Output("audience", "data", allow_duplicate=True),
                  Output("review-client", "value"), Output("metric-results", "children", allow_duplicate=True),
                  Output("control-results", "children", allow_duplicate=True),
                  Output("reset-event", "data"), Output("reset-message", "children"),
                  Output("review-message", "children", allow_duplicate=True), Output("action-message", "children", allow_duplicate=True),
                  Output("audience-message", "children", allow_duplicate=True),
                  Input("reset-confirm", "submit_n_clicks"), prevent_initial_call=True)
    def reset_workspace(_):
        try:
            result = service.reset_workspace()
            return (None, None, None, None, None, uuid.uuid4().hex,
                    f"Workspace reset. Cleared {result['campaigns_cleared']} campaigns, archived {result['audit_records_cleared']} audit records, and cleared {result['evaluations_cleared']} evaluation reports.",
                    "", "", "")
        except ValueError as exc:
            return (no_update,) * 6 + (str(exc),) + (no_update,) * 3

    @app.callback(Output("single-client", "value"), Output("batch-clients", "value"), Output("run-mode", "value"),
                  Input("reset-event", "data"), prevent_initial_call=True)
    def reset_run_controls(_):
        return None, list(service.audience_ids), "single"

    @app.callback(Output("evaluation-run", "data"), Output("evaluation-history", "value"), Output("evaluation-submit", "children"),
                  Input("run-controls", "n_clicks"), Input("reset-event", "data"), prevent_initial_call=True)
    def controls(_, __):
        if ctx.triggered_id == "reset-event":
            return None, None, ""
        try:
            run_id = service.evaluations.start(owner())
            return run_id, None, ""
        except Exception as exc:
            return no_update, no_update, str(exc)

    @app.callback(Output("control-results", "children"), Output("evaluation-history", "options"), Output("run-controls", "disabled"),
                  Output("reset-workspace", "disabled"), Input("poll", "n_intervals"), Input("evaluation-run", "data"),
                  Input("evaluation-history", "value"), Input("reset-event", "data"), State("view", "value"))
    def evaluation_progress(_, run_id, selected, __, view):
        active = bool(service.evaluations.active)
        if view != "evaluation" and ctx.triggered_id == "poll":
            # Skip DB reads while the lab is hidden; only keep the buttons' disabled state current.
            return no_update, no_update, active, active
        history = service.evaluations.store.history(owner())
        options = [{"value": h["run_id"], "label": local_time(h["started_at"]) + " · " + h["kind"].replace("_", " ") + " · " + h["status"].replace("_", " ")} for h in history]
        chosen = selected or run_id or (history[0]["run_id"] if history else None)
        if not chosen:
            return html.P("Run live evaluations or select a saved report.", className="muted"), options, active, active
        try:
            report = service.evaluations.store.get(chosen, owner())
            return render_evaluation(report), options, active, active
        except ValueError:
            return None, options, active, active

    @app.callback(Output("evaluation-download", "data"), Input("download-evaluation", "n_clicks"),
                  State("evaluation-run", "data"), State("evaluation-history", "value"), prevent_initial_call=True)
    def download_evaluation(_, run_id, selected):
        history = service.evaluations.store.history(owner())
        chosen = selected or run_id or (history[0]["run_id"] if history else None)
        if not chosen:
            return no_update
        report = service.evaluations.store.get(chosen, owner())
        return dcc.send_string(json.dumps(report, indent=2, ensure_ascii=False), "evaluation-" + chosen[:12] + ".json")

    @app.callback(Output("single-run-panel", "style"), Output("batch-run-panel", "style"),
                  Output("start", "children"), Input("run-mode", "value"))
    def run_mode(mode):
        batch = mode == "batch"
        return ({"display": "none"} if batch else {}, {} if batch else {"display": "none"},
                "Run batch clients" if batch else "Run single client")

    @app.callback(Output("audience", "data"), Output("audience-message", "children"),
                  Input("run-mode", "value"), Input("single-client", "value"), Input("batch-clients", "value"))
    def run_selection(mode, single, batch):
        ids = ([single] if single else []) if mode == "single" else (batch or [])
        if not ids:
            return [], "Select a client to begin." if mode == "single" else "Select at least one client for this batch."
        try:
            ids = service.validate_ids(ids)
            profiles = {r["client_id"]: r for r in service.customers()}
            blocked = sum(profiles[i]["eligible"] != "Eligible" for i in ids)
            return ids, html.Div([html.Strong(f"{len(ids)} client{'s' if len(ids) != 1 else ''} selected"),
                html.Span(f" · {blocked} age blocked · Each eligible client needs an individual review.")])
        except ValueError as exc:
            return [], str(exc)

    @app.callback(Output("campaign", "data"), Output("action-message", "children"), Input("start", "n_clicks"),
                  State("run-mode", "value"), State("single-client", "value"), State("batch-clients", "value"),
                  State("campaign", "data"), prevent_initial_call=True)
    def start_campaign(_, mode, single, batch, current):
        try:
            ids = ([single] if single else []) if mode == "single" else (batch or [])
            if not ids:
                raise ValueError("Select a client first." if mode == "single" else "Select at least one client for this batch.")
            if current:
                try:
                    previous = service.snapshot(current, owner())
                except ValueError:
                    previous = None
                if previous and (previous["busy"] or any(s.get("pending_review") for s in previous["states"].values())):
                    raise ValueError("Finish or reject pending reviews before starting another campaign.")
            campaign_id = service.start(ids, owner())
            return campaign_id, "Campaign started. Progress updates automatically."
        except ValueError as exc:
            return no_update, str(exc)

    @app.callback(Output("kpis", "children"), Output("progress", "children"), Output("campaign-table", "children"),
                  Output("review-client", "options"), Output("campaign-message", "children"), Output("progress-sig", "data"),
                  Output("review-client", "value", allow_duplicate=True),
                  Input("poll", "n_intervals"), Input("campaign", "data"), State("progress-sig", "data"), State("review-client", "value"),
                  prevent_initial_call="initial_duplicate")
    def progress(_, cid, last_sig, selected):
        empty = html.Div([html.Strong("No clients yet"), html.P("Choose a client above and press Run. The agents' progress shows up here.")], className="empty")
        if not cid:
            return queue_summary([]), None, empty, [], "", None, no_update
        try:
            c = service.snapshot(cid, owner())
        except ValueError:
            return queue_summary([]), None, empty, [], "Campaign history was cleared. Run clients again to continue.", None, no_update
        sig = signature(c["busy"], c["job"], c["error"], c["review_jobs"], selected, {k: (v.get("trace"), v.get("status"), v.get("pending_review"),
                        v.get("error"), v.get("client_response")) for k, v in c["states"].items()})
        if sig == last_sig:
            return (no_update,) * 7
        # Open the first client that needs a person, so the flow always shows what to do next.
        pick = no_update
        if not selected or selected not in c["ids"]:
            ready = [i for i in c["ids"] if c["states"].get(i, {}).get("trace")]
            need = [i for i in ready if next_action(c["states"][i])[2] == "need"]
            if need or ready:
                pick = (need or ready)[0]
                selected = pick
        status = html.Span([html.Span(className="spinner"), c["job"]] if c["busy"] else "All agents idle", className="pill " + ("info" if c["busy"] else "idle"))
        profiles = {r["client_id"]: r for r in service.customers()}
        options = [{"label": f"{i} · {next_action(c['states'][i])[0]}", "value": i} for i in c["ids"] if c["states"].get(i, {}).get("trace")]
        return queue_summary(c["states"].values()), status, queue_list(c, profiles, products, selected), options, c["error"], sig, pick

    @app.callback(Output("review-client", "value", allow_duplicate=True), Input({"type": "pick-client", "index": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def pick_client(clicks):
        # Re-rendered buttons fire with n_clicks=None; only a real click selects a client.
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            return no_update
        return ctx.triggered_id["index"]

    @app.callback(Output({"type": "pick-client", "index": ALL}, "className"), Input("review-client", "value"),
                  State({"type": "pick-client", "index": ALL}, "id"))
    def highlight_row(selected, ids):
        return ["q-row" + (" selected" if i["index"] == selected else "") for i in ids]

    @app.callback(Output("review-detail", "children"), Output("review-side", "children"), Output("review-details", "children"),
                  Output("approve", "disabled"), Output("reject", "disabled"), Output("open-revise", "disabled"), Output("revise", "disabled"),
                  Output("review-message", "children"), Output("review-sig", "data"), Output("response-panel", "style"),
                  Output("journey", "children"), Output("workspace-title", "children"), Output("decision-row", "style"),
                  Input("review-client", "value"), Input("poll", "n_intervals"), Input("campaign", "data"), State("review-sig", "data"))
    def show_review(client_id, _, cid, last_sig):
        hidden = {"display": "none"}
        if not cid or not client_id:
            empty = html.Div([html.Strong("Nothing open"), html.P("Run a client, then pick it in the pipeline. Its email, scores and checks open here.")], className="empty")
            return empty, None, None, True, True, True, True, "", None, hidden, None, "Client workspace", hidden
        try:
            c = service.snapshot(cid, owner())
        except ValueError:
            return html.Div([html.Strong("Workspace cleared")], className="empty"), None, None, True, True, True, True, "", None, hidden, None, "Client workspace", hidden
        s = c["states"].get(client_id, {})
        comp = s.get("compliance") or {}
        pending = s.get("pending_review", False) and not c["busy"]
        feedback = review_feedback(c, s)
        revisions_left = service.ns["MAX_REVISIONS"] - s.get("revision_count", 0)
        buttons = (not (pending and comp.get("passed")), not pending, not (pending and revisions_left > 0), not (pending and revisions_left > 0))
        job = c.get("review_jobs", {}).get(client_id, {})
        sig = signature(client_id, c["busy"], s.get("trace"), s.get("status"), s.get("draft"), s.get("final_email"), s.get("pending_review"),
                        s.get("error"), job, comp, s.get("client_response"))
        response_style = {} if s.get("final_email") and not s.get("client_response") else hidden
        decision_style = {} if s.get("pending_review") else hidden
        if sig == last_sig:
            return (no_update, no_update, no_update, *buttons, feedback, no_update, response_style, no_update, no_update, decision_style)
        terminal = bool(s.get("final_email")) or s.get("approval_status") == "reject"
        revision_state = job.get("status")
        revision_message = {"generating": "Writing a new version from your note…", "ready": "Revised email ready.",
                            "unchanged": "The AI returned the same email. Try a more specific note.", "incomplete": "The edit was only partly applied.",
                            "failed": "Revision failed: " + job.get("error", "")}.get(revision_state)
        displayed_email = s.get("final_email") or s.get("draft") or job.get("previous_draft")
        heading = "Revised email" if revision_state == "ready" else "Previous email, revision in progress" if revision_state == "generating" else \
            "Sent email" if s.get("final_email") else f"Draft, revision {s.get('revision_count', 0)}"
        main = [html.H3(heading) if displayed_email else None,
                html.P(revision_message, className="fine") if revision_message and not terminal else None,
                html.P("Your note: " + job["instruction"], className="fine") if job else None,
                html.P(job.get("error"), role="alert", className="alert-line") if job.get("error") else None,
                email_preview(displayed_email, s.get("product")) if displayed_email else
                html.Div([html.Strong("No email for this client"), html.P(next_action(s, c["busy"])[1])], className="empty"),
                html.Details([html.Summary("Compare with the previous version"), html.Pre(job["previous_draft"], className="email plain")]) if job.get("previous_draft") else None,
                html.Details([html.Summary("Draft history and your notes"), *[html.Pre(m["role"] + ": " + m["message"], className="history") for m in s.get("conversation", [])]])
                if s.get("conversation") else None]
        scores = (s.get("propensity") or {}).get("scores", {})
        side = [html.Div([html.H3("Why this product"), html.Span(f"{revisions_left} revisions left", className="fine")], className="side-head"),
                score_bars(scores, s.get("product"), products) if scores else html.P("Not scored.", className="fine"),
                html.P(s.get("propensity_summary") or s.get("recommendation_reason") or "", className="why") if (s.get("propensity_summary") or s.get("recommendation_reason")) else None,
                html.H3("Compliance checks"), compliance_checklist(comp)]
        reply = s.get("client_response")
        if reply:
            side[0:0] = [html.H3("Client reply"), html.Div([html.Span("Interested" if reply["interested"] else "Not interested",
                         className="pill " + ("ok" if reply["interested"] else "idle")), html.Span(" " + local_time(reply["recorded_at"]), className="fine")]),
                         html.P(reply["note"], className="why") if reply["note"] else None]
        details = render_workflow_details(s, service.ns["docs"], c["busy"])
        title = [f"Client {client_id}", html.Span((s.get("product") or "").title(), className="badge")] if s.get("product") else f"Client {client_id}"
        return (main, side, details, *buttons, feedback, sig, response_style, journey_rail(s, c["busy"] and not s.get("pending_review")), title, decision_style)

    @app.callback(Output("revision-panel", "style"), Output("revision-note", "value"), Output("review-submit-message", "children", allow_duplicate=True),
                  Input("open-revise", "n_clicks"), Input("review-client", "value"), Input("revision-event", "data"), Input("reset-event", "data"),
                  prevent_initial_call=True)
    def revision_panel(_, __, ___, ____):
        return ({"display": "block"}, "", "") if ctx.triggered_id == "open-revise" else ({"display": "none"}, "", "")

    @app.callback(Output("review-submit-message", "children"), Output("revision-event", "data"),
                  Input("approve", "n_clicks"), Input("reject", "n_clicks"), Input("revise", "n_clicks"),
                  State("campaign", "data"), State("review-client", "value"), State("revision-note", "value"), prevent_initial_call=True)
    def review(_, __, ___, cid, client_id, note):
        try:
            service.review(cid, owner(), client_id, ctx.triggered_id, note or "")
            return "", uuid.uuid4().hex
        except ValueError as exc:
            return str(exc), no_update

    @app.callback(Output("response-message", "children"), Output("response-note", "value"),
                  Input("respond-yes", "n_clicks"), Input("respond-no", "n_clicks"),
                  State("campaign", "data"), State("review-client", "value"), State("response-note", "value"), prevent_initial_call=True)
    def respond(_, __, cid, client_id, note):
        try:
            service.record_response(cid, owner(), client_id, ctx.triggered_id == "respond-yes", note or "")
            return "", ""
        except ValueError as exc:
            return str(exc), no_update

    @app.callback(Output("metric-results", "children"), Input("metrics", "n_clicks"), State("campaign", "data"), State("metric-scope", "value"), prevent_initial_call=True)
    def metrics(_, cid, scope):
        if not cid and scope == "campaign":
            return "Start a campaign first."
        try:
            summary, detail, products_, comparison, workflow = service.metrics(cid, owner(), scope)
            run_id = service.evaluations.save_metrics(owner(), scope, summary, detail, products_, comparison, workflow)
            summary_rows = [{"metric": str(k), "value": str(v)} for k, v in summary.items()]
            auc_rows = [{"product": p, "historical_ownership_auc": v} for p, v in service.ns["HOLDOUT_AUC"].items()]
            return [html.P("Saved metric run " + run_id[:12]), html.H3("Ranking performance · " + scope), table(summary_rows), html.H3("Per-product landed recall"), table(safe_records(products_)),
                    html.H3("Historical ownership holdout AUC"), table(auc_rows),
                    html.P("Recall@k: acquired products in top k / all acquired products. Landed recall@k: landed product in top k among customers with an acquisition. NaN means no valid denominator. Campaign results may include training customers.", className="hint"),
                    html.H3("Workflow selection vs outcomes"), table([{"metric": "approval_coverage" if k == "simulated_release_coverage" else k, "value": str(v)} for k, v in workflow.items()]),
                    html.Details([html.Summary("Customer ranking details"), table(safe_records(detail))]),
                    html.Details([html.Summary("Workflow outcomes"), table(safe_records(comparison))]),
                    html.P("Acquisition outcomes are used for evaluation and do not influence model scores or email drafting.", className="hint")]
        except (ValueError, KeyError) as exc:
            return str(exc)

    app.config.suppress_callback_exceptions = True
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NextBestAI Dash campaign studio")
    parser.add_argument("--mode", choices=["demo", "live"], default="live")
    parser.add_argument("--host", default=os.environ.get("NEXTBEST_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NEXTBEST_PORT", "8117")))
    args = parser.parse_args()

    service = CampaignService(args.mode)
    app = create_app(service)
    app.run(host=args.host, port=args.port, debug=False)
