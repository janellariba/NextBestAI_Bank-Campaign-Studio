"""SVG swimlane renderer for the NextBestAI agent workflow.

One engine draws both the live per-client swimlane (built from the workflow's recorded
trace and status history, never invented) and the static "How it works" system diagram.
"""
from __future__ import annotations

import base64
from html import escape

FONT = "Inter,Segoe UI,Arial,sans-serif"
STYLE = {  # kind: (fill, stroke, text, dash)
    "agent": ("#e5f3ef", "#167568", "#16243c", ""),
    "route": ("#eff3f8", "#9aa9bd", "#46566e", ""),
    "human": ("#fff4e0", "#c9861a", "#5c3d07", ""),
    "studio": ("#e8eef6", "#163c62", "#163c62", ""),
    "tool": ("#ffffff", "#7d8ea6", "#46566e", "4 3"),
    "ok": ("#167568", "#167568", "#ffffff", ""),
    "bad": ("#fae9e9", "#9b4242", "#7a2e2e", ""),
    "pending": ("#fff4e0", "#c9861a", "#5c3d07", "5 3"),
    "error": ("#fae9e9", "#9b4242", "#7a2e2e", "5 3"),
}
LANE_W = 150
NODE_W, NODE_H = 116, 42
ROUTE_W = 92


def _wrap(text, width=17):
    words, lines, line = str(text).split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return lines[:3] or [""]


def render(lanes, nodes, edges, title="", notes=()):
    """lanes: [(key, label, height)]; nodes: dicts with id, label, lane, col, kind, optional dy/sub/badge/narrow.
    edges: (from_id, to_id, label, style) with style in {"flow", "loop", "tool", "alt"}."""
    lane_y, y = {}, 44 if title else 12
    for key, _, h in lanes:
        lane_y[key] = (y, h)
        y += h
    height = y + 16
    cols = max((n["col"] for n in nodes), default=0) + 1
    col_w = []
    for c in range(cols):
        members = [n for n in nodes if n["col"] == c]
        col_w.append(ROUTE_W + 18 if members and all(n.get("narrow") for n in members) else NODE_W + 24)
    col_x, x = [], LANE_W + 14
    for w in col_w:
        col_x.append(x + w / 2)
        x += w
    width = x + 14
    pos = {}
    for n in nodes:
        top, h = lane_y[n["lane"]]
        w = ROUTE_W if n.get("narrow") else NODE_W
        cx, cy = col_x[n["col"]], top + h / 2 + n.get("dy", 0)
        pos[n["id"]] = (cx, cy, w)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" font-family="{FONT}">',
           '<defs>'
           '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#546780"/></marker>'
           '<marker id="l" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#c9861a"/></marker>'
           '<marker id="r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#9b4242"/></marker>'
           '</defs>', f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>']
    if title:
        out.append(f'<text x="16" y="28" font-size="15" font-weight="700" fill="#16243c">{escape(title)}</text>')
    for i, (key, label, h) in enumerate(lanes):
        top, _ = lane_y[key]
        out.append(f'<rect x="0" y="{top}" width="{width:.0f}" height="{h}" fill="{"#f6f8fc" if i % 2 == 0 else "#ffffff"}"/>')
        out.append(f'<line x1="0" y1="{top}" x2="{width:.0f}" y2="{top}" stroke="#e1e7f0"/>')
        out.append(f'<rect x="0" y="{top}" width="{LANE_W}" height="{h}" fill="#163c62" opacity="{0.94 - 0.07 * (i % 2)}"/>')
        for j, line in enumerate(_wrap(label, 16)):
            out.append(f'<text x="14" y="{top + h / 2 - 6 + j * 15:.0f}" font-size="12" font-weight="600" fill="#ffffff">{escape(line)}</text>')
    out.append(f'<line x1="0" y1="{height - 16}" x2="{width:.0f}" y2="{height - 16}" stroke="#e1e7f0"/>')
    for lane, c0, c1, text in notes:
        top, h = lane_y[lane]
        x0, x1 = col_x[c0] - col_w[c0] / 2 + 4, col_x[c1] + col_w[c1] / 2 - 4
        out.append(f'<rect x="{x0:.0f}" y="{top + h - 22}" width="{x1 - x0:.0f}" height="16" rx="8" fill="#e8eef6"/>')
        out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{top + h - 10.5}" font-size="10" text-anchor="middle" fill="#426783">{escape(text)}</text>')

    labels = []
    for src, dst, label, style in edges:
        sx, sy, sw = pos[src]
        tx, ty, tw = pos[dst]
        colour, marker, dash = {"flow": ("#546780", "a", ""), "loop": ("#c9861a", "l", "6 4"),
                                "tool": ("#9aa9bd", "", "3 3"), "alt": ("#9b4242", "r", "6 4")}[style]
        if style == "tool":  # vertical association between an agent and the tool/data it calls
            down = ty > sy  # the tools lane can sit above (audit) or below (agents) the calling node
            d = f"M{sx:.1f},{sy + (NODE_H / 2 if down else -NODE_H / 2):.1f} V{ty + (-NODE_H / 2 if down else NODE_H / 2):.1f}"
        elif tx > sx + 1:  # forward: elbow right, down/up, right
            x0, x1 = sx + sw / 2, tx - tw / 2
            mid = x0 + (x1 - x0) / 2
            d = f"M{x0:.1f},{sy:.1f} H{mid:.1f} V{ty:.1f} H{x1 - 2:.1f}"
            # Cross-lane hops: label near the target so stacked branch labels never collide.
            labels.append((mid, ty - 34 if abs(ty - sy) > 2 else sy - 9, label))
        elif abs(tx - sx) <= 1:  # same column, different lane
            d = f"M{sx:.1f},{sy + (NODE_H / 2 if ty > sy else -NODE_H / 2):.1f} V{ty + (-NODE_H / 2 if ty > sy else NODE_H / 2) - (2 if ty > sy else -2):.1f}"
            labels.append((sx + 6, (sy + ty) / 2, label))
        elif abs(ty - sy) < 2:  # backward within a lane: arc over the top
            d = f"M{sx:.1f},{sy - NODE_H / 2:.1f} C{sx:.1f},{sy - NODE_H / 2 - 34:.1f} {tx:.1f},{ty - NODE_H / 2 - 34:.1f} {tx:.1f},{ty - NODE_H / 2 - 2:.1f}"
            labels.append(((sx + tx) / 2, sy - NODE_H / 2 - 26, label))
        else:  # backward across lanes: leave left, travel to target column, enter vertically
            enter = ty - NODE_H / 2 - 2 if ty > sy else ty + NODE_H / 2 + 2
            d = f"M{sx - sw / 2:.1f},{sy:.1f} H{tx:.1f} V{enter:.1f}"
            labels.append(((sx - sw / 2 + tx) / 2, sy - 8, label))
        out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.6"'
                   + (f' stroke-dasharray="{dash}"' if dash else "") + (f' marker-end="url(#{marker})"' if marker else "") + "/>")

    for n in nodes:
        cx, cy, w = pos[n["id"]]
        fill, stroke, text, dash = STYLE[n.get("kind", "agent")]
        x0, y0 = cx - w / 2, cy - NODE_H / 2
        out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w}" height="{NODE_H}" rx="{20 if n.get("narrow") else 9}" fill="{fill}" stroke="{stroke}" stroke-width="{2.2 if n.get("current") else 1.4}"'
                   + (f' stroke-dasharray="{dash}"' if dash else "") + ">"
                   + ('<animate attributeName="opacity" values="1;.45;1" dur="1.6s" repeatCount="indefinite"/>' if n.get("kind") in ("pending",) else "")
                   + "</rect>")
        lines = _wrap(n["label"], 13 if n.get("narrow") else 17)
        if n.get("sub"):
            lines = lines[:2] + [n["sub"]]
        start = cy - (len(lines) - 1) * 6.5 + 4
        for j, line in enumerate(lines):
            small = n.get("sub") and j == len(lines) - 1
            out.append(f'<text x="{cx:.1f}" y="{start + j * 13:.1f}" font-size="{9.5 if small else 11}" font-weight="{400 if small else 600}" text-anchor="middle" fill="{text}"'
                       + (' opacity=".8"' if small else "") + f">{escape(line)}</text>")
        if n.get("badge"):
            out.append(f'<circle cx="{x0 + 2:.1f}" cy="{y0 + 2:.1f}" r="9" fill="#163c62"/><text x="{x0 + 2:.1f}" y="{y0 + 5.5:.1f}" font-size="9" font-weight="700" text-anchor="middle" fill="#fff">{escape(str(n["badge"]))}</text>')
    for lx, ly, label in labels:
        if label:
            w = 6.1 * len(label) + 10
            out.append(f'<rect x="{lx - w / 2:.1f}" y="{ly - 9:.1f}" width="{w:.1f}" height="15" rx="4" fill="#ffffff" opacity=".92"/>'
                       f'<text x="{lx:.1f}" y="{ly + 2:.1f}" font-size="10" text-anchor="middle" fill="#46566e">{escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def data_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


# ---------------------------------------------------------------- live per-client swimlane
LIVE_LANES = [("human", "Marketer (human review)", 76), ("orch", "Orchestration · User Prompt & Supervisor", 76),
              ("agents", "Specialist agents", 84), ("tools", "Tools, models & data", 76), ("audit", "Release & audit", 76)]
AGENT_LANE = {"User Prompt": "orch", "Eligibility Agent": "agents", "Propensity Agent": "agents", "Email Agent": "agents",
              "Compliance Agent": "agents", "Human Reviewer": "human", "Release Agent": "audit", "Reject Agent": "audit", "Record Agent": "audit"}
TOOLS = {"Eligibility Agent": ("get_client_profile", "Bank DB"), "Propensity Agent": ("predict_propensity", "ML + LLM summary"),
         "Email Agent": ("RAG + LLM draft", "BM25 + SBERT"), "Compliance Agent": ("Rules + LLM judge", "fail-closed"),
         "Record Agent": ("campaign_audit", "Audit DB")}
# Display names follow the project's architecture diagram; the trace keeps the notebook's names for audit.
DISPLAY = {"Propensity Agent": "Scoring Agent"}
OUTCOME_BY_AGENT = {"Eligibility Agent": ("ELIGIBILITY CHECKED", "BLOCKED"), "Compliance Agent": ("COMPLIANCE PASSED", "COMPLIANCE FAILED"),
                    "Human Reviewer": ("APPROVED", "REJECTED")}


def live_swimlane(state, busy=False):
    """Build the swimlane strictly from the recorded trace and status_history."""
    trace = state.get("trace", []) or []
    history = state.get("status_history", []) or []
    # The k-th history entry tagged "(Agent)" belongs to the k-th time that agent appears in the trace.
    outcomes = {}
    for entry in history:
        if entry.endswith(")") and "(" in entry:
            agent = entry[entry.rindex("(") + 1:-1]
            outcomes.setdefault(agent, []).append(entry.split(" -> ", 1)[-1].rsplit(" (", 1)[0])
    seen, nodes, edges = {}, [], []
    for i, step in enumerate(trace):
        nid = f"s{i}"
        if step.startswith("Supervisor ->"):
            target = step.split("->", 1)[1].strip().replace("_agent", "").replace("_", " ")
            nodes.append({"id": nid, "label": "Supervisor", "sub": "→ " + target, "lane": "orch", "col": i, "kind": "route", "narrow": True})
        else:
            k = seen.get(step, 0)
            seen[step] = k + 1
            result = (outcomes.get(step) or [None] * (k + 1))[k] if k < len(outcomes.get(step, [])) else None
            kind = "human" if step == "Human Reviewer" else "agent"
            sub = None
            if result:
                sub = result.title().replace("Revision Requested", "Revise")
                if result.startswith("BLOCKED"):
                    sub = "Blocked · " + result.split(":", 1)[-1].strip().split()[0].lower()
                sub = sub if len(sub) <= 22 else sub[:21] + "…"
                if result.startswith(("BLOCKED", "COMPLIANCE FAILED", "REJECTED", "ERROR")):
                    kind = "bad"
                elif result.startswith("COMPLETED"):
                    kind, sub = "ok", "Released (simulated)"
            nodes.append({"id": nid, "label": DISPLAY.get(step, step), "sub": sub, "lane": AGENT_LANE.get(step, "orch"), "col": i, "kind": kind, "badge": i + 1})
            if step in TOOLS:
                tool, detail = TOOLS[step]
                nodes.append({"id": nid + "t", "label": tool, "sub": detail, "lane": "tools", "col": i, "kind": "tool"})
                edges.append((nid, nid + "t", "", "tool"))
        if i:
            prev = next(n for n in nodes if n["id"] == f"s{i - 1}")
            # Columns only move forward in a recorded trace; colour the hops that start a loop.
            if prev["label"] == "Human Reviewer" and prev.get("sub") == "Revise":
                edges.append((prev["id"], nid, "revise", "loop"))
            elif prev["label"] == "Compliance Agent" and prev["kind"] == "bad":
                edges.append((prev["id"], nid, "failed", "loop"))
            else:
                edges.append((prev["id"], nid, "", "flow"))
    terminal = str(state.get("status", "")).startswith(("COMPLETED", "REJECTED", "BLOCKED", "ERROR")) or "Record Agent" in trace
    extra = None
    response = state.get("client_response")
    if response and trace:
        # Business flow: Sales Team "Client interested?" then the Approval lane (outside the prototype).
        col = len(trace)
        nodes.append({"id": "resp", "label": "Client interested?", "sub": "Yes" if response["interested"] else "No",
                      "lane": "human", "col": col, "kind": "human"})
        nodes.append({"id": "next_step", "label": "Application & approval" if response["interested"] else "Closed",
                      "sub": "outside prototype" if response["interested"] else "not interested",
                      "lane": "audit", "col": col + 1, "kind": "tool" if response["interested"] else "bad", "current": True})
        edges += [(f"s{len(trace) - 1}", "resp", "", "flow"), ("resp", "next_step", "yes" if response["interested"] else "no", "flow" if response["interested"] else "alt")]
        return render(LIVE_LANES, nodes, edges)
    if state.get("final_email") and not state.get("error") and not busy:
        extra = {"label": "Client interested?", "sub": "log the reply", "lane": "human", "kind": "pending"}
    elif state.get("error"):
        extra = {"label": "Workflow error", "sub": "see message", "lane": "orch", "kind": "error"}
    elif state.get("pending_review") and not busy:
        extra = {"label": "Your decision", "sub": "approve · revise · reject", "lane": "human", "kind": "pending"}
    elif busy and not terminal:
        extra = {"label": "Working…", "sub": "next step running", "lane": "orch", "kind": "pending"}
    if extra:
        extra.update(id="next", col=len(trace), current=True)
        nodes.append(extra)
        if trace:
            edges.append((f"s{len(trace) - 1}", "next", "", "flow"))
    elif trace:
        for n in nodes:
            if n["id"] == f"s{len(trace) - 1}":
                n["current"] = True
    if not nodes:
        nodes.append({"id": "q", "label": "Queued", "sub": "waiting for worker", "lane": "orch", "col": 0, "kind": "pending"})
    return render(LIVE_LANES, nodes, edges)


# ---------------------------------------------------------------- static system swimlane
def system_swimlane():
    lanes = [("human", "Marketer", 80), ("studio", "Campaign studio (Dash + CampaignService)", 80),
             ("orch", "Orchestration (LangGraph Supervisor)", 92), ("agents", "Specialist agents", 92),
             ("tools", "Tools, models & data", 80), ("audit", "Release & audit", 112)]
    N = lambda i, label, lane, col, kind="agent", sub=None, dy=0: {"id": i, "label": label, "lane": lane, "col": col, "kind": kind, "sub": sub, "dy": dy}
    nodes = [
        N("pick", "Select clients", "human", 0, "human", "single or batch"),
        N("queue", "Queue workflows", "studio", 1, "studio", "1 thread per client"),
        N("gate", "User Prompt gate", "orch", 2, "agent", "strict request grammar"),
        N("elig", "Eligibility Agent", "agents", 3, "agent", "age 18 to 60"),
        N("t_elig", "Bank DB", "tools", 3, "tool", "get_client_profile"),
        N("block", "Do not contact", "audit", 4, "bad", "blocked, audited", dy=28),
        N("prop", "Scoring Agent", "agents", 4, "agent", "propensity model + why"),
        N("t_prop", "Logistic models", "tools", 4, "tool", "per product + drivers"),
        N("email", "Email Agent", "agents", 5, "agent", "grounded draft"),
        N("t_email", "Product KB", "tools", 5, "tool", "BM25 + SBERT RAG"),
        N("comp", "Compliance Agent", "agents", 6, "agent", "rules, then LLM judge"),
        N("t_comp", "Rule checks", "tools", 6, "tool", "privacy, claims, format"),
        N("review", "Email review panel", "studio", 7, "studio", "draft, evidence, trace"),
        N("decide", "Review decision", "human", 7, "human", "approve, revise, reject"),
        N("release", "Release Agent", "audit", 8, "ok", "simulated send", dy=-28),
        N("reject", "Reject Agent", "audit", 8, "bad", "nothing sent", dy=28),
        N("record", "Record Agent", "audit", 9, "agent", "campaign_audit"),
        N("outcome", "Outcome & metrics", "studio", 10, "studio", "evaluation lab"),
    ]
    edges = [("pick", "queue", "Run", "flow"), ("queue", "gate", "stream()", "flow"), ("gate", "elig", "valid", "flow"),
             ("elig", "t_elig", "", "tool"), ("elig", "block", "age gate fails", "alt"), ("elig", "prop", "", "flow"),
             ("prop", "t_prop", "", "tool"), ("prop", "email", "", "flow"), ("email", "t_email", "", "tool"),
             ("email", "comp", "", "flow"), ("comp", "t_comp", "", "tool"), ("comp", "email", "failed: redraft (max 2)", "loop"),
             ("comp", "review", "interrupt()", "flow"), ("review", "decide", "", "flow"), ("decide", "email", "revise + note", "loop"),
             ("decide", "release", "approve", "flow"), ("decide", "reject", "reject", "alt"),
             ("release", "record", "", "flow"), ("reject", "record", "", "flow"), ("record", "outcome", "", "flow")]
    notes = [("orch", 3, 6, "Supervisor routes between every agent step (rule-based; LLM picks only when 2 moves are valid)")]
    return render(lanes, nodes, edges, "NextBestAI · agent workflow swimlane", notes)


# ---------------------------------------------------------------- business flow swimlane
def business_swimlane():
    """The team's Business Flow Chart lanes, showing which agent or screen now does each step."""
    lanes = [("mkt", "Marketing Team", 84), ("ds", "DS Team", 100), ("sales", "Sales Team", 100), ("appr", "Approval", 84)]
    N = lambda i, label, lane, col, kind="agent", sub=None, dy=0: {"id": i, "label": label, "lane": lane, "col": col, "kind": kind, "sub": sub, "dy": dy}
    nodes = [
        N("start", "Call for campaign", "mkt", 0, "human", "Run clients"),
        N("ask", "Ask for leads", "mkt", 1, "studio", "Supervisor routes"),
        N("elig", "Eligibility Agent", "ds", 2, "agent", "age 18 to 60"),
        N("score", "Scoring Agent", "ds", 3, "agent", "ML model, best product"),
        N("block", "End: do not contact", "appr", 2, "bad", "age blocked, audited"),
        N("email", "Email Agent", "sales", 4, "agent", "RAG-grounded draft"),
        N("comp", "Compliance Agent", "sales", 5, "agent", "rules + LLM judge"),
        N("review", "Review email", "mkt", 6, "human", "approve, revise, reject"),
        N("send", "Send email", "sales", 7, "ok", "simulated release"),
        N("reply", "Client interested?", "sales", 8, "human", "logged by sales"),
        N("process", "Process application", "appr", 8, "tool", "outside prototype"),
        N("end", "End", "appr", 9, "agent", "audit record"),
    ]
    edges = [("start", "ask", "", "flow"), ("ask", "elig", "", "flow"), ("elig", "score", "", "flow"),
             ("elig", "block", "too young / old", "alt"), ("score", "email", "best product", "flow"),
             ("email", "comp", "", "flow"), ("comp", "email", "failed: redraft", "loop"), ("comp", "review", "", "flow"),
             ("review", "email", "revise", "loop"), ("review", "send", "approve", "flow"), ("send", "reply", "", "flow"),
             ("reply", "process", "YES", "flow"), ("reply", "end", "NO", "alt"), ("process", "end", "", "flow")]
    notes = [("ds", 2, 3, "AI-assisted: agent identifies the best product per customer"),
             ("sales", 4, 7, "AI-assisted: automated e-mailer to contact client")]
    return render(lanes, nodes, edges, "NextBestAI · business flow (as implemented)", notes)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    target = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "assets" / "swimlane.svg")
    target.write_text(system_swimlane(), encoding="utf-8")
    business = target.with_name("business_flow.svg")
    business.write_text(business_swimlane(), encoding="utf-8")
    print("Wrote", target, "and", business)
