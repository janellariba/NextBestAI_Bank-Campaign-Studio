"""Opt-in live Ollama smoke test; no email is sent. Writes a reviewable report."""
import json
import time
from backend import CampaignService, ROOT


def wait(service, cid):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        state = service.snapshot(cid, "live-check")
        if not state["busy"]:
            if state["error"]:
                raise RuntimeError(state["error"])
            return state["states"]["112250"]
        time.sleep(.5)
    raise TimeoutError("Live workflow exceeded 15 minutes")


if __name__ == "__main__":
    service = CampaignService("live")
    try:
        print("Live models loaded; generating initial email.", flush=True)
        cid = service.start(["112250"], "live-check")
        initial = wait(service, cid)
        assert initial.get("pending_review"), initial
        print("Initial draft:", initial["draft"], flush=True)
        note = "Change the subject to 'A closer look at our credit card'. Keep the body short, with only two sentences after the required greeting."
        service.review(cid, "live-check", "112250", "revise", note)
        revised = wait(service, cid)
        assert revised.get("pending_review"), revised
        assert initial["draft"] != revised["draft"], "Live model returned unchanged draft"
        assert "a closer look at our credit card" in revised["draft"].splitlines()[0].lower(), "Requested subject was not applied"
        assert revised["compliance"]["passed"], revised["compliance"]
        print("Subject change verified; testing full body rewrite.", flush=True)
        translation_note = "Translate the entire email body into Filipino. Only keep 'Dear Valued Customer' and 'credit card' in English. Rewrite all other sentences in Filipino."
        service.review(cid, "live-check", "112250", "revise", translation_note)
        translated = wait(service, cid)
        assert "Our credit card" not in translated["draft"] and "To learn how" not in translated["draft"], translated["draft"]
        assert sum(word in translated["draft"].lower().split() for word in ("ang", "aming", "mga", "para", "na")) >= 3, translated["draft"]
        assert translated["draft"] != revised["draft"], "Body rewrite did not change the email"
        assert translated["compliance"]["passed"], translated["compliance"]
        report = {"mode": "live", "model": service.ns["MODEL"], "original_email": initial["draft"],
                  "revision_instruction": note, "revised_email": revised["draft"], "compliance": revised["compliance"],
                  "revision_count": translated["revision_count"], "trace": translated["trace"],
                  "body_rewrite_instruction": translation_note, "body_rewrite_email": translated["draft"],
                  "body_rewrite_compliance": translated["compliance"],
                  "word_filter_ids": service.resolve("show users aged between 18 and 30 with ADB above 50,000")}
        path = ROOT / "runtime" / "live-verification.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Revised draft:", revised["draft"], flush=True)
        print("Body rewrite:", translated["draft"], flush=True)
        service.review(cid, "live-check", "112250", "reject")
        wait(service, cid)
        print("PASS: live generation, requested revision, compliance and word filtering. Report:", path, flush=True)
    finally:
        service.worker.shutdown(wait=True)
