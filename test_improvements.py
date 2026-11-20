"""Tests for the v2 guardrails, batch resilience, audit archive and swimlane UI."""
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing

import swimlane
from app import create_app, client_status
from backend import CampaignService, email_rule_checks


class ImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.service = CampaignService("demo", cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.service.worker.shutdown(wait=True)
        cls.temp.cleanup()

    def wait(self, service, cid, owner="t"):
        end = time.monotonic() + 30
        while service.snapshot(cid, owner)["busy"] and time.monotonic() < end:
            time.sleep(.02)
        return service.snapshot(cid, owner)

    def state(self):
        return {"product": "loan", "evidence": ["C0002"], "client": {"client_id": "112250", "age": 27, "adb": 64321}}

    def test_rules_pass_a_clean_email(self):
        ok = "Subject: Explore our loan\nBody: Dear Valued Customer, a loan is repaid over an agreed term. Please visit our website to learn more."
        self.assertTrue(all(r["passed"] for r in email_rule_checks(self.state(), ok, ["credit card", "loan", "insurance"])))

    def test_rules_catch_privacy_claims_and_pressure(self):
        products = ["credit card", "loan", "insurance"]
        cases = {
            "No customer data": "Dear Valued Customer, at 27 you deserve a loan.",
            "No rates, fees or amounts": "Dear Valued Customer, our loan has a 5% rate.",
            "No approval guarantees": "Dear Valued Customer, your loan is guaranteed.",
            "Low-pressure tone": "Dear Valued Customer, act now for a loan.",
            "No other products": "Dear Valued Customer, try a loan or insurance.",
            "Required greeting": "Hello, explore our loan.",
        }
        for check, body in cases.items():
            with self.subTest(check=check):
                result = {r["check"]: r["passed"] for r in email_rule_checks(self.state(), "Subject: Loan\nBody: " + body, products)}
                self.assertFalse(result[check])

    def test_guard_fails_closed_without_calling_llm_judge(self):
        calls = []
        saved = self.service.ns["ask_json"]
        self.service.ns["ask_json"] = lambda *a: calls.append(a) or {"passed": True, "issues": []}
        try:
            verdict = self.service.ns["judge_email"](self.state(), "Subject: x\nBody: Dear Valued Customer, guaranteed loan approval!")
        finally:
            self.service.ns["ask_json"] = saved
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["source"], "rules")
        self.assertFalse(calls, "LLM judge should not run once a hard rule fails")

    def test_bad_draft_is_held_in_real_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            service = CampaignService("demo", folder)
            try:
                service.ns["ask_llm"] = lambda prompt, system="", **k: "Subject: Act now\nBody: Dear Valued Customer, your credit card is guaranteed at 0% for you."
                cid = service.start(["112250"], "t")
                state = self.wait(service, cid)["states"]["112250"]
                self.assertFalse(state["compliance"]["passed"])
                self.assertEqual(state["compliance"]["judge"]["source"], "rules")
                with self.assertRaises(ValueError):
                    service.review(cid, "t", "112250", "approve")
            finally:
                service.worker.shutdown(wait=True)

    def test_one_failing_client_does_not_abort_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            service = CampaignService("demo", folder)
            original = service._invoke

            def flaky(campaign_id, client_id, payload):
                if client_id == "112250":
                    raise TimeoutError("model timed out")
                return original(campaign_id, client_id, payload)
            service._invoke = flaky
            try:
                cid = service.start(["112250", "564124"], "t")
                c = self.wait(service, cid)
                self.assertIn("1 of 2 clients failed", c["error"])
                self.assertEqual(client_status(c["states"]["112250"])[0], "Error")
                self.assertTrue(c["states"]["564124"].get("pending_review"))
            finally:
                service.worker.shutdown(wait=True)

    def test_reset_archives_audit_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            service = CampaignService("demo", folder)
            try:
                cid = service.start(["112250"], "t")
                self.wait(service, cid)
                service.review(cid, "t", "112250", "reject")
                self.wait(service, cid)
                service.reset_workspace()
                with closing(sqlite3.connect(service.ns["AUDIT_DB"])) as con:
                    self.assertEqual(con.execute("SELECT COUNT(*) FROM campaign_audit").fetchone()[0], 0)
                    row = con.execute("SELECT client_id, human_decision, archived_at FROM campaign_audit_archive").fetchone()
                self.assertEqual(row[:2], ("112250", "reject"))
                self.assertTrue(row[2])
            finally:
                service.worker.shutdown(wait=True)

    def test_swimlane_is_built_from_recorded_trace(self):
        state = {"trace": ["User Prompt", "Supervisor -> eligibility_agent", "Eligibility Agent", "Supervisor -> record_agent", "Record Agent"],
                 "status_history": ["RECEIVED -> REQUEST VALIDATED (User Prompt)", "REQUEST VALIDATED -> BLOCKED: AGE RESTRICTION (Eligibility Agent)"],
                 "status": "BLOCKED: AGE RESTRICTION"}
        svg = swimlane.live_swimlane(state)
        self.assertIn("Blocked · age", svg)
        self.assertIn("get_client_profile", svg)
        self.assertNotIn("Your decision", svg)
        self.assertIn("Your decision", swimlane.live_swimlane({**state, "status": "x", "pending_review": True}))
        self.assertIn("<svg", swimlane.system_swimlane())

    def test_client_response_follows_business_flow(self):
        cid = self.service.start(["564124"], "resp")
        self.wait(self.service, cid, "resp")
        with self.assertRaises(ValueError):  # no reply before the email is sent
            self.service.record_response(cid, "resp", "564124", True)
        self.service.review(cid, "resp", "564124", "approve")
        self.wait(self.service, cid, "resp")
        with self.assertRaises(ValueError):
            self.service.record_response(cid, "resp", "564124", "yes")
        reply = self.service.record_response(cid, "resp", "564124", True, "Asked for a branch visit")
        self.assertIn("APPLICATION & APPROVAL", reply["next_step"])
        with self.assertRaises(ValueError):  # one reply per workflow
            self.service.record_response(cid, "resp", "564124", False)
        state = self.service.snapshot(cid, "resp")["states"]["564124"]
        self.assertEqual(client_status(state), ("Interested · to approval", "ok"))
        with closing(sqlite3.connect(self.service.ns["AUDIT_DB"])) as con:
            row = con.execute("SELECT interested, note, recorded_by FROM client_responses WHERE workflow_id=?", (state["workflow_id"],)).fetchone()
        self.assertEqual(row, (1, "Asked for a branch visit", "resp"))
        svg = swimlane.live_swimlane(state)
        self.assertIn("interested?", svg)
        self.assertIn("Scoring Agent", svg)
        self.assertNotIn(">Propensity Agent<", svg)

    def test_business_flow_diagram_uses_team_lanes(self):
        svg = swimlane.business_swimlane()
        for lane in ("Marketing Team", "DS Team", "Sales Team", "Approval", "interested?", "outside prototype"):
            self.assertIn(lane, svg)

    def test_health_and_views(self):
        app = create_app(self.service)
        client = app.server.test_client()
        health = client.get("/health").get_json()
        self.assertEqual(health["mode"], "demo")
        layout = client.get("/_dash-layout").get_data(as_text=True)
        for text in ("How it works", "Evaluation lab", "Campaign pipeline"):
            self.assertIn(text, layout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
