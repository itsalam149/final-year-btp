"""
End-to-End Test Suite for Privacy-First Offline Mobile Agent.
Verifies Tool Validation, SQLite Storage, Confirmation Gates, and ReAct Loop.
"""

import unittest
import os
import json

from mobile_agent.storage.db import LocalStorage
from mobile_agent.tools.registry import ToolRegistry
from mobile_agent.tools.schemas import SecurityLevel
from mobile_agent.core.agent import MobileAgent, MockLocalInferenceEngine


class TestMobileAgentSystem(unittest.TestCase):

    def setUp(self):
        # Use an in-memory database for clean, isolated tests
        self.storage = LocalStorage(":memory:")
        self.registry = ToolRegistry()
        self.agent = MobileAgent(
            storage=self.storage,
            tool_registry=self.registry,
            llm_engine=MockLocalInferenceEngine(),
            session_id="test_session"
        )

    def tearDown(self):
        self.storage.close()

    def test_tool_registry_validation(self):
        """Test strict parameter validation against tool schemas."""
        # Valid reminder parameters
        is_valid, err = self.registry.validate_call(
            "create_reminder",
            {"title": "Submit BTP Thesis", "due_time_iso": "2026-11-27T12:00:00", "priority": "high"}
        )
        self.assertTrue(is_valid, f"Validation failed unexpectedly: {err}")

        # Missing required parameter ('due_time_iso')
        is_valid, err = self.registry.validate_call(
            "create_reminder",
            {"title": "Submit BTP Thesis"}
        )
        self.assertFalse(is_valid)
        self.assertIn("Missing required parameter 'due_time_iso'", err)

        # Unexpected hallucinated parameter
        is_valid, err = self.registry.validate_call(
            "create_reminder",
            {"title": "Submit BTP Thesis", "due_time_iso": "2026-11-27T12:00:00", "hallucinated_param": 123}
        )
        self.assertFalse(is_valid)
        self.assertIn("Unexpected parameter 'hallucinated_param'", err)

        # Invalid enum value
        is_valid, err = self.registry.validate_call(
            "create_reminder",
            {"title": "Submit BTP Thesis", "due_time_iso": "2026-11-27T12:00:00", "priority": "ultra_urgent"}
        )
        self.assertFalse(is_valid)
        self.assertIn("not in allowed values", err)

    def test_local_storage_crud(self):
        """Test SQLite persistence for notes, reminders, and calendar."""
        # Notes
        note_id = self.storage.create_note("Algorithms", "Dynamic programming concepts.")
        self.assertGreater(note_id, 0)
        retrieved_note = self.storage.get_note_by_title("Algorithms")
        self.assertIsNotNone(retrieved_note)
        self.assertEqual(retrieved_note["content"], "Dynamic programming concepts.")

        # Reminders
        rem_id = self.storage.create_reminder("Call Guide", "2026-11-27T10:00:00", "high")
        self.assertGreater(rem_id, 0)
        reminders = self.storage.list_reminders()
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0]["title"], "Call Guide")

        # Calendar
        cal_id = self.storage.add_calendar_event("BTP Mid-Term Viva", "2026-11-27T14:00:00", "2026-11-27T15:00:00", "SPS-9 DTU")
        self.assertGreater(cal_id, 0)
        events = self.storage.search_calendar("2026-11-27")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["title"], "BTP Mid-Term Viva")

    def test_agent_reminder_flow(self):
        """Test agent parsing and executing a reminder instruction."""
        response = self.agent.process_message("Remind me to call Rahul tomorrow at 10 am")
        self.assertEqual(response["status"], "COMPLETED")
        self.assertEqual(response["response_type"], "ACTION_EXECUTED")
        self.assertEqual(response["tool"], "create_reminder")
        self.assertIn("call rahul", response["display_text"].lower())

        # Verify it was stored in the database
        reminders = self.storage.list_reminders()
        self.assertEqual(len(reminders), 1)
        self.assertIn("call rahul", reminders[0]["title"].lower())

    def test_agent_calendar_search_flow(self):
        """Test agent searching calendar events on target date."""
        self.storage.add_calendar_event("Meeting with Ms. Gull Kaur", "2026-11-27T11:00:00", "2026-11-27T12:00:00", "Cabin 204")
        response = self.agent.process_message("Check my schedule for tomorrow")
        self.assertEqual(response["status"], "COMPLETED")
        self.assertEqual(response["tool"], "search_calendar")
        self.assertEqual(response["result_data"]["event_count"], 1)

    def test_agent_notes_flow(self):
        """Test agent creating and saving a local note."""
        response = self.agent.process_message("Draft a note titled Study Session")
        self.assertEqual(response["status"], "COMPLETED")
        self.assertEqual(response["tool"], "manage_notes")
        self.assertEqual(response["result_data"]["status"], "CREATED")

        note = self.storage.get_note_by_title("Study Session")
        self.assertIsNotNone(note)

    def test_agent_document_search_and_summary_flow(self):
        """Test agent document search and contextual excerpt extraction."""
        # Preload a document into local storage
        self.storage.add_document(
            filename="project_proposal.pdf",
            title="BTP Project Proposal",
            content_text="Project Title: Privacy-First Offline AI Mobile Agent.\nResearch objectives: Deploy quantized LLMs on mobile.\nHardware constraints: 16GB VRAM training, 4GB RAM edge deployment.",
            tags="btp, mobile, ai"
        )

        response = self.agent.process_message("Summarize the research objectives from my project proposal")
        self.assertEqual(response["status"], "COMPLETED")
        self.assertEqual(response["tool"], "read_document_summary")
        self.assertIn("Research objectives", response["result_data"]["excerpt"])

    def test_sensitive_action_confirmation_gate(self):
        """
        Security Test: Outgoing message drafting is classified as REQUIRES_CONFIRMATION.
        Verifies that without user confirmation, the action is blocked and requires confirmation.
        """
        # Step 1: Without confirmation
        response = self.agent.process_message(
            "Draft a WhatsApp message telling Rahul that I will be late",
            user_confirmed=False
        )
        self.assertEqual(response["status"], "AWAITING_CONFIRMATION")
        self.assertEqual(response["response_type"], "ACTION_CONFIRMATION_REQUIRED")
        self.assertEqual(response["tool"], "draft_message")
        self.assertIn("requires your confirmation", response["display_text"])

        # Check audit log: should show CONFIRMATION_REQUIRED
        logs = self.storage.get_audit_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "CONFIRMATION_REQUIRED")
        self.assertEqual(logs[0]["user_confirmed"], 0)

        # Step 2: With explicit user confirmation
        response_confirmed = self.agent.process_message(
            "Draft a WhatsApp message telling Rahul that I will be late",
            user_confirmed=True
        )
        self.assertEqual(response_confirmed["status"], "COMPLETED")
        self.assertEqual(response_confirmed["response_type"], "ACTION_EXECUTED")
        self.assertEqual(response_confirmed["tool"], "draft_message")
        self.assertEqual(response_confirmed["result_data"]["status"], "DRAFT_READY")

        # Check audit log: should now show SUCCESS with user_confirmed=1
        logs_after = self.storage.get_audit_logs()
        self.assertEqual(len(logs_after), 2)
        self.assertEqual(logs_after[0]["status"], "SUCCESS")
        self.assertEqual(logs_after[0]["user_confirmed"], 1)

    def test_gbnf_grammar_file_exists(self):
        """Verify the GBNF grammar definition exists and is valid."""
        grammar_path = os.path.join(
            os.path.dirname(__file__), "..", "mobile_agent", "grammar", "tool_calling.gbnf"
        )
        self.assertTrue(os.path.exists(grammar_path))
        with open(grammar_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("root ::=", content)
        self.assertIn("create_reminder", content)
        self.assertIn("search_calendar", content)
        self.assertIn("manage_notes", content)

    def test_quantization_manager_and_aaqb(self):
        """Verify GGUF model manager and AAQB benchmark execution."""
        from mobile_agent.quantization.manager import GGUFModelManager, ModelPrecision
        from mobile_agent.benchmark.aaqb import AAQBRunner

        mgr = GGUFModelManager()
        spec = mgr.get_spec("llama-3.2-1b", ModelPrecision.Q4_K_M)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.precision, ModelPrecision.Q4_K_M)
        self.assertGreater(spec.estimated_size_mb, 0)

        # Run quick AAQB benchmark on first 5 tasks as smoke test
        from mobile_agent.benchmark.tasks import AAQB_50_TASKS
        runner = AAQBRunner(tasks=AAQB_50_TASKS[:5])
        summary = runner.evaluate_configuration("Smoke Test", precision=ModelPrecision.Q4_K_M, use_grammar=True)
        self.assertEqual(summary.total_tasks, 5)
        self.assertGreaterEqual(summary.syntax_adherence_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
