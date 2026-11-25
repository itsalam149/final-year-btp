"""
Agent Controller and ReAct Execution Engine.
Orchestrates prompt formatting, intent understanding, grammar-constrained decoding,
tool execution dispatch, confirmation gates, and memory management.
"""

from typing import Any, Callable, Dict, List, Optional
import json
import re
from datetime import datetime

from mobile_agent.storage.db import LocalStorage
from mobile_agent.tools.registry import ToolRegistry
from mobile_agent.tools.handlers import MobileToolHandlers
from mobile_agent.tools.schemas import SecurityLevel


SYSTEM_PROMPT_TEMPLATE = """You are JARVIS, an on-device privacy-preserving intelligent mobile assistant running 100% locally on the user's smartphone.
You operate completely offline with ZERO internet access. All data stays strictly on this device.

Current Device Context:
- Current Date/Time: {current_time}
- Platform: Android 14 (Offline Mobile Sandbox)
- Battery: Safe / Disconnected from Cloud

Available Tools:
{tools_doc}

OUTPUT FORMAT RULE:
You MUST ALWAYS output a single valid JSON object adhering strictly to one of two formats:

Format 1: When a tool action is required to fulfill the user's request:
```json
{{
  "type": "action",
  "thought": "Reasoning step explaining why this tool is selected and what parameters were extracted.",
  "tool": "tool_name",
  "parameters": {{
    "param1": "value1"
  }}
}}
```

Format 2: When responding directly to the user or reporting completed actions:
```json
{{
  "type": "message",
  "thought": "Brief internal summary of the conversational response.",
  "content": "Natural language response to show to the user."
}}
```
Never output text outside the JSON block. Do not hallucinate tools that are not listed above.
"""


class MockLocalInferenceEngine:
    """
    Lightweight, deterministic rule-based simulation engine for development,
    unit testing, and automated verification without loading multi-gigabyte models.
    """

    def generate(self, prompt: str, grammar: Optional[str] = None) -> str:
        # Extract the last user message from the prompt
        user_msg = ""
        for line in reversed(prompt.split("\n")):
            if line.startswith("User:"):
                user_msg = line.replace("User:", "").strip().lower()
                break

        # Check for reminder intent
        if "remind me to" in user_msg or "set a reminder" in user_msg:
            title = user_msg.replace("remind me to", "").replace("set a reminder to", "").strip()
            # Simple heuristic for time
            due_time = "2026-11-27T09:00:00"
            if "tomorrow" in user_msg and "10 am" in user_msg:
                due_time = "2026-11-27T10:00:00"
            elif "tomorrow" in user_msg and "11 am" in user_msg:
                due_time = "2026-11-27T11:00:00"
            return json.dumps({
                "type": "action",
                "thought": f"The user wants to set a reminder about '{title}'.",
                "tool": "create_reminder",
                "parameters": {
                    "title": title.capitalize(),
                    "due_time_iso": due_time,
                    "priority": "normal"
                }
            })

        # Check for calendar search intent
        if "calendar" in user_msg or "schedule" in user_msg or "free slot" in user_msg:
            date_iso = "2026-11-27"
            return json.dumps({
                "type": "action",
                "thought": "User is inquiring about calendar schedule.",
                "tool": "search_calendar",
                "parameters": {
                    "date_iso": date_iso,
                    "query": "meeting" if "meeting" in user_msg else None
                }
            })

        # Check for notes intent
        if "note" in user_msg:
            if "create" in user_msg or "save" in user_msg or "draft a note" in user_msg:
                title = "Study Session" if "study session" in user_msg else "Quick Note"
                return json.dumps({
                    "type": "action",
                    "thought": "User wants to create a new local note.",
                    "tool": "manage_notes",
                    "parameters": {
                        "action": "create",
                        "title": title,
                        "content": user_msg
                    }
                })
            elif "list" in user_msg or "show" in user_msg:
                return json.dumps({
                    "type": "action",
                    "thought": "User wants to list saved notes.",
                    "tool": "manage_notes",
                    "parameters": {"action": "list"}
                })

        # Check for document search / summary intent
        if "pdf" in user_msg or "document" in user_msg or "proposal" in user_msg:
            if "summarize" in user_msg or "summary" in user_msg:
                return json.dumps({
                    "type": "action",
                    "thought": "User wants a summary of a local document.",
                    "tool": "read_document_summary",
                    "parameters": {
                        "filename": "project_proposal.pdf",
                        "focus_topic": "research objectives"
                    }
                })
            else:
                return json.dumps({
                    "type": "action",
                    "thought": "User is searching for a local document.",
                    "tool": "search_local_docs",
                    "parameters": {
                        "query": "proposal" if "proposal" in user_msg else "report",
                        "max_results": 3
                    }
                })

        # Check for message drafting intent
        if "draft" in user_msg and ("message" in user_msg or "whatsapp" in user_msg or "sms" in user_msg):
            recipient = "Rahul" if "rahul" in user_msg else "Contact"
            return json.dumps({
                "type": "action",
                "thought": "User wants to draft an outgoing message to a contact.",
                "tool": "draft_message",
                "parameters": {
                    "recipient": recipient,
                    "message_body": "I will be late.",
                    "platform": "whatsapp" if "whatsapp" in user_msg else "sms"
                }
            })

        # Check for app opening intent (Level 2)
        if "open " in user_msg or "launch " in user_msg or "start " in user_msg:
            app_target = user_msg.replace("open ", "").replace("launch ", "").replace("start ", "").replace("app", "").strip()
            return json.dumps({
                "type": "action",
                "thought": f"User requests launching mobile app '{app_target}'.",
                "tool": "open_app",
                "parameters": {
                    "app_name": app_target or "settings"
                }
            })

        # Check for UI automation intent (Level 3 Accessibility)
        if any(w in user_msg for w in ["click", "tap", "press", "type in", "read screen", "inspect screen"]):
            if "read screen" in user_msg or "inspect screen" in user_msg or "what is on my screen" in user_msg:
                return json.dumps({
                    "type": "action",
                    "thought": "User requests inspecting visible screen text via Accessibility Service.",
                    "tool": "automate_ui",
                    "parameters": {
                        "action": "read_screen"
                    }
                })
            elif "type" in user_msg:
                return json.dumps({
                    "type": "action",
                    "thought": "User requests entering text into an active UI field.",
                    "tool": "automate_ui",
                    "parameters": {
                        "action": "type",
                        "target": "search",
                        "input_text": "BTP final year report"
                    }
                })
            else:
                target_btn = user_msg.replace("click ", "").replace("tap ", "").replace("press ", "").strip()
                return json.dumps({
                    "type": "action",
                    "thought": f"User requests clicking UI button '{target_btn}' via Accessibility.",
                    "tool": "automate_ui",
                    "parameters": {
                        "action": "click",
                        "target": target_btn or "confirm"
                    }
                })

        # Default conversational message
        return json.dumps({
            "type": "message",
            "thought": "Standard conversational response.",
            "content": f"I am your offline assistant. I can help manage notes, create reminders, check your schedule, launch apps, or interact with screens without internet connectivity."
        })


class MobileAgent:
    """
    Core Controller for the Privacy-First Offline Mobile Agent.
    """

    def __init__(
        self,
        storage: Optional[LocalStorage] = None,
        tool_registry: Optional[ToolRegistry] = None,
        llm_engine: Optional[Any] = None,
        session_id: str = "default_session"
    ):
        self.storage = storage or LocalStorage(":memory:")
        self.registry = tool_registry or ToolRegistry()
        self.handlers = MobileToolHandlers(self.storage)

        # Bind standard handlers
        self.registry.bind_handler("create_reminder", self.handlers.create_reminder)
        self.registry.bind_handler("search_calendar", self.handlers.search_calendar)
        self.registry.bind_handler("manage_notes", self.handlers.manage_notes)
        self.registry.bind_handler("search_local_docs", self.handlers.search_local_docs)
        self.registry.bind_handler("read_document_summary", self.handlers.read_document_summary)
        self.registry.bind_handler("draft_message", self.handlers.draft_message)
        self.registry.bind_handler("open_app", self.handlers.open_app)
        self.registry.bind_handler("automate_ui", self.handlers.automate_ui)

        self.llm_engine = llm_engine or MockLocalInferenceEngine()
        self.session_id = session_id
        self.max_history_turns = 6

    def build_prompt(self, user_input: str) -> str:
        """Construct the complete prompt including system prompt, tools, and history."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tools_doc = self.registry.format_tools_for_prompt()
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(current_time=now_str, tools_doc=tools_doc)

        # Fetch recent history
        history = self.storage.get_history(self.session_id, limit=self.max_history_turns)
        history_str = ""
        for item in history:
            history_str += f"{item['role'].capitalize()}: {item['content']}\n"

        full_prompt = f"{system_prompt}\n\nRecent Conversation:\n{history_str}\nUser: {user_input}\nAssistant:"
        return full_prompt

    def _parse_llm_json(self, raw_output: str) -> Dict[str, Any]:
        """Safely parse JSON from raw LLM output, stripping markdown fences if present."""
        cleaned = raw_output.strip()
        # Remove markdown code blocks if the model emitted them
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-len("```")].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Fallback if raw text isn't clean JSON
            return {
                "type": "message",
                "thought": "Fallback JSON parser",
                "content": raw_output
            }

    def process_message(
        self,
        user_input: str,
        user_confirmed: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the full ReAct pipeline on a user message:
        1. Log user message to conversation history
        2. Format prompt with context & tools
        3. Infer with LLM engine
        4. If action, validate and execute with confirmation gate
        5. Return structured response to user interface
        """
        # Step 1: Save user message in local SQLite
        self.storage.append_message(self.session_id, "user", user_input)

        # Step 2: Build prompt & call LLM
        prompt = self.build_prompt(user_input)
        raw_output = self.llm_engine.generate(prompt)
        payload = self._parse_llm_json(raw_output)

        output_type = payload.get("type", "message")
        thought = payload.get("thought", "")

        # Case A: Model outputs a direct conversational response
        if output_type == "message":
            content = payload.get("content", "")
            self.storage.append_message(self.session_id, "assistant", content)
            return {
                "status": "COMPLETED",
                "response_type": "MESSAGE",
                "thought": thought,
                "display_text": content
            }

        # Case B: Model requests a tool action
        elif output_type == "action":
            tool_name = payload.get("tool", "")
            parameters = payload.get("parameters", {})

            # Execute tool through validated registry
            exec_result = self.registry.execute_tool(
                tool_name=tool_name,
                parameters=parameters,
                user_confirmed=user_confirmed
            )

            status = exec_result.get("status")

            # Log to local security audit log
            self.storage.log_audit(
                tool_name=tool_name,
                parameters=parameters,
                status=status,
                user_confirmed=user_confirmed
            )

            # Check if confirmation is required
            if status == "CONFIRMATION_REQUIRED":
                confirm_msg = f"Confirmation required before executing {tool_name}."
                self.storage.append_message(self.session_id, "assistant", confirm_msg)
                return {
                    "status": "AWAITING_CONFIRMATION",
                    "response_type": "ACTION_CONFIRMATION_REQUIRED",
                    "thought": thought,
                    "tool": tool_name,
                    "parameters": parameters,
                    "display_text": f"This action requires your confirmation:\nTool: {tool_name}\nDetails: {json.dumps(parameters, indent=2)}"
                }

            elif status == "SUCCESS":
                tool_data = exec_result.get("result", {})
                tool_msg = tool_data.get("message", "Action completed successfully.")
                self.storage.append_message(self.session_id, "assistant", tool_msg)
                return {
                    "status": "COMPLETED",
                    "response_type": "ACTION_EXECUTED",
                    "thought": thought,
                    "tool": tool_name,
                    "parameters": parameters,
                    "result_data": tool_data,
                    "display_text": tool_msg
                }

            else:  # ERROR
                err_msg = exec_result.get("message", "An error occurred during tool execution.")
                self.storage.append_message(self.session_id, "assistant", f"Failed: {err_msg}")
                return {
                    "status": "FAILED",
                    "response_type": "ERROR",
                    "thought": thought,
                    "tool": tool_name,
                    "error_message": err_msg,
                    "display_text": f"Sorry, could not complete action: {err_msg}"
                }

        # Fallback
        return {
            "status": "COMPLETED",
            "response_type": "MESSAGE",
            "thought": "Unrecognized output type",
            "display_text": str(payload)
        }
