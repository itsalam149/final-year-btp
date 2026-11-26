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


SYSTEM_PROMPT_TEMPLATE = """You are Offline Mobile AI Agent, an on-device privacy-preserving intelligent mobile assistant developed for DTU CSE GP159 running 100% locally on the user's smartphone.
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

        # Check for reminder or alarm intent
        if "remind" in user_msg or "reminder" in user_msg or "alarm" in user_msg:
            # Extract title cleanly
            m = re.search(r'(?:remind me to|set a reminder to|set a reminder for|schedule a reminder to|create a reminder to|set an alarm to|remind me for|add a reminder to|i need a reminder to|reminder to)\s+([^,\.]+?)(?:\s+(?:at|tomorrow|on|urgent)\b|$)', user_msg)
            title = m.group(1).strip() if m else "Follow up on task"
            title = re.sub(r'^(urgent|please)\s+', '', title).strip()

            # Dynamic time parsing (look for 'at HH:MM' or 'HH:MM am/pm' specifically to avoid matching 5000 steps)
            hour = 9
            minute = 0
            time_m = re.search(r'\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b', user_msg)
            if not time_m:
                time_m = re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', user_msg)

            if time_m:
                h = int(time_m.group(1))
                minute = int(time_m.group(2)) if time_m.group(2) else 0
                meridiem = time_m.group(3)
                if meridiem:
                    if meridiem.lower() == "pm" and h < 12:
                        h += 12
                    elif meridiem.lower() == "am" and h == 12:
                        h = 0
                elif "pm" in user_msg and h < 12:
                    h += 12
                hour = h

            day = "2026-11-27" if "tomorrow" in user_msg else "2026-11-26"
            due_time = f"{day}T{hour:02d}:{minute:02d}:00"
            priority = "high" if "urgent" in user_msg else "normal"

            return json.dumps({
                "type": "action",
                "thought": f"The user wants to set a reminder about '{title}'.",
                "tool": "create_reminder",
                "parameters": {
                    "title": title.capitalize(),
                    "due_time_iso": due_time,
                    "priority": priority
                }
            })

        # Check for calendar schedule search intent (unless explicitly opening the calendar app)
        is_calendar_app_launch = any(w in user_msg for w in ["open calendar", "launch calendar", "calendar app", "start calendar", "switch to calendar", "bring up calendar", "quick open calendar", "i want to open calendar"])
        if not is_calendar_app_launch and ("calendar" in user_msg or "schedule" in user_msg or "free slot" in user_msg or "meeting" in user_msg):
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

        # Check for compound Hisaab / Kharcha / Ledger note with calculation
        if "hisaab" in user_msg or "kharcha" in user_msg or ("note" in user_msg and ("total" in user_msg or "sum" in user_msg)):
            # Parse amounts and names
            text_clean = re.sub(r'(?i)^(?:create\s+a\s+note\s+with\s+hisaab:?|save\s+a\s+note\s+for\s+expense\s+ledger:?|ledger\s+note\s+create\s+karo:?|add\s+expenses\s+to\s+notes:?|notes?\s+(?:pr|pe|mein|par)?\s*(?:ek)?\s*(?:hisaab|kharcha)?\s*(?:likho|banao|daalo|entry\s+karo)?:?)', '', user_msg).strip()
            clean = re.sub(r'(?i)\b(notes?|mein|kharcha|p[re]|ek|hisaab|likho|ki|make|total|and|all|bhi|karo|banao|calculate|final|sum|entry)\b', ' ', text_clean)
            chunks = re.split(r'[,;\n]|(?:\baur\b)', clean)
            pairs = []
            stop = {'rs', 'inr', 'rupaye', 'm', 'like', 'this', 'total', 'all', 'the', 'with', 'create', 'note', 'notes', 'save', 'ledger', 'add', 'expenses', 'banao', 'karo', 'entry'}
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                num_m = re.search(r'(\d+)', chunk)
                words = [w for w in re.findall(r'[a-zA-Z]+', chunk) if w.lower() not in stop and len(w) > 1]
                if num_m and words:
                    amt = int(num_m.group(1))
                    name = words[0].capitalize()
                    pairs.append((name, amt))

            if pairs:
                total_sum = sum(p[1] for p in pairs)
                lines = [f"• {name}: ₹{amt}" for name, amt in pairs]
                content = "\n".join(lines) + f"\n--------------------\nTotal Hisaab: ₹{total_sum}"
                title = f"Hisaab: {len(pairs)} Entries (Total: ₹{total_sum})"
                return json.dumps({
                    "type": "action",
                    "thought": f"Calculated total hisaab ₹{total_sum} across {len(pairs)} entries and creating itemized note.",
                    "tool": "manage_notes",
                    "parameters": {
                        "action": "create",
                        "title": title,
                        "content": content
                    }
                })

        # Check for notes intent
        if "note" in user_msg:
            if "check" in user_msg or "list" in user_msg or "show" in user_msg or "read" in user_msg or "view" in user_msg:
                return json.dumps({
                    "type": "action",
                    "thought": "User wants to list or view saved notes.",
                    "tool": "manage_notes",
                    "parameters": {"action": "list"}
                })
            elif "create" in user_msg or "save" in user_msg or "draft a note" in user_msg or "take a note" in user_msg or "write" in user_msg:
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

        # Check for compound Split Bill calculation
        if ("split" in user_msg or "share" in user_msg) and any(w in user_msg for w in ["bill", "dinner", "total", "party", "lunch", "trip", "cafe", "hotel", "outing", "food", "farewell", "birthday", "buffet", "travel", "cab"]):
            amounts = [int(a) for a in re.findall(r'(?:rs|inr|₹)?\s*(\d{3,6})', user_msg)]
            if amounts:
                total_bill = max(amounts)
                people_m = re.search(r'(\d+)\s*(?:people|persons|logon|members|friends)', user_msg)
                num_people = int(people_m.group(1)) if people_m else 4
                split_share = total_bill // num_people
                
                # Dynamic recipient extraction (e.g. 'with Shilpa', 'with Rahul')
                with_m = re.search(r'with\s+([a-zA-Z]+)', user_msg)
                if with_m:
                    recipient = with_m.group(1).capitalize()
                else:
                    contacts = ["rahul", "priya", "ekansh", "guneet", "bhawishya", "sharukh", "kalli", "haaji", "naari"]
                    recipient = "Friend"
                    for c in contacts:
                        if c in user_msg:
                            recipient = c.capitalize()
                            break

                msg_body = f"Hey {recipient}, our total bill was ₹{total_bill} ({num_people} people). Your split share is ₹{split_share}."
                return json.dumps({
                    "type": "action",
                    "thought": f"Calculated split share ₹{split_share} ({total_bill} / {num_people}) and drafting message.",
                    "tool": "draft_message",
                    "parameters": {
                        "recipient": recipient,
                        "message_body": msg_body,
                        "platform": "whatsapp"
                    }
                })

        # Check for app opening / media playback intent (Level 2)
        media_verbs = [
            "open ", "launch ", "start ", "play ", "listen to ", "watch ", "search ",
            "navigate to ", "directions to ", "find route to ", "take me to ", "show route to "
        ]
        media_apps = ["spotify", "youtube", "maps", "whatsapp", "settings", "calculator", "camera", "gmail", "calendar", "clock"]
        if any(w in user_msg for w in media_verbs) or any(app in user_msg for app in media_apps):
            app = None
            for candidate in ["youtube", "spotify", "maps", "google maps", "settings", "camera", "calculator", "whatsapp", "gmail", "clock", "calendar"]:
                if candidate in user_msg:
                    app = "maps" if candidate == "google maps" else candidate
                    break

            if not app:
                if any(w in user_msg for w in ["listen", "song", "music", "track", "album", "lofi", "acoustic", "playlist", "hits", "unplugged", "slowed", "mix", "remix", "soundtrack", "play "]):
                    app = "spotify"
                elif any(w in user_msg for w in ["watch", "video", "tutorial", "lecture", "tour", "highlights", "documentary"]):
                    app = "youtube"
                elif any(w in user_msg for w in ["navigate", "direction", "route", "take me to", "maps"]):
                    app = "maps"
                else:
                    app = "whatsapp"

            query = ""
            prefixes_to_strip = [
                "search spotify for ", "search youtube for ", "watch on youtube ", "play on youtube ",
                "play on spotify ", "find on youtube ", "open youtube with ", "open youtube and play ",
                "launch youtube and play ", "launch spotify and play ", "open spotify and play ",
                "open maps and navigate to ", "open google maps for ", "directions to ", "find route to ",
                "take me to ", "show route to ", "listen to ", "and play ", "navigate to "
            ]
            for pfx in prefixes_to_strip:
                if pfx in user_msg:
                    query = user_msg.split(pfx, 1)[1].strip()
                    break

            if not query and "play " in user_msg:
                q = user_msg.split("play ", 1)[1].strip()
                query = re.sub(r'(?i)\s+on\s+(youtube|spotify)', '', q).strip()

            params = {"app_name": app}
            if query:
                params["query"] = query

            return json.dumps({
                "type": "action",
                "thought": f"User requests launching mobile app '{app}'" + (f" with query '{query}'." if query else "."),
                "tool": "open_app",
                "parameters": params
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

        # Check for out-of-scope / irrelevant / physical / adversarial boundary queries
        irrelevant_keywords = [
            "wash", "iron", "cook", "clean", "tie", "repair", "paint", "polish", "fold",
            "deliver", "serve", "stitch", "bake", "fry", "dust", "sweep", "mop",
            "teleport", "time travel", "levitate", "beam", "transport", "materialize",
            "hack", "malware", "steal", "bypass security", "crack", "power grid", "intercept",
            "what color is my", "what am i thinking", "how many hairs", "who will i marry",
            "where are my lost", "what did i eat", "what will be my exact salary",
            "skibidi", "asdfghjkl", "qwertyuiop", "colorless green ideas", "flim flam", "beep boop",
            "blah blah", "foobar baz", "potato tomato helicopter"
        ]
        if any(k in user_msg for k in irrelevant_keywords):
            return json.dumps({
                "type": "message",
                "thought": "Query is out-of-scope, physically impossible, or adversarial. Gracefully declining.",
                "content": "I am an offline mobile assistant running strictly on your smartphone. I cannot perform physical tasks, external security penetrations, or unobservable personal deductions. I can help schedule reminders, manage notes and expense ledgers, search local files, and launch device apps."
            })

        # Check for factual knowledge queries (e.g. capitals, science, coding, math)
        if "capital of " in user_msg:
            country = user_msg.split("capital of ", 1)[1].replace("?", "").strip()
            capitals = {
                "france": "Paris", "japan": "Tokyo", "germany": "Berlin", "india": "New Delhi",
                "usa": "Washington, D.C.", "uk": "London", "canada": "Ottawa", "australia": "Canberra",
                "italy": "Rome", "spain": "Madrid", "russia": "Moscow", "china": "Beijing",
                "brazil": "Brasília", "south korea": "Seoul", "norway": "Oslo", "sweden": "Stockholm",
                "egypt": "Cairo", "turkey": "Ankara", "netherlands": "Amsterdam", "switzerland": "Bern",
                "mexico": "Mexico City", "argentina": "Buenos Aires", "saudi arabia": "Riyadh",
                "indonesia": "Jakarta", "uae": "Abu Dhabi", "south africa": "Pretoria", "new zealand": "Wellington",
                "singapore": "Singapore", "thailand": "Bangkok", "vietnam": "Hanoi", "portugal": "Lisbon",
                "greece": "Athens", "poland": "Warsaw", "austria": "Vienna", "belgium": "Brussels",
                "denmark": "Copenhagen", "finland": "Helsinki", "ireland": "Dublin", "kenya": "Nairobi"
            }
            cap = capitals.get(country, "the capital city")
            return json.dumps({
                "type": "message",
                "thought": f"Answering factual geography query for {country}.",
                "content": f"The capital of {country.title()} is {cap}."
            })

        # Science queries
        science_map = {
            "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (~3 × 10⁸ m/s).",
            "chemical formula of water": "The chemical formula of water is H2O.",
            "formula of carbon dioxide": "The chemical formula of carbon dioxide is CO2.",
            "powerhouse of the cell": "The mitochondria is the powerhouse of the cell, generating ATP.",
            "chromosomes do humans have": "Humans typically have 46 chromosomes (23 pairs).",
            "gravity on earth": "The acceleration due to gravity on Earth is approximately 9.8 m/s².",
            "boiling point of water": "The boiling point of water at standard pressure is 100°C (212°F).",
            "absolute zero": "Absolute zero is -273.15°C (0 Kelvin).",
            "photosynthesis": "Plants use photosynthesis to convert sunlight, carbon dioxide, and water into glucose and oxygen.",
            "plants use to make food": "Plants use photosynthesis to convert sunlight, carbon dioxide, and water into glucose and oxygen.",
            "largest planet": "Jupiter is the largest planet in our solar system.",
            "atomic number of carbon": "The atomic number of carbon is 6.",
            "cellular respiration": "Humans inhale oxygen (O2) for cellular respiration.",
            "hardest natural mineral": "Diamond is the hardest known natural mineral.",
            "chemical symbol for gold": "The chemical symbol for gold is Au (from Latin Aurum).",
            "electrical resistance": "The SI unit of electrical resistance is the ohm (Ω)."
        }
        for k, v in science_map.items():
            if k in user_msg:
                return json.dumps({
                    "type": "message",
                    "thought": f"Answering science query on {k}.",
                    "content": v
                })

        # Coding queries
        coding_map = {
            "binary search": "Binary search runs in O(log n) time complexity by repeatedly halving the search interval.",
            "quicksort": "Quicksort has an average time complexity of O(n log n), with worst case O(n²).",
            "first in first out": "A Queue follows the First-In, First-Out (FIFO) principle.",
            "last in first out": "A Stack follows the Last-In, First-Out (LIFO) principle.",
            "linear search": "Linear search has a worst-case time complexity of O(n).",
            "hash table": "Hash tables have an average lookup and insertion time of O(1).",
            "404": "HTTP 404 indicates 'Not Found' - the server cannot find the requested resource.",
            "not found": "HTTP 404 indicates 'Not Found' - the server cannot find the requested resource.",
            "200": "HTTP 200 indicates 'OK' - the request succeeded.",
            "represents success": "HTTP 200 indicates 'OK' - the request succeeded.",
            "git checkout -b": "The command 'git checkout -b <branch>' creates a new branch and immediately switches to it.",
            "creates a new branch": "The command 'git checkout -b <branch>' creates a new branch and immediately switches to it.",
            "rest api": "REST stands for Representational State Transfer, an architectural style for networked hypermedia applications.",
            "dijkstra": "Dijkstra's algorithm finds the shortest path between nodes in a graph with non-negative edge weights.",
            "shortest paths": "Dijkstra's algorithm finds the shortest path between nodes in a graph with non-negative edge weights.",
            "weighted graphs": "Dijkstra's algorithm finds the shortest path between nodes in a graph with non-negative edge weights.",
            "merge sort": "Merge sort requires O(n) auxiliary space complexity for temporary sub-arrays."
        }
        for k, v in coding_map.items():
            if k in user_msg:
                return json.dumps({
                    "type": "message",
                    "thought": f"Answering computer science query on {k}.",
                    "content": v
                })

        # Math multiplication queries
        math_m = re.search(r'(\d+)\s*\*\s*(\d+)', user_msg)
        if math_m:
            a = int(math_m.group(1))
            b = int(math_m.group(2))
            prod = a * b
            return json.dumps({
                "type": "message",
                "thought": f"Calculating product {a} * {b} = {prod}.",
                "content": f"{a} * {b} = {prod}."
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
