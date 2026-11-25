"""
Execution handlers for the 6 Core Mobile Tools.
Interacts with the LocalStorage engine and simulates Android OS system services.
"""

from typing import Any, Dict, List, Optional
from mobile_agent.storage.db import LocalStorage


class MobileToolHandlers:
    """Encapsulates tool handlers bound to a LocalStorage database instance."""

    def __init__(self, storage: LocalStorage):
        self.storage = storage

    def create_reminder(self, title: str, due_time_iso: str, priority: str = "normal") -> Dict[str, Any]:
        """Schedules a local reminder via AlarmManager / Reminders table."""
        reminder_id = self.storage.create_reminder(title, due_time_iso, priority)
        return {
            "reminder_id": reminder_id,
            "title": title,
            "due_time": due_time_iso,
            "priority": priority,
            "status": "SCHEDULED",
            "message": f"Reminder '{title}' scheduled for {due_time_iso}."
        }

    def search_calendar(self, date_iso: str, query: Optional[str] = None) -> Dict[str, Any]:
        """Searches CalendarProvider for scheduled events on target date."""
        events = self.storage.search_calendar(date_iso, query)
        return {
            "target_date": date_iso,
            "event_count": len(events),
            "events": events,
            "has_events": len(events) > 0,
            "message": f"Found {len(events)} event(s) for {date_iso}."
        }

    def manage_notes(
        self,
        action: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handles note creation, reading, searching, or listing in Room DB."""
        if action == "create":
            if not title:
                return {"status": "ERROR", "message": "Action 'create' requires a note 'title'."}
            note_id = self.storage.create_note(title, content or "")
            return {
                "note_id": note_id,
                "title": title,
                "action": "create",
                "status": "CREATED",
                "message": f"Note '{title}' saved successfully to local secure storage."
            }

        elif action == "read":
            if not title:
                return {"status": "ERROR", "message": "Action 'read' requires a note 'title'."}
            note = self.storage.get_note_by_title(title)
            if not note:
                return {"status": "NOT_FOUND", "message": f"No note matching title '{title}'."}
            return {"status": "FOUND", "note": note}

        elif action == "search":
            search_query = query or title or ""
            notes = self.storage.search_notes(search_query)
            return {"status": "FOUND", "count": len(notes), "notes": notes}

        elif action == "list":
            notes = self.storage.list_notes()
            return {"status": "LISTED", "count": len(notes), "notes": notes}

        return {"status": "ERROR", "message": f"Unrecognized action '{action}' for notes."}

    def search_local_docs(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Searches permitted documents in local storage (PDFs, TXT)."""
        docs = self.storage.search_documents(query, max_results)
        return {
            "query": query,
            "match_count": len(docs),
            "documents": docs,
            "message": f"Retrieved {len(docs)} local document(s) matching '{query}'."
        }

    def read_document_summary(self, filename: str, focus_topic: Optional[str] = None) -> Dict[str, Any]:
        """Extracts text from a local document and provides contextual summary."""
        doc = self.storage.get_document_by_name(filename)
        if not doc:
            return {
                "status": "FILE_NOT_FOUND",
                "filename": filename,
                "message": f"Document '{filename}' not found in authorized local storage."
            }

        full_text = doc["content_text"]
        # Basic chunk/summary extraction logic (offline simulated RAG)
        lines = full_text.split("\n")
        relevant_lines = []
        if focus_topic:
            topic_lower = focus_topic.lower()
            relevant_lines = [l for l in lines if any(w in l.lower() for w in topic_lower.split())]

        if not relevant_lines:
            relevant_lines = lines[:15]  # first 15 lines if no specific topic match

        summary_excerpt = "\n".join(relevant_lines[:8])

        return {
            "status": "SUCCESS",
            "filename": filename,
            "title": doc.get("title", filename),
            "focus_topic": focus_topic,
            "excerpt": summary_excerpt,
            "message": f"Extracted excerpt from {filename} regarding '{focus_topic or 'general'}': {summary_excerpt[:100]}..."
        }

    def draft_message(self, recipient: str, message_body: str, platform: str = "sms") -> Dict[str, Any]:
        """Prepares a communication draft. Strictly requires user confirmation."""
        return {
            "status": "DRAFT_READY",
            "recipient": recipient,
            "message_body": message_body,
            "platform": platform,
            "user_action_required": "Please review and tap Send in your messaging app.",
            "message": f"Prepared {platform.upper()} draft for {recipient}: '{message_body}'"
        }
