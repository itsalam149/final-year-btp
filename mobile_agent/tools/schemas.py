"""
Tool schemas and definitions for the Mobile Agent.
Defines parameters, JSON-schema constraints, and security sensitivity levels.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SecurityLevel(str, Enum):
    """Classification of tool execution risk."""
    SAFE = "SAFE"                              # Read-only or safe local writes (notes, reminders)
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"  # External communication or consequential changes


@dataclass
class ToolDefinition:
    name: str
    description: str
    security_level: SecurityLevel
    parameters_schema: Dict[str, Any]
    required_permissions: List[str] = field(default_factory=list)


# -------------------------------------------------------------------------
# Core Mobile Tool Definitions (6 Core Tools)
# -------------------------------------------------------------------------

CREATE_REMINDER_TOOL = ToolDefinition(
    name="create_reminder",
    description="Schedule a local alarm or notification reminder on the user's mobile device.",
    security_level=SecurityLevel.SAFE,
    required_permissions=["android.permission.SCHEDULE_EXACT_ALARM"],
    parameters_schema={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Brief description of the reminder."
            },
            "due_time_iso": {
                "type": "string",
                "description": "ISO-8601 formatted datetime string (e.g., '2026-11-27T09:00:00')."
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "Priority level of the reminder."
            }
        },
        "required": ["title", "due_time_iso"]
    }
)

SEARCH_CALENDAR_TOOL = ToolDefinition(
    name="search_calendar",
    description="Search local calendar events to find schedules, appointments, or free time slots.",
    security_level=SecurityLevel.SAFE,
    required_permissions=["android.permission.READ_CALENDAR"],
    parameters_schema={
        "type": "object",
        "properties": {
            "date_iso": {
                "type": "string",
                "description": "Target date formatted as YYYY-MM-DD (e.g., '2026-11-27')."
            },
            "query": {
                "type": "string",
                "description": "Optional keyword to filter events (e.g., 'meeting', 'lecture')."
            }
        },
        "required": ["date_iso"]
    }
)

MANAGE_NOTES_TOOL = ToolDefinition(
    name="manage_notes",
    description="Create, read, search, or list personal notes in local secure storage.",
    security_level=SecurityLevel.SAFE,
    required_permissions=[],
    parameters_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "read", "list", "search"],
                "description": "Action to perform on notes."
            },
            "title": {
                "type": "string",
                "description": "Title of the note (required for 'create' and 'read')."
            },
            "content": {
                "type": "string",
                "description": "Body content of the note (used when creating a note)."
            },
            "query": {
                "type": "string",
                "description": "Keyword search query (used when action is 'search')."
            }
        },
        "required": ["action"]
    }
)

SEARCH_LOCAL_DOCS_TOOL = ToolDefinition(
    name="search_local_docs",
    description="Search permitted local documents (PDF, TXT) on the smartphone storage.",
    security_level=SecurityLevel.SAFE,
    required_permissions=["android.permission.READ_MEDIA_DOCUMENTS"],
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keyword or phrase to match against document names or tags."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of documents to return (default: 3)."
            }
        },
        "required": ["query"]
    }
)

READ_DOCUMENT_SUMMARY_TOOL = ToolDefinition(
    name="read_document_summary",
    description="Read and extract a contextual summary of a specific local PDF or text document.",
    security_level=SecurityLevel.SAFE,
    required_permissions=["android.permission.READ_MEDIA_DOCUMENTS"],
    parameters_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name or path of the local document."
            },
            "focus_topic": {
                "type": "string",
                "description": "Specific topic or question to answer from the document."
            }
        },
        "required": ["filename"]
    }
)

DRAFT_MESSAGE_TOOL = ToolDefinition(
    name="draft_message",
    description="Prepare a draft message (SMS or WhatsApp) with a pre-filled recipient and text. Requires user confirmation before opening.",
    security_level=SecurityLevel.REQUIRES_CONFIRMATION,
    required_permissions=[],
    parameters_schema={
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Name or phone number of the recipient."
            },
            "message_body": {
                "type": "string",
                "description": "The exact body text of the message."
            },
            "platform": {
                "type": "string",
                "enum": ["sms", "whatsapp"],
                "description": "Target messaging platform."
            }
        },
        "required": ["recipient", "message_body"]
    }
)

OPEN_APP_TOOL = ToolDefinition(
    name="open_app",
    description="Launch an installed application on the mobile device (e.g., 'spotify', 'maps', 'youtube', 'whatsapp', 'settings').",
    security_level=SecurityLevel.SAFE,
    required_permissions=[],
    parameters_schema={
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "Common name or package identifier of the app to open."
            },
            "query": {
                "type": "string",
                "description": "Optional search term, music track, or destination address."
            }
        },
        "required": ["app_name"]
    }
)

AUTOMATE_UI_TOOL = ToolDefinition(
    name="automate_ui",
    description="Interact with active app UI elements on screen via Android Accessibility Service (click buttons, enter text, inspect screen).",
    security_level=SecurityLevel.REQUIRES_CONFIRMATION,
    required_permissions=["android.permission.BIND_ACCESSIBILITY_SERVICE"],
    parameters_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["click", "type", "read_screen", "navigate"],
                "description": "UI automation action to perform."
            },
            "target": {
                "type": "string",
                "description": "Visible text or label of the button or field to interact with."
            },
            "input_text": {
                "type": "string",
                "description": "Text to enter if action is 'type'."
            }
        },
        "required": ["action"]
    }
)

ALL_CORE_TOOLS = [
    CREATE_REMINDER_TOOL,
    SEARCH_CALENDAR_TOOL,
    MANAGE_NOTES_TOOL,
    SEARCH_LOCAL_DOCS_TOOL,
    READ_DOCUMENT_SUMMARY_TOOL,
    DRAFT_MESSAGE_TOOL,
    OPEN_APP_TOOL,
    AUTOMATE_UI_TOOL
]
