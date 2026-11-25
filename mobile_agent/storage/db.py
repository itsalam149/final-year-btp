"""
Local SQLite Storage Engine (Room Database Mirror for Android).
Provides offline persistence for Notes, Reminders, Calendar Events,
Local Documents, Conversation History, and the Security Audit Log.
"""

import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os


class LocalStorage:
    """Manages SQLite storage for the on-device mobile agent."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        """Create database tables matching Android Room entity specifications."""
        with self.conn:
            # 1. Notes Table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # 2. Reminders Table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    due_time_iso TEXT NOT NULL,
                    priority TEXT DEFAULT 'normal',
                    is_completed INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
            """)

            # 3. Calendar Events Table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS calendar_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    start_time_iso TEXT NOT NULL,
                    end_time_iso TEXT NOT NULL,
                    location TEXT
                );
            """)

            # 4. Local Documents Table (Mock Scoped Storage)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    title TEXT,
                    content_text TEXT NOT NULL,
                    tags TEXT
                );
            """)

            # 5. Conversation History Table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)

            # 6. Security & Privacy Audit Log
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    user_confirmed INTEGER DEFAULT 0
                );
            """)

    # ---------------------------------------------------------------------
    # Notes Operations
    # ---------------------------------------------------------------------

    def create_note(self, title: str, content: str = "") -> int:
        now = datetime.now().isoformat()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO notes (title, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (title, content, now, now)
            )
            return cur.lastrowid

    def get_note_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM notes WHERE title LIKE ? LIMIT 1", (f"%{title}%",))
        row = cur.fetchone()
        return dict(row) if row else None

    def search_notes(self, query: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM notes WHERE title LIKE ? OR content LIKE ? ORDER BY updated_at DESC",
            (f"%{query}%", f"%{query}%")
        )
        return [dict(row) for row in cur.fetchall()]

    def list_notes(self, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM notes ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]

    # ---------------------------------------------------------------------
    # Reminder Operations
    # ---------------------------------------------------------------------

    def create_reminder(self, title: str, due_time_iso: str, priority: str = "normal") -> int:
        now = datetime.now().isoformat()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO reminders (title, due_time_iso, priority, created_at) VALUES (?, ?, ?, ?)",
                (title, due_time_iso, priority, now)
            )
            return cur.lastrowid

    def list_reminders(self, include_completed: bool = False) -> List[Dict[str, Any]]:
        query = "SELECT * FROM reminders"
        if not include_completed:
            query += " WHERE is_completed = 0"
        query += " ORDER BY due_time_iso ASC"
        cur = self.conn.execute(query)
        return [dict(row) for row in cur.fetchall()]

    # ---------------------------------------------------------------------
    # Calendar Operations
    # ---------------------------------------------------------------------

    def add_calendar_event(self, title: str, start_time_iso: str, end_time_iso: str, location: str = "") -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO calendar_events (title, start_time_iso, end_time_iso, location) VALUES (?, ?, ?, ?)",
                (title, start_time_iso, end_time_iso, location)
            )
            return cur.lastrowid

    def search_calendar(self, date_iso: str, query: Optional[str] = None) -> List[Dict[str, Any]]:
        # Match events where start_time_iso starts with date_iso (YYYY-MM-DD)
        sql = "SELECT * FROM calendar_events WHERE start_time_iso LIKE ?"
        params = [f"{date_iso}%"]
        if query:
            sql += " AND (title LIKE ? OR location LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        sql += " ORDER BY start_time_iso ASC"
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    # ---------------------------------------------------------------------
    # Document Operations
    # ---------------------------------------------------------------------

    def add_document(self, filename: str, title: str, content_text: str, tags: str = "") -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR REPLACE INTO documents (filename, title, content_text, tags) VALUES (?, ?, ?, ?)",
                (filename, title, content_text, tags)
            )
            return cur.lastrowid

    def search_documents(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT id, filename, title, tags, substr(content_text, 1, 300) as snippet
            FROM documents
            WHERE filename LIKE ? OR title LIKE ? OR content_text LIKE ? OR tags LIKE ?
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", max_results)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_document_by_name(self, filename: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM documents WHERE filename LIKE ? LIMIT 1", (f"%{filename}%",))
        row = cur.fetchone()
        return dict(row) if row else None

    # ---------------------------------------------------------------------
    # Conversation History & Memory
    # ---------------------------------------------------------------------

    def append_message(self, session_id: str, role: str, content: str) -> None:
        now = datetime.now().isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT INTO conversation_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now)
            )

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT role, content, timestamp FROM conversation_history WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit)
        )
        return [dict(row) for row in cur.fetchall()]

    # ---------------------------------------------------------------------
    # Audit Logging
    # ---------------------------------------------------------------------

    def log_audit(self, tool_name: str, parameters: Dict[str, Any], status: str, user_confirmed: bool) -> None:
        now = datetime.now().isoformat()
        params_str = json.dumps(parameters)
        with self.conn:
            self.conn.execute(
                "INSERT INTO audit_log (timestamp, tool_name, parameters_json, status, user_confirmed) VALUES (?, ?, ?, ?, ?)",
                (now, tool_name, params_str, status, 1 if user_confirmed else 0)
            )

    def get_audit_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
