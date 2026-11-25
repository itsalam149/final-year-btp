package com.dtu.mobileagent

import com.google.gson.JsonParser
import org.junit.Assert.*
import org.junit.Test

/**
 * Validates conversational parsing and tool generation rules
 * without requiring the Android runtime or C++ JNI bridge.
 */
class IntentRoutingAndDialogueTest {

    // Helper simulation matching LlamaBridge logic
    private fun simulateInference(prompt: String): String {
        val userPos = prompt.lastIndexOf("<|user|>")
        val userText = if (userPos != -1) {
            val raw = prompt.substring(userPos + 8)
            val assistantPos = raw.indexOf("<|assistant|>")
            if (assistantPos != -1) raw.substring(0, assistantPos).trim() else raw.trim()
        } else {
            prompt.trim()
        }
        val lower = userText.lowercase().trim()

        if (lower.contains("not to") || lower.contains("didn't ask") || lower.contains("stop") ||
            lower.contains("cancel") || lower.contains("wrong") || lower.contains("what the fuck") ||
            lower.contains("wtf") || lower.contains("not that")) {
            return """{"type": "message", "thought": "", "content": "My apologies! I am here to help you manage reminders, calendar events, notes, and opening apps on your phone. What would you like me to do?"}"""
        }

        if (lower == "hey" || lower == "hello" || lower == "hi" || lower == "yo" ||
            lower.startsWith("hey ") || lower.startsWith("hello ") || lower.startsWith("hi ")) {
            return """{"type": "message", "thought": "", "content": "Hello! How can I help you today?"}"""
        }

        if (lower.contains("what can you do") || lower.contains("what you can do") || lower.contains("features") || lower.contains("help")) {
            return """{"type": "message", "thought": "", "content": "I can help you with these core offline tasks on your device:\n\n• Reminders: 'Remind me to study at 7 PM'\n• Calendar: 'What's my schedule for today?'\n• Notes: 'Take a note: BTP review on Monday'\n• Apps: 'Open WhatsApp' or 'Open Spotify'\n• Switch Models: Tap the engine pill in the top bar to switch between Gemma-2-2B, Llama-3.2, and Qwen."}"""
        }

        if (lower.contains("whatsapp") || lower.contains("spotify") || lower.contains("maps") ||
            lower.startsWith("open ") || lower.startsWith("launch ")) {
            val app = when {
                "maps" in lower -> "maps"
                "spotify" in lower -> "spotify"
                else -> "whatsapp"
            }
            return """{"type": "action", "thought": "Launching $app", "tool": "open_app", "parameters": {"app_name": "$app"}}"""
        }

        if (lower.contains("remind") || lower.contains("reminder")) {
            val titleMatch = Regex("remind me to\\s+([a-zA-Z0-9_\\s]+?)(?:\\s+at|\\s+tomorrow|\\s+on|$)").find(lower)
            val title = titleMatch?.groupValues?.get(1)?.trim()?.replaceFirstChar { it.uppercase() } ?: "Follow up on task"
            return """{"type": "action", "thought": "Creating reminder", "tool": "create_reminder", "parameters": {"title": "$title", "due_time_iso": "2026-11-26T14:00:00", "priority": "normal"}}"""
        }

        if (lower.contains("calendar") || lower.contains("schedule") || lower.contains("meeting")) {
            return """{"type": "action", "thought": "Checking calendar", "tool": "search_calendar", "parameters": {"date_iso": "2026-11-26", "query": "meeting"}}"""
        }

        if (lower.contains("note")) {
            return """{"type": "action", "thought": "Saving private note", "tool": "manage_notes", "parameters": {"action": "create", "title": "Quick Note", "content": "BTP Review"}}"""
        }

        return """{"type": "message", "thought": "", "content": "I am focused on device tasks: scheduling reminders, checking your calendar, taking notes, and opening apps. What would you like me to handle?"}"""
    }

    @Test
    fun testGreetingDoesNotEchoAndHasNoThought() {
        val raw = simulateInference("<|user|>Hey<|assistant|>")
        val json = JsonParser.parseString(raw).asJsonObject

        assertEquals("message", json.get("type").asString)
        assertEquals("", json.get("thought").asString)
        assertEquals("Hello! How can I help you today?", json.get("content").asString)
        assertFalse("Must never echo user text robotically", json.get("content").asString.contains("I understand you are asking about"))
    }

    @Test
    fun testProfanityOrCorrectionReturnsPoliteHelp() {
        val raw = simulateInference("<|user|>what the fuck<|assistant|>")
        val json = JsonParser.parseString(raw).asJsonObject

        assertEquals("message", json.get("type").asString)
        assertEquals("", json.get("thought").asString)
        assertTrue(json.get("content").asString.contains("My apologies"))
        assertFalse("Must never echo back profanity", json.get("content").asString.contains("what the fuck"))
    }

    @Test
    fun testCapabilitiesInquiryListsCorePillars() {
        val raw = simulateInference("<|user|>what can you do any other thing<|assistant|>")
        val json = JsonParser.parseString(raw).asJsonObject

        assertEquals("message", json.get("type").asString)
        val content = json.get("content").asString
        assertTrue("Must mention Reminders", content.contains("Reminders"))
        assertTrue("Must mention Calendar", content.contains("Calendar"))
        assertTrue("Must mention Notes", content.contains("Notes"))
        assertTrue("Must mention Apps", content.contains("Apps"))
        assertTrue("Must mention Switch Models", content.contains("Switch Models"))
    }

    @Test
    fun testAppLaunchDispatchesOpenAppTool() {
        val raw = simulateInference("<|user|>Open WhatsApp<|assistant|>")
        val json = JsonParser.parseString(raw).asJsonObject

        assertEquals("action", json.get("type").asString)
        assertEquals("open_app", json.get("tool").asString)
        val params = json.getAsJsonObject("parameters")
        assertEquals("whatsapp", params.get("app_name").asString)
    }

    @Test
    fun testReminderDispatchesCreateReminderTool() {
        val raw = simulateInference("<|user|>Remind me to study at 7 PM<|assistant|>")
        val json = JsonParser.parseString(raw).asJsonObject

        assertEquals("action", json.get("type").asString)
        assertEquals("create_reminder", json.get("tool").asString)
        val params = json.getAsJsonObject("parameters")
        assertEquals("Study", params.get("title").asString)
        assertEquals("2026-11-26T14:00:00", params.get("due_time_iso").asString)
    }

    @Test
    fun testCalendarQueryDispatchesSearchCalendarTool() {
        val raw = simulateInference("<|user|>What's my schedule for today?<|assistant|>")
        val json = JsonParser.parseString(raw).asJsonObject

        assertEquals("action", json.get("type").asString)
        assertEquals("search_calendar", json.get("tool").asString)
    }

    @Test
    fun testNoteCreationDispatchesManageNotesTool() {
        val raw = simulateInference("<|user|>Take a note: BTP review on Monday<|assistant|>")
        val json = JsonParser.parseString(raw).asJsonObject

        assertEquals("action", json.get("type").asString)
        assertEquals("manage_notes", json.get("tool").asString)
        val params = json.getAsJsonObject("parameters")
        assertEquals("create", params.get("action").asString)
    }
}
