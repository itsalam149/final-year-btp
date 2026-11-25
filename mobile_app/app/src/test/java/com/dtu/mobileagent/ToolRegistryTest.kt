package com.dtu.mobileagent

import com.dtu.mobileagent.engine.ToolRegistry
import com.dtu.mobileagent.engine.ToolSafetyLevel
import org.junit.Assert.*
import org.junit.Test

class ToolRegistryTest {

    @Test
    fun testCoreToolsAreRegistered() {
        val tools = ToolRegistry.TOOLS
        assertTrue("Must support create_reminder", tools.containsKey("create_reminder"))
        assertTrue("Must support search_calendar", tools.containsKey("search_calendar"))
        assertTrue("Must support manage_notes", tools.containsKey("manage_notes"))
        assertTrue("Must support open_app", tools.containsKey("open_app"))
        assertTrue("Must support draft_message", tools.containsKey("draft_message"))
    }

    @Test
    fun testSafetyLevels() {
        // Safe autonomous operations
        assertEquals(ToolSafetyLevel.SAFE_AUTONOMOUS, ToolRegistry.TOOLS["create_reminder"]?.safetyLevel)
        assertEquals(ToolSafetyLevel.SAFE_AUTONOMOUS, ToolRegistry.TOOLS["search_calendar"]?.safetyLevel)
        assertEquals(ToolSafetyLevel.SAFE_AUTONOMOUS, ToolRegistry.TOOLS["manage_notes"]?.safetyLevel)
        assertEquals(ToolSafetyLevel.SAFE_AUTONOMOUS, ToolRegistry.TOOLS["open_app"]?.safetyLevel)

        // Dangerous / external actions requiring confirmation
        assertEquals(ToolSafetyLevel.REQUIRES_CONFIRMATION, ToolRegistry.TOOLS["draft_message"]?.safetyLevel)
        assertTrue(ToolRegistry.isConfirmationRequired("draft_message"))
        assertFalse(ToolRegistry.isConfirmationRequired("create_reminder"))
        assertFalse(ToolRegistry.isConfirmationRequired("open_app"))
    }

    @Test
    fun testRequiredParameters() {
        assertEquals(listOf("title", "due_time_iso"), ToolRegistry.TOOLS["create_reminder"]?.requiredParameters)
        assertEquals(listOf("date_iso"), ToolRegistry.TOOLS["search_calendar"]?.requiredParameters)
        assertEquals(listOf("action"), ToolRegistry.TOOLS["manage_notes"]?.requiredParameters)
        assertEquals(listOf("app_name"), ToolRegistry.TOOLS["open_app"]?.requiredParameters)
        assertEquals(listOf("recipient", "message_body"), ToolRegistry.TOOLS["draft_message"]?.requiredParameters)
    }
}
