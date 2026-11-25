package com.dtu.mobileagent.engine

import com.google.gson.JsonObject

enum class ToolSafetyLevel {
    SAFE_AUTONOMOUS,        // Safe local operations: read/write notes, local reminders, read calendar, search docs
    REQUIRES_CONFIRMATION   // External side effects: sending SMS/messages, making external calls, deleting files
}

data class ToolDefinition(
    val name: String,
    val description: String,
    val safetyLevel: ToolSafetyLevel,
    val requiredParameters: List<String>
)

object ToolRegistry {
    val TOOLS = mapOf(
        "create_reminder" to ToolDefinition(
            name = "create_reminder",
            description = "Schedules a local notification reminder on the phone.",
            safetyLevel = ToolSafetyLevel.SAFE_AUTONOMOUS,
            requiredParameters = listOf("title", "due_time_iso")
        ),
        "search_calendar" to ToolDefinition(
            name = "search_calendar",
            description = "Queries the local calendar for events on a specific date.",
            safetyLevel = ToolSafetyLevel.SAFE_AUTONOMOUS,
            requiredParameters = listOf("date_iso")
        ),
        "manage_notes" to ToolDefinition(
            name = "manage_notes",
            description = "Creates, reads, lists, or searches local notes in private storage.",
            safetyLevel = ToolSafetyLevel.SAFE_AUTONOMOUS,
            requiredParameters = listOf("action")
        ),
        "search_local_docs" to ToolDefinition(
            name = "search_local_docs",
            description = "Performs keyword search over indexed on-device documents.",
            safetyLevel = ToolSafetyLevel.SAFE_AUTONOMOUS,
            requiredParameters = listOf("query")
        ),
        "read_document_summary" to ToolDefinition(
            name = "read_document_summary",
            description = "Retrieves stored summary of a specific indexed local document.",
            safetyLevel = ToolSafetyLevel.SAFE_AUTONOMOUS,
            requiredParameters = listOf("filename")
        ),
        "draft_message" to ToolDefinition(
            name = "draft_message",
            description = "Prepares an SMS or messaging intent to send to a contact. REQUIRES EXPLICIT USER CONFIRMATION.",
            safetyLevel = ToolSafetyLevel.REQUIRES_CONFIRMATION,
            requiredParameters = listOf("recipient", "message_body")
        )
    )

    fun isConfirmationRequired(toolName: String): Boolean {
        return TOOLS[toolName]?.safetyLevel == ToolSafetyLevel.REQUIRES_CONFIRMATION
    }
}

// Model output representation
sealed class ModelResponse {
    data class Action(
        val thought: String,
        val tool: String,
        val parameters: JsonObject
    ) : ModelResponse()

    data class Message(
        val thought: String,
        val content: String
    ) : ModelResponse()

    data class ParseError(
        val rawOutput: String,
        val error: String
    ) : ModelResponse()
}

// Tool execution outcome
data class ExecutionResult(
    val toolName: String,
    val success: Boolean,
    val resultSummary: String,
    val data: JsonObject? = null,
    val requiresConfirmation: Boolean = false,
    val pendingActionId: Long? = null
)
