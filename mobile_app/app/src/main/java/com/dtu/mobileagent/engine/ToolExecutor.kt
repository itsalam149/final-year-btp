package com.dtu.mobileagent.engine

import android.content.Context
import android.content.Intent
import android.net.Uri
import com.dtu.mobileagent.data.local.AppDatabase
import com.dtu.mobileagent.data.local.entities.AuditLogEntity
import com.dtu.mobileagent.data.local.entities.CalendarEventEntity
import com.dtu.mobileagent.data.local.entities.NoteEntity
import com.dtu.mobileagent.data.local.entities.ReminderEntity
import com.google.gson.Gson
import com.google.gson.JsonObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ToolExecutor(
    private val context: Context,
    private val database: AppDatabase
) {
    private val gson = Gson()

    suspend fun execute(toolName: String, parameters: JsonObject): ExecutionResult =
        withContext(Dispatchers.IO) {
            val toolDef = ToolRegistry.TOOLS[toolName]
                ?: return@withContext ExecutionResult(
                    toolName = toolName,
                    success = false,
                    resultSummary = "Unknown tool: '$toolName'. Permitted tools are: ${ToolRegistry.TOOLS.keys}"
                )

            // Validate required parameters
            for (param in toolDef.requiredParameters) {
                if (!parameters.has(param) || parameters.get(param).isJsonNull) {
                    val errorMsg = "Missing required parameter '$param' for tool '$toolName'"
                    logAudit(toolName, parameters, "FAILED", errorMsg)
                    return@withContext ExecutionResult(
                        toolName = toolName,
                        success = false,
                        resultSummary = errorMsg
                    )
                }
            }

            // Check if confirmation is required before actual execution
            if (toolDef.safetyLevel == ToolSafetyLevel.REQUIRES_CONFIRMATION) {
                logAudit(toolName, parameters, "CONFIRMATION_REQUIRED", "Awaiting user approval before dispatch.")
                return@withContext ExecutionResult(
                    toolName = toolName,
                    success = true,
                    resultSummary = "Action requires confirmation: Send message to '${parameters.get("recipient").asString}'",
                    data = parameters,
                    requiresConfirmation = true
                )
            }

            // Execute safe autonomous tools
            return@withContext try {
                when (toolName) {
                    "create_reminder" -> executeCreateReminder(parameters)
                    "search_calendar" -> executeSearchCalendar(parameters)
                    "manage_notes" -> executeManageNotes(parameters)
                    "search_local_docs" -> executeSearchLocalDocs(parameters)
                    "read_document_summary" -> executeReadDocSummary(parameters)
                    else -> ExecutionResult(toolName, false, "Unhandled tool $toolName")
                }
            } catch (e: Exception) {
                val error = "Execution error in $toolName: ${e.message}"
                logAudit(toolName, parameters, "FAILED", error)
                ExecutionResult(toolName, false, error)
            }
        }

    suspend fun executeConfirmedAction(toolName: String, parameters: JsonObject): ExecutionResult =
        withContext(Dispatchers.IO) {
            if (toolName == "draft_message") {
                val recipient = parameters.get("recipient")?.asString ?: ""
                val body = parameters.get("message_body")?.asString ?: ""
                val platform = if (parameters.has("platform")) parameters.get("platform").asString else "sms"

                // Create Android Intent for system SMS dispatcher
                val intent = Intent(Intent.ACTION_VIEW).apply {
                    data = Uri.parse("sms:$recipient")
                    putExtra("sms_body", body)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }

                try {
                    context.startActivity(intent)
                    val summary = "Dispatched $platform message to $recipient: \"$body\""
                    logAudit(toolName, parameters, "EXECUTED", summary)
                    ExecutionResult(toolName, true, summary)
                } catch (e: Exception) {
                    val err = "Could not open messaging intent: ${e.message}"
                    logAudit(toolName, parameters, "FAILED", err)
                    ExecutionResult(toolName, false, err)
                }
            } else {
                ExecutionResult(toolName, false, "Unknown confirmation action: $toolName")
            }
        }

    private suspend fun executeCreateReminder(params: JsonObject): ExecutionResult {
        val title = params.get("title").asString
        val dueTime = params.get("due_time_iso").asString
        val priority = if (params.has("priority")) params.get("priority").asString else "normal"

        val id = database.reminderDao().insertReminder(
            ReminderEntity(
                title = title,
                dueTimeIso = dueTime,
                priority = priority
            )
        )
        val summary = "Created reminder #$id: '$title' due at $dueTime (Priority: $priority)"
        logAudit("create_reminder", params, "EXECUTED", summary)
        return ExecutionResult("create_reminder", true, summary)
    }

    private suspend fun executeSearchCalendar(params: JsonObject): ExecutionResult {
        val dateIso = params.get("date_iso").asString
        val query = if (params.has("query")) params.get("query").asString else ""

        val events = if (query.isNotBlank()) {
            database.calendarDao().searchEvents(dateIso, query)
        } else {
            database.calendarDao().getEventsForDate(dateIso)
        }

        val summary = if (events.isEmpty()) {
            "No calendar events found for $dateIso."
        } else {
            "Found ${events.size} event(s) on $dateIso:\n" +
                    events.joinToString("\n") { "• [${it.startTime} - ${it.endTime}] ${it.title} (${it.location})" }
        }

        logAudit("search_calendar", params, "EXECUTED", "Returned ${events.size} events")
        return ExecutionResult("search_calendar", true, summary)
    }

    private suspend fun executeManageNotes(params: JsonObject): ExecutionResult {
        val action = params.get("action").asString
        return when (action) {
            "create" -> {
                val title = if (params.has("title")) params.get("title").asString else "Quick Note"
                val content = if (params.has("content")) params.get("content").asString else ""
                val id = database.noteDao().insertNote(
                    NoteEntity(title = title, content = content)
                )
                val msg = "Created note #$id: '$title'"
                logAudit("manage_notes", params, "EXECUTED", msg)
                ExecutionResult("manage_notes", true, msg)
            }
            "search" -> {
                val query = if (params.has("query")) params.get("query").asString else ""
                val notes = database.noteDao().searchNotes(query)
                val msg = if (notes.isEmpty()) "No notes found matching '$query'."
                else "Found ${notes.size} note(s):\n" + notes.joinToString("\n") { "• ${it.title}: ${it.content}" }
                logAudit("manage_notes", params, "EXECUTED", "Found ${notes.size} matches")
                ExecutionResult("manage_notes", true, msg)
            }
            "read" -> {
                val title = if (params.has("title")) params.get("title").asString else ""
                val note = database.noteDao().findByTitle(title)
                val msg = if (note == null) "Note titled '$title' not found." else "Note '${note.title}': ${note.content}"
                logAudit("manage_notes", params, "EXECUTED", "Read note '$title'")
                ExecutionResult("manage_notes", note != null, msg)
            }
            else -> {
                val notes = database.noteDao().searchNotes("")
                val msg = "Local notes (${notes.size}):\n" + notes.joinToString("\n") { "• ${it.title}" }
                ExecutionResult("manage_notes", true, msg)
            }
        }
    }

    private suspend fun executeSearchLocalDocs(params: JsonObject): ExecutionResult {
        val query = params.get("query").asString
        val limit = if (params.has("max_results")) params.get("max_results").asInt else 3
        val docs = database.documentDao().searchDocuments(query, limit)

        val summary = if (docs.isEmpty()) {
            "No local documents found matching '$query'."
        } else {
            "Found ${docs.size} matching document(s):\n" +
                    docs.joinToString("\n") { "• ${it.filename}: ${it.summary.ifEmpty { it.content.take(120) }}" }
        }

        logAudit("search_local_docs", params, "EXECUTED", "Found ${docs.size} docs")
        return ExecutionResult("search_local_docs", true, summary)
    }

    private suspend fun executeReadDocSummary(params: JsonObject): ExecutionResult {
        val filename = params.get("filename").asString
        val doc = database.documentDao().findByFilename(filename)
        val msg = if (doc == null) {
            "Document '$filename' not found in local index."
        } else {
            "Summary of '${doc.filename}':\n${doc.summary.ifEmpty { doc.content.take(200) }}"
        }
        logAudit("read_document_summary", params, "EXECUTED", "Read summary for $filename")
        return ExecutionResult("read_document_summary", doc != null, msg)
    }

    private suspend fun logAudit(tool: String, params: JsonObject, status: String, details: String) {
        database.auditLogDao().logAction(
            AuditLogEntity(
                toolName = tool,
                parametersJson = gson.toJson(params),
                status = status,
                executionDetails = details
            )
        )
    }
}
