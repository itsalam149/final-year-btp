package com.dtu.mobileagent.engine

import android.Manifest
import android.app.SearchManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.ContactsContract
import android.provider.MediaStore
import androidx.core.content.ContextCompat
import com.dtu.mobileagent.data.local.AppDatabase
import com.dtu.mobileagent.data.local.entities.AuditLogEntity
import com.dtu.mobileagent.data.local.entities.CalendarEventEntity
import com.dtu.mobileagent.data.local.entities.NoteEntity
import com.dtu.mobileagent.data.local.entities.ReminderEntity
import com.dtu.mobileagent.service.AgentAccessibilityService
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
            val isReadScreen = toolName == "automate_ui" && parameters.has("action") && parameters.get("action").asString == "read_screen"
            if (toolDef.safetyLevel == ToolSafetyLevel.REQUIRES_CONFIRMATION && !isReadScreen) {
                logAudit(toolName, parameters, "CONFIRMATION_REQUIRED", "Awaiting user approval before dispatch.")
                val desc = if (toolName == "draft_message") {
                    "Send message to '${parameters.get("recipient")?.asString ?: "contact"}'"
                } else {
                    "UI Action: ${parameters.get("action")?.asString ?: "automate"} on '${parameters.get("target")?.asString ?: "screen"}'"
                }
                return@withContext ExecutionResult(
                    toolName = toolName,
                    success = true,
                    resultSummary = "Action requires confirmation: $desc",
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
                    "open_app" -> executeOpenApp(parameters)
                    "draft_message" -> executeDraftMessage(parameters)
                    "automate_ui" -> executeAutomateUi(parameters)
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
                executeDraftMessage(parameters)
            } else if (toolName == "automate_ui") {
                executeAutomateUi(parameters)
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

                // Dispatch to the phone's actual Notes application
                dispatchToDeviceNotesApp(title, content)

                val msg = if (content.isNotBlank() && (content.contains("₹") || content.contains("•") || content.contains("\n"))) {
                    "Recorded in your private ledger and opened your Notes app:\n\n$content"
                } else {
                    "Created note '$title' and opened your Notes app."
                }
                logAudit("manage_notes", params, "EXECUTED", msg)
                ExecutionResult("manage_notes", true, msg)
            }
            "search" -> {
                val query = if (params.has("query")) params.get("query").asString.trim() else ""
                val notes = if (query.isNotBlank()) database.noteDao().searchNotes(query) else database.noteDao().searchNotes("")
                val msg = if (notes.isEmpty()) {
                    "You don't have any saved notes yet. You can say 'Take a note: study AI' to create one."
                } else {
                    "Found ${notes.size} note(s):\n" + notes.joinToString("\n") { "• ${it.title}: ${it.content}" }
                }
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

    private suspend fun executeOpenApp(params: JsonObject): ExecutionResult {
        val appName = params.get("app_name").asString.trim()
        val query = if (params.has("query")) params.get("query").asString.trim() else ""
        val lowerApp = appName.lowercase()

        // 1. YouTube with direct search/play intent
        if (lowerApp == "youtube" && query.isNotBlank()) {
            var launched = false
            // Direct playback intent
            try {
                val playIntent = Intent(MediaStore.INTENT_ACTION_MEDIA_PLAY_FROM_SEARCH).apply {
                    putExtra(SearchManager.QUERY, query)
                    setPackage("com.google.android.youtube")
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(playIntent)
                launched = true
            } catch (e: Exception) {
                launched = false
            }

            if (!launched) {
                try {
                    val searchIntent = Intent(Intent.ACTION_SEARCH).apply {
                        setPackage("com.google.android.youtube")
                        putExtra(SearchManager.QUERY, query)
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    context.startActivity(searchIntent)
                    launched = true
                } catch (e: Exception) {
                    launched = false
                }
            }

            if (!launched) {
                val uri = Uri.parse("https://www.youtube.com/results?search_query=" + Uri.encode(query))
                val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                    setPackage("com.google.android.youtube")
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                try {
                    context.startActivity(intent)
                } catch (e: Exception) {
                    intent.setPackage(null)
                    context.startActivity(intent)
                }
            }
            val summary = if (query.isNotBlank()) "Opening YouTube to play '$query'..." else "Opening YouTube..."
            logAudit("open_app", params, "EXECUTED", summary)
            return ExecutionResult("open_app", true, summary)
        }

        // 2. Spotify with search/playback intent
        if (lowerApp == "spotify" && query.isNotBlank()) {
            val uri = Uri.parse("spotify:search:" + Uri.encode(query))
            val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            try {
                context.startActivity(intent)
            } catch (e: Exception) {
                val webUri = Uri.parse("https://open.spotify.com/search/" + Uri.encode(query))
                context.startActivity(Intent(Intent.ACTION_VIEW, webUri).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) })
            }
            val summary = if (query.isNotBlank()) "Opening Spotify to play '$query'..." else "Opening Spotify..."
            logAudit("open_app", params, "EXECUTED", summary)
            return ExecutionResult("open_app", true, summary)
        }

        // 3. Google Maps with navigation/search intent
        if ((lowerApp == "maps" || lowerApp == "google maps") && query.isNotBlank()) {
            val uri = Uri.parse("geo:0,0?q=" + Uri.encode(query))
            val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                setPackage("com.google.android.apps.maps")
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            try {
                context.startActivity(intent)
            } catch (e: Exception) {
                intent.setPackage(null)
                context.startActivity(intent)
            }
            val summary = "Navigating to '$query' on Google Maps..."
            logAudit("open_app", params, "EXECUTED", summary)
            return ExecutionResult("open_app", true, summary)
        }

        val packageName = when (lowerApp) {
            "whatsapp" -> "com.whatsapp"
            "maps", "google maps" -> "com.google.android.apps.maps"
            "spotify" -> "com.spotify.music"
            "youtube" -> "com.google.android.youtube"
            "gmail" -> "com.google.android.gm"
            "clock", "alarm" -> "com.google.android.deskclock"
            "calculator" -> "com.google.android.calculator"
            "calendar" -> "com.google.android.calendar"
            "settings" -> "com.android.settings"
            "camera" -> "com.android.camera2"
            else -> if (appName.contains(".")) appName else ""
        }

        val launchIntent = if (packageName.isNotBlank()) {
            context.packageManager.getLaunchIntentForPackage(packageName)
        } else {
            null
        }

        return if (launchIntent != null) {
            launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(launchIntent)
            val friendlyName = when (lowerApp) {
                "whatsapp" -> "WhatsApp"
                "youtube" -> "YouTube"
                "spotify" -> "Spotify"
                "maps", "google maps" -> "Google Maps"
                "camera" -> "Camera"
                "settings" -> "Settings"
                "calculator" -> "Calculator"
                "clock", "alarm" -> "Clock"
                "calendar" -> "Calendar"
                "gmail" -> "Gmail"
                else -> appName.replaceFirstChar { it.uppercase() }
            }
            val summary = "Opening $friendlyName..."
            logAudit("open_app", params, "EXECUTED", summary)
            ExecutionResult("open_app", true, summary)
        } else {
            val err = "Could not find installed application matching '$appName'."
            logAudit("open_app", params, "FAILED", err)
            ExecutionResult("open_app", false, err)
        }
    }

    private fun dispatchToDeviceNotesApp(title: String, content: String) {
        var dispatched = false

        // 1. Android standard CREATE_NOTE intent
        try {
            val createNoteIntent = Intent("android.intent.action.CREATE_NOTE").apply {
                putExtra(Intent.EXTRA_TITLE, title)
                putExtra(Intent.EXTRA_TEXT, content)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            if (createNoteIntent.resolveActivity(context.packageManager) != null) {
                context.startActivity(createNoteIntent)
                dispatched = true
            }
        } catch (e: Exception) {
            dispatched = false
        }

        // 2. Direct package intents for primary Android notes apps
        if (!dispatched) {
            val notePackages = listOf(
                "com.google.android.keep",           // Google Keep
                "com.miui.notes",                    // Xiaomi Notes
                "com.samsung.android.app.notes",     // Samsung Notes
                "com.coloros.note",                  // Oppo / Realme Notes
                "com.oneplus.note",                  // OnePlus Notes
                "com.vivo.notes"                     // Vivo / iQOO Notes
            )

            for (pkg in notePackages) {
                try {
                    val intent = Intent(Intent.ACTION_SEND).apply {
                        type = "text/plain"
                        setPackage(pkg)
                        putExtra(Intent.EXTRA_SUBJECT, title)
                        putExtra(Intent.EXTRA_TEXT, "$title\n\n$content")
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    if (intent.resolveActivity(context.packageManager) != null) {
                        context.startActivity(intent)
                        dispatched = true
                        break
                    }
                } catch (e: Exception) {
                    // Try next
                }
            }
        }

        // 3. Fallback to generic ACTION_SEND intent
        if (!dispatched) {
            try {
                val sendIntent = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_SUBJECT, title)
                    putExtra(Intent.EXTRA_TEXT, "$title\n\n$content")
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(sendIntent)
                dispatched = true
            } catch (e: Exception) {
                // Ignore
            }
        }
    }

    private suspend fun executeAutomateUi(params: JsonObject): ExecutionResult {
        val service = AgentAccessibilityService.instance
        if (service == null) {
            val err = "Accessibility Service is not enabled. Please enable 'Agent Accessibility Service' in Android Settings -> Accessibility."
            logAudit("automate_ui", params, "FAILED", err)
            return ExecutionResult("automate_ui", false, err)
        }

        val action = params.get("action").asString.lowercase()
        val target = if (params.has("target") && !params.get("target").isJsonNull) params.get("target").asString else ""
        val textInput = if (params.has("input_text") && !params.get("input_text").isJsonNull) {
            params.get("input_text").asString
        } else if (params.has("text_input") && !params.get("text_input").isJsonNull) {
            params.get("text_input").asString
        } else {
            ""
        }

        return when (action) {
            "click" -> {
                val ok = service.clickElementByText(target)
                val summary = if (ok) "Successfully clicked element matching '$target'" else "Failed to find clickable element matching '$target'"
                logAudit("automate_ui", params, if (ok) "EXECUTED" else "FAILED", summary)
                ExecutionResult("automate_ui", ok, summary)
            }
            "type" -> {
                val ok = service.enterTextInField(target, textInput)
                val summary = if (ok) "Entered '$textInput' into field matching '$target'" else "Could not locate editable field matching '$target'"
                logAudit("automate_ui", params, if (ok) "EXECUTED" else "FAILED", summary)
                ExecutionResult("automate_ui", ok, summary)
            }
            "read_screen" -> {
                val texts = service.dumpVisibleScreenText()
                val summary = if (texts.isEmpty()) {
                    "No readable elements on active screen."
                } else {
                    "Active screen elements (${texts.size}):\n" + texts.take(15).joinToString("\n") { "• $it" }
                }
                logAudit("automate_ui", params, "EXECUTED", "Read ${texts.size} screen elements")
                ExecutionResult("automate_ui", true, summary, data = params)
            }
            "navigate", "back", "home", "recents" -> {
                val navTarget = if (action == "navigate") (if (target.isNotBlank()) target else "back") else action
                val ok = service.performNavigation(navTarget)
                val summary = if (ok) "Performed global navigation: $navTarget" else "Navigation action failed: $navTarget"
                logAudit("automate_ui", params, if (ok) "EXECUTED" else "FAILED", summary)
                ExecutionResult("automate_ui", ok, summary)
            }
            else -> {
                val err = "Unsupported UI action '$action'. Permitted: click, type, read_screen, navigate."
                logAudit("automate_ui", params, "FAILED", err)
                ExecutionResult("automate_ui", false, err)
            }
        }
    }

    private fun findPhoneNumberByName(name: String): String? {
        val trimmed = name.trim()
        if (trimmed.isBlank()) return null
        if (trimmed.matches(Regex("""\+?[0-9]{7,15}"""))) {
            return trimmed
        }

        try {
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
                return null
            }
            val uri = ContactsContract.CommonDataKinds.Phone.CONTENT_URI
            val projection = arrayOf(
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Phone.NUMBER
            )
            val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
            val selectionArgs = arrayOf("%$trimmed%")
            context.contentResolver.query(uri, projection, selection, selectionArgs, null)?.use { cursor ->
                val numberIndex = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
                while (cursor.moveToNext()) {
                    if (numberIndex != -1) {
                        val number = cursor.getString(numberIndex)
                        if (!number.isNullOrBlank()) {
                            val clean = number.replace(Regex("[^0-9+]"), "")
                            if (clean.isNotBlank()) return clean
                        }
                    }
                }
            }
        } catch (e: Exception) {
            // Permission or cursor exception
        }
        return null
    }

    private fun formatPhoneForWhatsApp(raw: String): String {
        val clean = raw.replace(Regex("[^0-9]"), "")
        return when {
            clean.length == 10 -> "91$clean"
            clean.length == 11 && clean.startsWith("0") -> "91" + clean.substring(1)
            clean.length == 12 && clean.startsWith("91") -> clean
            else -> clean
        }
    }

    private suspend fun executeDraftMessage(params: JsonObject): ExecutionResult {
        val recipient = params.get("recipient")?.asString?.trim() ?: ""
        val body = params.get("message_body")?.asString?.trim() ?: ""
        val platform = if (params.has("platform")) params.get("platform").asString.lowercase() else "whatsapp"

        val phone = findPhoneNumberByName(recipient)

        if (platform == "whatsapp") {
            var sent = false
            if (!phone.isNullOrBlank()) {
                val formattedPhone = formatPhoneForWhatsApp(phone)
                try {
                    val uri = Uri.parse("https://api.whatsapp.com/send?phone=$formattedPhone&text=" + Uri.encode(body))
                    val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                        setPackage("com.whatsapp")
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    context.startActivity(intent)
                    sent = true
                } catch (e: Exception) {
                    try {
                        val uri = Uri.parse("https://api.whatsapp.com/send?phone=$formattedPhone&text=" + Uri.encode(body))
                        val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        }
                        context.startActivity(intent)
                        sent = true
                    } catch (e2: Exception) {
                        sent = false
                    }
                }
            }

            if (!sent) {
                // Open WhatsApp share/send with pre-filled message
                try {
                    val sendIntent = Intent(Intent.ACTION_SEND).apply {
                        type = "text/plain"
                        setPackage("com.whatsapp")
                        putExtra(Intent.EXTRA_TEXT, body)
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    context.startActivity(sendIntent)
                    sent = true
                } catch (e: Exception) {
                    val fallbackUri = Uri.parse("https://api.whatsapp.com/send?text=" + Uri.encode(body))
                    context.startActivity(Intent(Intent.ACTION_VIEW, fallbackUri).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) })
                    sent = true
                }
            }

            val summary = if (!phone.isNullOrBlank()) {
                "Opened WhatsApp chat with $recipient ($phone) with message: \"$body\""
            } else {
                "Opened WhatsApp with message: \"$body\" (select $recipient to send)"
            }
            logAudit("draft_message", params, "EXECUTED", summary)
            return ExecutionResult("draft_message", true, summary)
        } else {
            // SMS platform
            val targetNumber = phone ?: recipient
            val smsUri = if (targetNumber.isNotBlank()) Uri.parse("smsto:$targetNumber") else Uri.parse("sms:")
            val intent = Intent(Intent.ACTION_SENDTO, smsUri).apply {
                putExtra("sms_body", body)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            try {
                context.startActivity(intent)
                val summary = "Dispatched SMS to $recipient ($targetNumber): \"$body\""
                logAudit("draft_message", params, "EXECUTED", summary)
                return ExecutionResult("draft_message", true, summary)
            } catch (e: Exception) {
                val err = "Could not open SMS messaging intent: ${e.message}"
                logAudit("draft_message", params, "FAILED", err)
                return ExecutionResult("draft_message", false, err)
            }
        }
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
