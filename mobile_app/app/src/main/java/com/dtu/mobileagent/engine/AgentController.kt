package com.dtu.mobileagent.engine

import android.content.Context
import android.util.Log
import com.dtu.mobileagent.data.local.AppDatabase
import com.dtu.mobileagent.data.local.entities.ChatMessageEntity
import com.dtu.mobileagent.data.local.entities.NoteEntity
import com.dtu.mobileagent.data.local.entities.ReminderEntity
import com.dtu.mobileagent.data.local.entities.AuditLogEntity
import com.dtu.mobileagent.data.local.entities.UserMemoryEntity
import com.dtu.mobileagent.data.local.entities.DocumentEntity
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class AgentUiState(
    val isProcessing: Boolean = false,
    val isModelLoaded: Boolean = false,
    val isVoiceOutputEnabled: Boolean = true,
    val isHybridRoutingEnabled: Boolean = true,
    val currentTaskCategory: String = "Hybrid Auto-Router",
    val pendingConfirmation: PendingAction? = null,
    val activeModel: ModelSpec = ModelGalleryManager.CATALOG[0],
    val currentTokensPerSec: Double = 22.4,
    val currentMemoryMb: Double = 1629.0
)

data class PendingAction(
    val messageId: Long,
    val toolName: String,
    val parameters: JsonObject,
    val description: String
)

class AgentController(
    private val context: Context,
    private val database: AppDatabase
) {
    private val TAG = "AgentController"
    private val gson = Gson()
    private val llamaBridge = LlamaBridge(context)
    private val toolExecutor = ToolExecutor(context, database)
    private val documentIndexer = DocumentIndexer(context, database)
    val hardwareProfiler = HardwareProfiler(context)
    private var speechCallback: ((String) -> Unit)? = null

    private val _uiState = MutableStateFlow(AgentUiState())
    val uiState: StateFlow<AgentUiState> = _uiState.asStateFlow()

    fun setSpeechOutputCallback(callback: (String) -> Unit) {
        speechCallback = callback
    }

    fun toggleVoiceOutput() {
        val nextState = !_uiState.value.isVoiceOutputEnabled
        _uiState.value = _uiState.value.copy(isVoiceOutputEnabled = nextState)
    }

    fun setHybridRoutingEnabled(enabled: Boolean) {
        _uiState.value = _uiState.value.copy(isHybridRoutingEnabled = enabled)
    }

    fun routeModelForTask(text: String): Pair<ModelSpec, String> {
        val lower = text.lowercase().trim()

        // 1. Tool dispatch / App launch / Fast Alarms -> Qwen2.5-0.5B (41.2 tok/s, 392 MB)
        val isFastTool = lower.startsWith("open ") || lower.startsWith("launch ") ||
                lower.startsWith("play ") || lower.contains("alarm") ||
                lower.startsWith("remind ") || lower.contains("reminder") ||
                lower.contains("camera") || lower.contains("settings") ||
                lower.contains("calculator") || lower == "time" || lower == "date"

        if (isFastTool) {
            val spec = ModelGalleryManager.CATALOG.find { it.id == "qwen2.5-0.5b" }
                ?: ModelGalleryManager.CATALOG[0]
            return Pair(spec, "Fast Tool Dispatch (41.2 tok/s • 392 MB)")
        }

        // 2. Structured Agent Operations (WhatsApp, SMS, Split Bill, Hisaab, Calendar, Documents) -> Llama-3.2-1B (28.6 tok/s, 804 MB)
        val isStructuredAgent = lower.contains("whatsapp") || lower.contains("message") ||
                lower.contains("sms") || lower.contains("text ") || lower.contains("split") ||
                lower.contains("hisaab") || lower.contains("kharcha") || lower.contains("ledger") ||
                lower.contains("calendar") || lower.contains("meeting") || lower.contains("schedule") ||
                lower.contains("files") || lower.contains("documents") || lower.contains("note")

        if (isStructuredAgent) {
            val spec = ModelGalleryManager.CATALOG.find { it.id == "llama-3.2-1b" }
                ?: ModelGalleryManager.CATALOG[0]
            return Pair(spec, "Structured Agent Operations (28.6 tok/s • 804 MB)")
        }

        // 3. Multi-Step Symbolic Math & Code Generation -> Llama-3.2-3B (18.2 tok/s, 1925 MB)
        val isComplexMathOrCode = lower.contains("code") || lower.contains("python") ||
                lower.contains("kotlin") || lower.contains("algorithm") || lower.contains("solve") ||
                lower.contains("derivative") || lower.contains("integral") || lower.contains("calculate") ||
                lower.contains("complexity") || lower.contains("function")

        if (isComplexMathOrCode) {
            val spec = ModelGalleryManager.CATALOG.find { it.id == "llama-3.2-3b" }
                ?: ModelGalleryManager.CATALOG[0]
            return Pair(spec, "Multi-Step Symbolic Reasoning (18.2 tok/s • 1925 MB)")
        }

        // 4. Knowledge, Science, Concepts, Academic Project -> Gemma-2-2B (22.4 tok/s, 1629 MB)
        val spec = ModelGalleryManager.CATALOG.find { it.id == "gemma-2-2b" }
            ?: ModelGalleryManager.CATALOG[0]
        return Pair(spec, "Semantic Knowledge & Reasoning (22.4 tok/s • 1629 MB)")
    }

    val messages: Flow<List<ChatMessageEntity>> = database.chatMessageDao().getAllMessages()
    val allNotes: Flow<List<NoteEntity>> = database.noteDao().getAllNotes()
    val activeReminders: Flow<List<ReminderEntity>> = database.reminderDao().getActiveReminders()
    val allMemories: Flow<List<UserMemoryEntity>> = database.userMemoryDao().getAllMemoriesFlow()
    val allDocuments: Flow<List<DocumentEntity>> = database.documentDao().getAllDocumentsFlow()
    val recentAuditLogs: Flow<List<AuditLogEntity>> = database.auditLogDao().getRecentLogs(50)
    val telemetry: StateFlow<HardwareTelemetry> = hardwareProfiler.telemetry

    fun refreshTelemetry() {
        hardwareProfiler.refreshTelemetry(_uiState.value.currentTokensPerSec)
    }

    suspend fun deleteNote(id: Long) {
        database.noteDao().deleteNote(id)
    }

    suspend fun markReminderCompleted(id: Long) {
        database.reminderDao().markCompleted(id)
    }

    suspend fun deleteMemory(keyTag: String) {
        database.userMemoryDao().deleteMemoryByKey(keyTag)
    }

    suspend fun deleteMemoryById(id: Long) {
        database.userMemoryDao().deleteMemoryById(id)
    }

    suspend fun initialize() {
        _uiState.value = _uiState.value.copy(isProcessing = true)
        val initialSpec = ModelGalleryManager.CATALOG[0]
        val success = llamaBridge.initialize(initialSpec)
        val metrics = llamaBridge.getMetrics()
        hardwareProfiler.refreshTelemetry(metrics.tokensPerSecond)
        _uiState.value = _uiState.value.copy(
            isProcessing = false,
            isModelLoaded = success,
            activeModel = initialSpec,
            currentTokensPerSec = metrics.tokensPerSecond,
            currentMemoryMb = metrics.memoryFootprintMb
        )

        // Asynchronously scan and index device files in background
        CoroutineScope(Dispatchers.IO).launch {
            try {
                documentIndexer.scanDeviceDocuments()
            } catch (e: Exception) {
                Log.w(TAG, "Background document indexing failed", e)
            }
        }
    }

    suspend fun switchModel(spec: ModelSpec) {
        _uiState.value = _uiState.value.copy(isProcessing = true)
        val success = llamaBridge.initialize(spec)
        val metrics = llamaBridge.getMetrics()
        hardwareProfiler.refreshTelemetry(metrics.tokensPerSecond)
        _uiState.value = _uiState.value.copy(
            isProcessing = false,
            isModelLoaded = success,
            activeModel = spec,
            currentTokensPerSec = metrics.tokensPerSecond,
            currentMemoryMb = metrics.memoryFootprintMb
        )
    }

    suspend fun switchModel(modelFilename: String) {
        val spec = ModelGalleryManager.CATALOG.find { it.filename == modelFilename || it.id == modelFilename }
            ?: ModelGalleryManager.CATALOG[0]
        switchModel(spec)
    }

    suspend fun handleUserMessage(text: String) {
        if (text.isBlank()) return

        // 1. Record user message in local Room DB
        database.chatMessageDao().insertMessage(
            ChatMessageEntity(sender = "user", content = text)
        )

        _uiState.value = _uiState.value.copy(isProcessing = true)

        try {
            // Task-Conditioned Hybrid Model Dynamic Routing
            if (_uiState.value.isHybridRoutingEnabled) {
                val (routedModel, category) = routeModelForTask(text)
                if (routedModel.id != _uiState.value.activeModel.id) {
                    llamaBridge.initialize(routedModel)
                    val metrics = llamaBridge.getMetrics()
                    hardwareProfiler.refreshTelemetry(metrics.tokensPerSecond)
                    _uiState.value = _uiState.value.copy(
                        activeModel = routedModel,
                        currentTaskCategory = category,
                        currentTokensPerSec = metrics.tokensPerSecond,
                        currentMemoryMb = metrics.memoryFootprintMb
                    )
                } else {
                    _uiState.value = _uiState.value.copy(currentTaskCategory = category)
                }
            }

            val currentModelShortName = _uiState.value.activeModel.displayName.replace("-Instruct", "").replace("-Chat", "")
            val hybridThoughtPrefix = if (_uiState.value.isHybridRoutingEnabled) "[$currentModelShortName • ${_uiState.value.currentTokensPerSec} tok/s] " else ""

            // Check direct memory storage, recall, or document indexing intents
            val memoryOrDocResp = tryHandleMemoryOrDocIntent(text)
            if (memoryOrDocResp != null) {
                database.chatMessageDao().insertMessage(
                    ChatMessageEntity(
                        sender = "agent",
                        content = memoryOrDocResp,
                        thought = hybridThoughtPrefix
                    )
                )
                if (_uiState.value.isVoiceOutputEnabled) {
                    speechCallback?.invoke(memoryOrDocResp)
                }
                _uiState.value = _uiState.value.copy(isProcessing = false)
                return
            }

            // 2. Build local ReAct prompt with system context & history budget
            val prompt = buildPrompt(text)

            // 3. Native inference with GBNF grammar constraint
            val rawOutput = llamaBridge.generate(prompt)
            val parsed = parseModelOutput(rawOutput)

            when (parsed) {
                is ModelResponse.Action -> {
                    val toolName = parsed.tool
                    val params = parsed.parameters

                    val isConfirmationReq = ToolRegistry.isConfirmationRequired(toolName)
                    if (isConfirmationReq) {
                        val msgId = database.chatMessageDao().insertMessage(
                            ChatMessageEntity(
                                sender = "agent",
                                content = "I need your authorization to execute: $toolName",
                                thought = hybridThoughtPrefix + parsed.thought,
                                toolName = toolName,
                                toolParamsJson = gson.toJson(params),
                                requiresConfirmation = true
                            )
                        )
                        _uiState.value = _uiState.value.copy(
                            isProcessing = false,
                            pendingConfirmation = PendingAction(
                                messageId = msgId,
                                toolName = toolName,
                                parameters = params,
                                description = "Send message to ${params.get("recipient")?.asString ?: "contact"}"
                            )
                        )
                    } else {
                        // Autonomous execution of safe local tool
                        val execution = toolExecutor.execute(toolName, params)
                        database.chatMessageDao().insertMessage(
                            ChatMessageEntity(
                                sender = "agent",
                                content = execution.resultSummary,
                                thought = hybridThoughtPrefix + parsed.thought,
                                toolName = toolName,
                                toolParamsJson = gson.toJson(params)
                            )
                        )
                        if (_uiState.value.isVoiceOutputEnabled) {
                            speechCallback?.invoke(execution.resultSummary)
                        }
                        _uiState.value = _uiState.value.copy(isProcessing = false)
                    }
                }
                is ModelResponse.Message -> {
                    database.chatMessageDao().insertMessage(
                        ChatMessageEntity(
                            sender = "agent",
                            content = parsed.content,
                            thought = hybridThoughtPrefix + parsed.thought
                        )
                    )
                    if (_uiState.value.isVoiceOutputEnabled) {
                        speechCallback?.invoke(parsed.content)
                    }
                    _uiState.value = _uiState.value.copy(isProcessing = false)
                }
                is ModelResponse.ParseError -> {
                    val fallbackContent = if (parsed.rawOutput.isNotBlank() && !parsed.rawOutput.startsWith("{")) {
                        parsed.rawOutput
                    } else {
                        "I am here to assist you with tasks like scheduling reminders, checking your calendar, taking notes, or answering questions. How can I help?"
                    }
                    database.chatMessageDao().insertMessage(
                        ChatMessageEntity(
                            sender = "agent",
                            content = fallbackContent,
                            thought = ""
                        )
                    )
                    if (_uiState.value.isVoiceOutputEnabled) {
                        speechCallback?.invoke(fallbackContent)
                    }
                    _uiState.value = _uiState.value.copy(isProcessing = false)
                }
            }

            // Update performance stats
            val metrics = llamaBridge.getMetrics()
            hardwareProfiler.refreshTelemetry(metrics.tokensPerSecond)
            _uiState.value = _uiState.value.copy(
                currentTokensPerSec = metrics.tokensPerSecond,
                currentMemoryMb = metrics.memoryFootprintMb
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error in agent loop", e)
            val friendlyFallback = "I'm running locally on your device! You can ask me questions, create reminders, manage notes, or check your schedule."
            database.chatMessageDao().insertMessage(
                ChatMessageEntity(
                    sender = "agent",
                    content = friendlyFallback
                )
            )
            if (_uiState.value.isVoiceOutputEnabled) {
                speechCallback?.invoke(friendlyFallback)
            }
            _uiState.value = _uiState.value.copy(isProcessing = false)
        }
    }

    suspend fun confirmPendingAction(approve: Boolean) {
        val pending = _uiState.value.pendingConfirmation ?: return
        _uiState.value = _uiState.value.copy(pendingConfirmation = null, isProcessing = true)

        if (approve) {
            database.chatMessageDao().markConfirmed(pending.messageId)
            val result = toolExecutor.executeConfirmedAction(pending.toolName, pending.parameters)
            database.chatMessageDao().insertMessage(
                ChatMessageEntity(
                    sender = "agent",
                    content = "Action Approved & Executed: ${result.resultSummary}"
                )
            )
        } else {
            database.chatMessageDao().insertMessage(
                ChatMessageEntity(
                    sender = "agent",
                    content = "Action Cancelled: Execution rejected by user."
                )
            )
        }

        _uiState.value = _uiState.value.copy(isProcessing = false)
    }

    suspend fun clearHistory() {
        database.chatMessageDao().clearHistory()
    }

    suspend fun rescanDocuments(): Int {
        return documentIndexer.scanDeviceDocuments()
    }

    suspend fun clearMemories() {
        database.userMemoryDao().clearMemories()
    }

    private suspend fun buildPrompt(userText: String): String {
        val recent = database.chatMessageDao().getRecentMessages(6).reversed()
        val historyStr = recent.joinToString("\n") { "${it.sender.uppercase()}: ${it.content}" }

        // Local RAG: inject top persistent memories and matching indexed documents
        val memories = try {
            database.userMemoryDao().getAllMemories().take(6)
        } catch (e: Exception) {
            emptyList()
        }
        val memorySection = if (memories.isNotEmpty()) {
            "\n[User Persistent Knowledge & Memory]:\n" + memories.joinToString("\n") { "• ${it.fact}" }
        } else ""

        val relevantDocs = try {
            database.documentDao().searchDocuments(userText, 2)
        } catch (e: Exception) {
            emptyList()
        }
        val docSection = if (relevantDocs.isNotEmpty()) {
            "\n[Relevant Indexed Documents]:\n" + relevantDocs.joinToString("\n") { "• [${it.filename}]: ${it.summary.ifEmpty { it.content.take(120) }}" }
        } else ""

        return """
            <|system|>
            You are an offline mobile AI assistant running locally on an Android device for DTU CSE GP159.
            You must output JSON matching the system GBNF schema.
            Permitted tools: create_reminder, search_calendar, manage_notes, open_app, search_local_docs, read_document_summary, draft_message.
            $memorySection
            $docSection
            <|history|>
            $historyStr
            <|user|>
            $userText
            <|assistant|>
        """.trimIndent()
    }

    private suspend fun tryHandleMemoryOrDocIntent(text: String): String? {
        val lower = text.lowercase().trim()

        // 1. Explicit Memory Storage: "remember that ...", "remember: ...", "remember ..."
        val remMatch = Regex("""(?i)^(?:remember\s+that|remember\s*:|remember|note\s+that|keep\s+in\s+mind\s+that)\s+(.+)""").find(text.trim())
        if (remMatch != null) {
            val rawFact = remMatch.groupValues[1].trim()
            val tag = if (rawFact.length > 20) rawFact.take(20) else rawFact
            database.userMemoryDao().insertMemory(
                UserMemoryEntity(
                    category = "personal",
                    keyTag = tag,
                    fact = rawFact
                )
            )
            return "I have saved that to your private on-device memory:\n• $rawFact"
        }

        // 2. Personal Fact Stated: "my [attribute] is [value]"
        val myMatch = Regex("""(?i)^my\s+([a-zA-Z0-9_\s]{2,30}?)\s+is\s+(.+)""").find(text.trim())
        if (myMatch != null) {
            val key = myMatch.groupValues[1].trim()
            val value = myMatch.groupValues[2].trim()
            val lowerKey = key.lowercase()
            if (lowerKey !in listOf("day", "schedule", "plan", "notes", "calendar", "reminder", "reminders", "files", "documents")) {
                val fact = "${key.replaceFirstChar { it.uppercase() }} is $value"
                database.userMemoryDao().insertMemory(
                    UserMemoryEntity(
                        category = "personal",
                        keyTag = key,
                        fact = fact
                    )
                )
                return "Saved to persistent memory:\n• $fact"
            }
        }

        // 3. User Memory Recall Inquiry
        val isQueryMemory = lower.contains("what do you remember") || lower.contains("what do you know about me") ||
                lower.contains("what's in your memory") || lower.contains("what is in your memory") ||
                lower.contains("show memory") || lower.contains("list memories") ||
                lower.contains("my memories") || lower == "memory"
        if (isQueryMemory) {
            val memories = database.userMemoryDao().getAllMemories()
            return if (memories.isEmpty()) {
                "I don't have any saved facts in your local memory yet. You can tell me 'Remember that my car number is DL4C1234' or 'My roll number is 2K20/CO/123'."
            } else {
                "Here is what I have stored in your private on-device memory (${memories.size} facts):\n" +
                        memories.joinToString("\n") { "• ${it.fact}" }
            }
        }

        // 4. Clear or delete memories
        if (lower == "clear memory" || lower == "clear memories" || lower == "delete all memories" || lower == "forget all") {
            database.userMemoryDao().clearMemories()
            return "All persistent user memories have been cleared from local on-device storage."
        }
        if (lower.startsWith("forget ") || lower.startsWith("delete memory ")) {
            val targetKey = text.replace(Regex("""(?i)^(?:forget\s+(?:that\s+)?|delete\s+memory\s+)"""), "").trim()
            database.userMemoryDao().deleteMemoryByKey(targetKey)
            return "Removed memory matching '$targetKey' from local storage."
        }

        // 5. Academic & DTU BTP Project Knowledge Recall
        if (lower.contains("who is my guide") || lower.contains("who is my supervisor") ||
            lower.contains("who is the guide") || lower.contains("project guide") || lower == "my guide") {
            return "Your project supervisor is Ms. Gull Kaur (DTU Computer Science and Engineering Department)."
        }
        if (lower.contains("group id") || lower.contains("what is my group") || lower.contains("which group")) {
            return "Your BTP Group ID is GP159. Team members: Faqre Alam, Ekansh Agarwal, Guneet Toppo, and Bhawishya."
        }
        if (lower.contains("team members") || lower.contains("who is in my team") ||
            lower.contains("who are my teammates") || lower.contains("group members")) {
            return "Your BTP team members for Group GP159 are Faqre Alam, Ekansh Agarwal, Guneet Toppo, and Bhawishya."
        }
        if (lower.contains("project topic") || lower.contains("what is my project") || lower.contains("my btp project")) {
            return "Your major project topic is: Privacy-First Offline AI Mobile Agent Using LLM Quantization (DTU CSE 2026)."
        }

        // 6. Device Document & File Indexing Inquiries
        if (lower.contains("what files") || lower.contains("list files") || lower.contains("list my files") ||
            lower.contains("list documents") || lower.contains("list my documents") || lower.contains("show files") ||
            lower.contains("show my files") || lower.contains("show documents") || lower == "files" || lower == "documents") {
            val docs = database.documentDao().getRecentDocuments(15)
            return if (docs.isEmpty()) {
                "No indexed documents found on your device yet. You can say 'Rescan files' or place documents in your device's Documents folder."
            } else {
                "Indexed on-device documents (${docs.size}):\n" +
                        docs.joinToString("\n") { "• ${it.filename}: ${it.summary.ifEmpty { it.content.take(80) }}" }
            }
        }
        if (lower.contains("rescan files") || lower.contains("scan files") || lower.contains("scan documents") ||
            lower.contains("index files") || lower.contains("update file index")) {
            val count = documentIndexer.scanDeviceDocuments()
            return "Scanned and indexed $count document(s) into local Room database for offline context retrieval."
        }
        if (lower.startsWith("search files for ") || lower.startsWith("search documents for ") || lower.startsWith("find in files ")) {
            val q = text.replace(Regex("""(?i)^(?:search\s+files\s+for|search\s+documents\s+for|find\s+in\s+files)\s+"""), "").trim()
            val docs = database.documentDao().searchDocuments(q, 5)
            return if (docs.isEmpty()) {
                "No indexed documents found matching '$q'."
            } else {
                "Found ${docs.size} matching document(s) for '$q':\n" +
                        docs.joinToString("\n") { "• ${it.filename}:\n  ${it.summary.ifEmpty { it.content.take(150) }}" }
            }
        }

        return null
    }

    private fun parseModelOutput(raw: String): ModelResponse {
        return try {
            val json = JsonParser.parseString(raw.trim()).asJsonObject
            val type = if (json.has("type")) json.get("type").asString else "message"
            val thought = if (json.has("thought")) json.get("thought").asString else ""

            if (type == "action") {
                val tool = json.get("tool").asString
                val params = json.getAsJsonObject("parameters")
                ModelResponse.Action(thought, tool, params)
            } else {
                val content = if (json.has("content")) json.get("content").asString else raw
                ModelResponse.Message(thought, content)
            }
        } catch (e: Exception) {
            ModelResponse.ParseError(raw, e.message ?: "Unknown parse error")
        }
    }
}
