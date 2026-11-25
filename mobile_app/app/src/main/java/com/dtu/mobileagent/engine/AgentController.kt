package com.dtu.mobileagent.engine

import android.content.Context
import android.util.Log
import com.dtu.mobileagent.data.local.AppDatabase
import com.dtu.mobileagent.data.local.entities.ChatMessageEntity
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class AgentUiState(
    val isProcessing: Boolean = false,
    val isModelLoaded: Boolean = false,
    val isVoiceOutputEnabled: Boolean = false,
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

    val messages: Flow<List<ChatMessageEntity>> = database.chatMessageDao().getAllMessages()

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
                                thought = parsed.thought,
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
                                thought = parsed.thought,
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
                            thought = parsed.thought
                        )
                    )
                    if (_uiState.value.isVoiceOutputEnabled) {
                        speechCallback?.invoke(parsed.content)
                    }
                    _uiState.value = _uiState.value.copy(isProcessing = false)
                }
                is ModelResponse.ParseError -> {
                    database.chatMessageDao().insertMessage(
                        ChatMessageEntity(
                            sender = "agent",
                            content = "I could not formulate an execution plan for that request. Output: ${parsed.rawOutput}",
                            thought = "Parsing failure"
                        )
                    )
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
            database.chatMessageDao().insertMessage(
                ChatMessageEntity(
                    sender = "agent",
                    content = "Error processing request locally: ${e.message}"
                )
            )
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

    private suspend fun buildPrompt(userText: String): String {
        val recent = database.chatMessageDao().getRecentMessages(6).reversed()
        val historyStr = recent.joinToString("\n") { "${it.sender.uppercase()}: ${it.content}" }

        return """
            <|system|>
            You are an offline mobile AI assistant running locally on an Android device for DTU CSE GP159.
            You must output JSON matching the system GBNF schema.
            Permitted tools: create_reminder, search_calendar, manage_notes, open_app, search_local_docs, read_document_summary, draft_message.
            <|history|>
            $historyStr
            <|user|>
            $userText
            <|assistant|>
        """.trimIndent()
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
