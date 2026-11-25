package com.dtu.mobileagent.engine

import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

data class ModelSpec(
    val id: String,
    val displayName: String,
    val filename: String,
    val sizeMb: Int,
    val ramMb: Int,
    val precision: String,
    val downloadUrl: String,
    val description: String
)

data class ModelDownloadProgress(
    val modelId: String,
    val isDownloading: Boolean = false,
    val progressPct: Float = 0.0f,
    val downloadedBytes: Long = 0L,
    val totalBytes: Long = 0L
)

/**
 * Edge Model Gallery Manager
 * Inspired by Google AI Edge Gallery and MLC LLM:
 * Allows users to browse, download, and hot-swap quantized GGUF models directly on-device.
 * Once downloaded, all reasoning and tool calling runs 100% offline.
 */
class ModelGalleryManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelGalleryManager"

        val CATALOG = listOf(
            ModelSpec(
                id = "llama-3.2-1b",
                displayName = "Llama-3.2-1B-Instruct",
                filename = "llama-3.2-1b-instruct-q4_k_m.gguf",
                sizeMb = 804,
                ramMb = 1040,
                precision = "GGUF Q4_K_M (4-bit)",
                downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                description = "Recommended for primary edge tasks. Excellent instruction following and grammar-guided tool calling."
            ),
            ModelSpec(
                id = "qwen2.5-0.5b",
                displayName = "Qwen2.5-0.5B-Instruct",
                filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                sizeMb = 392,
                ramMb = 580,
                precision = "GGUF Q4_K_M (4-bit)",
                downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                description = "Ultra-compact edge model. Sub-600MB RAM footprint, ideal for low-end Android smartphones."
            ),
            ModelSpec(
                id = "tinyllama-1.1b",
                displayName = "TinyLlama-1.1B-Chat",
                filename = "tinyllama-1.1b-chat-q4_k_m.gguf",
                sizeMb = 668,
                ramMb = 880,
                precision = "GGUF Q4_K_M (4-bit)",
                downloadUrl = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                description = "Fast conversational chat model trained on 3 trillion tokens. Smooth mobile dialogue."
            )
        )
    }

    private val modelsDir: File
        get() {
            val dir = File(context.filesDir, "models")
            if (!dir.exists()) dir.mkdirs()
            return dir
        }

    private val _activeModel = MutableStateFlow(CATALOG[0])
    val activeModel: StateFlow<ModelSpec> = _activeModel.asStateFlow()

    private val _downloadStates = MutableStateFlow<Map<String, ModelDownloadProgress>>(emptyMap())
    val downloadStates: StateFlow<Map<String, ModelDownloadProgress>> = _downloadStates.asStateFlow()

    fun isModelDownloaded(spec: ModelSpec): Boolean {
        val file = File(modelsDir, spec.filename)
        return file.exists() && file.length() > 1024 * 1024 // At least 1MB to avoid empty placeholders
    }

    fun getModelFile(spec: ModelSpec): File {
        return File(modelsDir, spec.filename)
    }

    suspend fun setActiveModel(spec: ModelSpec): Boolean = withContext(Dispatchers.IO) {
        _activeModel.value = spec
        Log.i(TAG, "Active edge model set to: ${spec.displayName} (${spec.filename})")
        true
    }

    /**
     * Import a user-provided .gguf file from phone storage (e.g. from Downloads).
     */
    suspend fun importLocalModel(uri: Uri, targetFileName: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val destFile = File(modelsDir, targetFileName)
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.i(TAG, "Successfully imported local GGUF model: ${destFile.absolutePath} (${destFile.length() / (1024*1024)} MB)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to import model from uri: $uri", e)
            false
        }
    }

    /**
     * Downloads model directly to internal storage via DownloadManager or HTTP client.
     */
    fun startDownload(spec: ModelSpec) {
        val current = _downloadStates.value.toMutableMap()
        current[spec.id] = ModelDownloadProgress(
            modelId = spec.id,
            isDownloading = true,
            progressPct = 0.05f
        )
        _downloadStates.value = current

        try {
            val downloadManager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
            val request = DownloadManager.Request(Uri.parse(spec.downloadUrl))
                .setTitle("Downloading ${spec.displayName}")
                .setDescription("Downloading on-device quantized AI model (${spec.sizeMb} MB)")
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                .setDestinationUri(Uri.fromFile(File(modelsDir, spec.filename)))
                .setAllowedOverMetered(true)
                .setAllowedOverRoaming(false)

            val downloadId = downloadManager.enqueue(request)
            Log.i(TAG, "Enqueued download for ${spec.displayName} (Download ID: $downloadId)")
        } catch (e: Exception) {
            Log.e(TAG, "Could not start download for ${spec.displayName}", e)
            current[spec.id] = ModelDownloadProgress(modelId = spec.id, isDownloading = false)
            _downloadStates.value = current
        }
    }
}
