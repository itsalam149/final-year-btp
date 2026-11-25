package com.dtu.mobileagent.engine

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

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
    val totalBytes: Long = 0L,
    val errorMessage: String? = null
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
                id = "gemma-2-2b",
                displayName = "Gemma-2-2B-Instruct",
                filename = "gemma-2-2b-it-q4_k_m.gguf",
                sizeMb = 1629,
                ramMb = 2200,
                precision = "GGUF Q4_K_M (4-bit)",
                downloadUrl = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
                description = "Google's premier 2.6B parameter edge model. Exceptional reasoning, wide general knowledge, and conversational depth."
            ),
            ModelSpec(
                id = "llama-3.2-3b",
                displayName = "Llama-3.2-3B-Instruct",
                filename = "llama-3.2-3b-instruct-q4_k_m.gguf",
                sizeMb = 1925,
                ramMb = 2600,
                precision = "GGUF Q4_K_M (4-bit)",
                downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                description = "Higher capacity 3-billion parameter model. Rich conversational vocabulary and complex multi-step reasoning."
            ),
            ModelSpec(
                id = "llama-3.2-1b",
                displayName = "Llama-3.2-1B-Instruct",
                filename = "llama-3.2-1b-instruct-q4_k_m.gguf",
                sizeMb = 804,
                ramMb = 1040,
                precision = "GGUF Q4_K_M (4-bit)",
                downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                description = "Fast edge model for standard mobile tasks. Excellent instruction following and grammar-guided tool calling."
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
                description = "Conversational chat model trained on 3 trillion tokens. Smooth mobile dialogue."
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

    private val downloadJobs = mutableMapOf<String, Job>()
    private val downloadScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

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
            val stream: InputStream? = context.contentResolver.openInputStream(uri)
            stream?.use { input ->
                destFile.outputStream().use { output ->
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
     * Downloads model directly to internal storage via streaming HTTP connection.
     * Follows redirects automatically and writes directly to app private storage.
     */
    fun startDownload(spec: ModelSpec) {
        if (downloadJobs[spec.id]?.isActive == true) return

        val current = _downloadStates.value.toMutableMap()
        current[spec.id] = ModelDownloadProgress(
            modelId = spec.id,
            isDownloading = true,
            progressPct = 0.001f,
            downloadedBytes = 0L,
            totalBytes = spec.sizeMb.toLong() * 1024L * 1024L
        )
        _downloadStates.value = current

        val job = downloadScope.launch {
            val destFile = File(modelsDir, spec.filename)
            val tempFile = File(modelsDir, "${spec.filename}.download")
            try {
                var currentUrl = spec.downloadUrl
                var connection: HttpURLConnection
                var redirects = 0

                while (true) {
                    val url = URL(currentUrl)
                    connection = url.openConnection() as HttpURLConnection
                    connection.instanceFollowRedirects = false
                    connection.connectTimeout = 30000
                    connection.readTimeout = 60000
                    connection.setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 14) MobileAgent/1.0")

                    val code = connection.responseCode
                    if (code == HttpURLConnection.HTTP_MOVED_PERM ||
                        code == HttpURLConnection.HTTP_MOVED_TEMP ||
                        code == HttpURLConnection.HTTP_SEE_OTHER ||
                        code == 307 || code == 308) {
                        val location = connection.getHeaderField("Location")
                        if (location.isNullOrEmpty()) throw IllegalStateException("Redirect missing Location header")
                        currentUrl = if (location.startsWith("http://") || location.startsWith("https://")) {
                            location
                        } else {
                            URL(URL(currentUrl), location).toString()
                        }
                        connection.disconnect()
                        redirects++
                        if (redirects > 10) throw IllegalStateException("Exceeded max redirects")
                        continue
                    }
                    if (code != HttpURLConnection.HTTP_OK) {
                        throw IllegalStateException("HTTP error $code: ${connection.responseMessage}")
                    }
                    break
                }

                val remoteSize = connection.contentLengthLong.let {
                    if (it > 0) it else connection.getHeaderField("x-linked-size")?.toLongOrNull() ?: (spec.sizeMb.toLong() * 1024L * 1024L)
                }

                if (tempFile.exists()) tempFile.delete()

                connection.inputStream.use { input ->
                    FileOutputStream(tempFile).use { output ->
                        val buffer = ByteArray(256 * 1024) // 256 KB buffer for high speed
                        var bytesRead: Int
                        var downloaded = 0L
                        var lastUpdate = System.currentTimeMillis()

                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                            downloaded += bytesRead

                            val now = System.currentTimeMillis()
                            if (now - lastUpdate > 250) {
                                lastUpdate = now
                                val pct = (downloaded.toFloat() / remoteSize.toFloat()).coerceIn(0f, 0.99f)
                                val updated = _downloadStates.value.toMutableMap()
                                updated[spec.id] = ModelDownloadProgress(
                                    modelId = spec.id,
                                    isDownloading = true,
                                    progressPct = pct,
                                    downloadedBytes = downloaded,
                                    totalBytes = remoteSize
                                )
                                _downloadStates.value = updated
                            }
                        }
                        output.flush()
                    }
                }

                if (tempFile.exists()) {
                    if (destFile.exists()) destFile.delete()
                    tempFile.renameTo(destFile)
                }

                val finished = _downloadStates.value.toMutableMap()
                finished[spec.id] = ModelDownloadProgress(
                    modelId = spec.id,
                    isDownloading = false,
                    progressPct = 1.0f,
                    downloadedBytes = destFile.length(),
                    totalBytes = destFile.length()
                )
                _downloadStates.value = finished
                Log.i(TAG, "Download successfully completed: ${destFile.absolutePath} (${destFile.length() / (1024 * 1024)} MB)")
            } catch (e: Exception) {
                Log.e(TAG, "Download failed for ${spec.displayName}", e)
                if (tempFile.exists()) tempFile.delete()
                val errMap = _downloadStates.value.toMutableMap()
                errMap[spec.id] = ModelDownloadProgress(
                    modelId = spec.id,
                    isDownloading = false,
                    progressPct = 0f,
                    errorMessage = e.message ?: "Download failed"
                )
                _downloadStates.value = errMap
            } finally {
                downloadJobs.remove(spec.id)
            }
        }
        downloadJobs[spec.id] = job
    }

    fun cancelDownload(spec: ModelSpec) {
        downloadJobs[spec.id]?.cancel()
        downloadJobs.remove(spec.id)
        val tempFile = File(modelsDir, "${spec.filename}.download")
        if (tempFile.exists()) tempFile.delete()
        val updated = _downloadStates.value.toMutableMap()
        updated[spec.id] = ModelDownloadProgress(
            modelId = spec.id,
            isDownloading = false,
            progressPct = 0f
        )
        _downloadStates.value = updated
    }
}
