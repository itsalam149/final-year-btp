package com.dtu.mobileagent.engine

import android.content.Context
import android.os.Environment
import android.util.Log
import com.dtu.mobileagent.data.local.AppDatabase
import com.dtu.mobileagent.data.local.entities.DocumentEntity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class DocumentIndexer(
    private val context: Context,
    private val database: AppDatabase
) {
    companion object {
        private const val TAG = "DocumentIndexer"
        private const val MAX_CONTENT_BYTES = 8192 // 8 KB content snippet budget per file
        private const val MAX_INDEXED_FILES = 100
        private val SUPPORTED_EXTENSIONS = setOf(
            "txt", "md", "markdown", "json", "csv", "pdf", "doc", "docx", "py", "kt", "c", "cpp"
        )
    }

    suspend fun scanDeviceDocuments(): Int = withContext(Dispatchers.IO) {
        var count = 0
        try {
            ensureInitialAppDocuments()

            val directoriesToScan = mutableListOf<File>()

            // 1. App-specific internal documents directory
            val internalDocs = File(context.filesDir, "documents")
            if (internalDocs.exists() && internalDocs.isDirectory) {
                directoriesToScan.add(internalDocs)
            }

            // 2. App-specific external documents directory
            val appExternalDocs = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            if (appExternalDocs != null && appExternalDocs.exists() && appExternalDocs.isDirectory) {
                directoriesToScan.add(appExternalDocs)
            }

            // 3. Shared Public Documents directory (if accessible)
            val publicDocs = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            if (publicDocs != null && publicDocs.exists() && publicDocs.isDirectory && publicDocs.canRead()) {
                directoriesToScan.add(publicDocs)
            }

            // 4. Shared Public Downloads directory (if accessible)
            val publicDownloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (publicDownloads != null && publicDownloads.exists() && publicDownloads.isDirectory && publicDownloads.canRead()) {
                directoriesToScan.add(publicDownloads)
            }

            for (dir in directoriesToScan) {
                count += indexDirectory(dir, remainingQuota = MAX_INDEXED_FILES - count)
                if (count >= MAX_INDEXED_FILES) break
            }

            Log.i(TAG, "Completed document scan: indexed $count documents into Room database.")
        } catch (e: Exception) {
            Log.e(TAG, "Error during document indexing scan", e)
        }
        count
    }

    private suspend fun indexDirectory(dir: File, remainingQuota: Int): Int {
        if (remainingQuota <= 0) return 0
        var indexed = 0

        val files = dir.listFiles() ?: return 0
        for (file in files) {
            if (indexed >= remainingQuota) break
            if (file.isDirectory && !file.name.startsWith(".")) {
                // Recursively scan subdirectories up to 1 level
                val subFiles = file.listFiles() ?: continue
                for (sub in subFiles) {
                    if (indexed >= remainingQuota) break
                    if (sub.isFile && isSupported(sub)) {
                        if (indexSingleFile(sub)) indexed++
                    }
                }
            } else if (file.isFile && isSupported(file)) {
                if (indexSingleFile(file)) indexed++
            }
        }
        return indexed
    }

    private fun isSupported(file: File): Boolean {
        if (file.isHidden || file.name.startsWith(".")) return false
        val ext = file.extension.lowercase()
        return ext in SUPPORTED_EXTENSIONS && file.length() < 10 * 1024 * 1024 // < 10MB
    }

    private suspend fun indexSingleFile(file: File): Boolean {
        return try {
            val ext = file.extension.lowercase()
            val isText = ext in setOf("txt", "md", "markdown", "json", "csv", "py", "kt", "c", "cpp")

            val contentSnippet: String
            val summary: String

            if (isText) {
                val fullText = try {
                    file.bufferedReader().use { reader ->
                        val buffer = CharArray(MAX_CONTENT_BYTES)
                        val readChars = reader.read(buffer, 0, MAX_CONTENT_BYTES)
                        if (readChars > 0) String(buffer, 0, readChars) else ""
                    }
                } catch (e: Exception) {
                    "File: ${file.name} (read error: ${e.message})"
                }
                contentSnippet = fullText
                summary = generateTextSummary(file.name, fullText)
            } else {
                val sizeKb = file.length() / 1024
                contentSnippet = "File name: ${file.name}\nType: ${ext.uppercase()} Document\nSize: $sizeKb KB\nLocation: ${file.absolutePath}"
                summary = "${ext.uppercase()} Document: ${file.name} ($sizeKb KB)"
            }

            database.documentDao().insertDocument(
                DocumentEntity(
                    filename = file.name,
                    filepath = file.absolutePath,
                    content = contentSnippet,
                    summary = summary
                )
            )
            true
        } catch (e: Exception) {
            Log.w(TAG, "Failed to index file: ${file.absolutePath}", e)
            false
        }
    }

    private fun generateTextSummary(filename: String, text: String): String {
        if (text.isBlank()) return "Document: $filename"
        val lines = text.lines().map { it.trim() }.filter { it.isNotBlank() }
        if (lines.isNotEmpty()) {
            val firstLine = lines.first()
            if (firstLine.length > 100) return firstLine.take(100) + "..."
            return firstLine
        }
        return "Document $filename (${text.length} chars)"
    }

    private fun ensureInitialAppDocuments() {
        val docsDir = File(context.filesDir, "documents")
        if (!docsDir.exists()) docsDir.mkdirs()

        val sampleOverview = File(docsDir, "btp_project_overview.txt")
        if (!sampleOverview.exists()) {
            sampleOverview.writeText(
                """
                Project Title: Privacy-First Offline AI Mobile Agent Using LLM Quantization
                Department: Computer Science and Engineering, Delhi Technological University (DTU)
                Supervisor: Ms. Gull Kaur
                Group ID: GP159
                Authors: Faqre Alam, Ekansh Agarwal, Guneet Toppo, Bhawishya
                Objective: Deploy a 100% offline edge AI assistant with zero network data egress, utilizing 4-bit quantized GGUF models (Gemma-2-2B, Llama-3.2-1B, Qwen2.5-0.5B), GBNF grammar-guided tool dispatch, Room SQLite memory persistence, and ARM NEON hardware acceleration.
                """.trimIndent()
            )
        }

        val sampleSpecs = File(docsDir, "quantization_specs.txt")
        if (!sampleSpecs.exists()) {
            sampleSpecs.writeText(
                """
                Quantization Specifications & Hardware Benchmarks:
                - Gemma-2-2B-IT (Q4_K_M): 1,629 MB RAM footprint, 22.4 tokens/second on Snapdragon 8 Gen 2 / Dimensity 9200.
                - Llama-3.2-1B-Instruct (Q4_K_M): 804 MB RAM footprint, 28.6 tokens/second.
                - Qwen2.5-0.5B-Instruct (Q4_K_M): 412 MB RAM footprint, 41.2 tokens/second.
                - GBNF Grammar Adherence: Eliminates syntax and parameter formatting hallucinations (100% valid JSON).
                """.trimIndent()
            )
        }

        val sampleCalendar = File(docsDir, "dtu_academic_calendar_2026.txt")
        if (!sampleCalendar.exists()) {
            sampleCalendar.writeText(
                """
                DTU Academic Calendar 2026 Highlights:
                - Major Project Phase II Presentations: Late November 2026
                - Final Year Viva Voce & Evaluation: December 2026
                - Department: Computer Science & Engineering, Delhi Technological University
                - Venue: Academic Block 3, DTU Main Campus, Shahbad Daulatpur, Bawana Road, Delhi
                """.trimIndent()
            )
        }
    }
}
