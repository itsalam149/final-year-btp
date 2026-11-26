package com.dtu.mobileagent

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.dtu.mobileagent.engine.AgentController
import com.dtu.mobileagent.ui.ChatScreen
import com.dtu.mobileagent.ui.OfflineMobileAgentTheme
import com.dtu.mobileagent.engine.ModelGalleryManager
import com.dtu.mobileagent.voice.VoiceManager
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var agentController: AgentController
    private lateinit var voiceManager: VoiceManager
    private lateinit var galleryManager: ModelGalleryManager

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            lifecycleScope.launch {
                val success = galleryManager.importLocalModel(it, "imported_model.gguf")
                if (success) {
                    agentController.switchModel("imported_model.gguf")
                }
            }
        }
    }

    private var pendingAudioAction: (() -> Unit)? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        if (grants[Manifest.permission.RECORD_AUDIO] == true) {
            pendingAudioAction?.invoke()
            pendingAudioAction = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val app = application as MobileAgentApplication
        galleryManager = ModelGalleryManager(this)
        agentController = AgentController(this, app.database)

        // Initialize offline voice pipeline
        voiceManager = VoiceManager(this) { transcript ->
            lifecycleScope.launch {
                agentController.handleUserMessage(transcript)
            }
        }

        // Connect agent speech output to native offline TTS
        agentController.setSpeechOutputCallback { responseText ->
            voiceManager.speak(responseText)
        }

        // Request local device permissions (Calendar, Notifications, Audio)
        requestLocalPermissions()

        // Initialize local LLM engine in background
        lifecycleScope.launch {
            agentController.initialize()
        }

        setContent {
            OfflineMobileAgentTheme {
                ChatScreen(
                    controller = agentController,
                    voiceManager = voiceManager,
                    galleryManager = galleryManager,
                    onImportLocalModel = { filePickerLauncher.launch("*/*") },
                    onRequestRecordAudio = { onGranted ->
                        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                            onGranted()
                        } else {
                            pendingAudioAction = onGranted
                            permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
                        }
                    }
                )
            }
        }
    }

    private fun requestLocalPermissions() {
        val permissions = mutableListOf<String>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.RECORD_AUDIO)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_CALENDAR)
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.READ_CALENDAR)
        }

        if (permissions.isNotEmpty()) {
            permissionLauncher.launch(permissions.toTypedArray())
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        voiceManager.destroy()
    }
}
