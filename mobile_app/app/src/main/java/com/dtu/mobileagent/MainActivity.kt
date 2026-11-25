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
import com.dtu.mobileagent.voice.VoiceManager
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var agentController: AgentController
    private lateinit var voiceManager: VoiceManager

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { _ ->
        // Permissions granted or handled via local fallback
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val app = application as MobileAgentApplication
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
                    voiceManager = voiceManager
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
