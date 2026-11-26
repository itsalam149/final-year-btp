package com.dtu.mobileagent.voice

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Locale

sealed class VoiceInputState {
    object Idle : VoiceInputState()
    object Listening : VoiceInputState()
    data class Decoded(val transcript: String) : VoiceInputState()
    data class Error(val message: String) : VoiceInputState()
}

class VoiceManager(
    private val context: Context,
    private val onTranscriptReady: (String) -> Unit
) {
    private val tag = "VoiceManager"

    private var speechRecognizer: SpeechRecognizer? = null
    private var textToSpeech: TextToSpeech? = null
    private var isTtsReady = false

    private val _inputState = MutableStateFlow<VoiceInputState>(VoiceInputState.Idle)
    val inputState: StateFlow<VoiceInputState> = _inputState.asStateFlow()

    private val _isSpeaking = MutableStateFlow(false)
    val isSpeaking: StateFlow<Boolean> = _isSpeaking.asStateFlow()

    init {
        initializeTts()
        initializeSpeechRecognizer()
    }

    private fun initializeTts() {
        textToSpeech = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech?.let { tts ->
                    val candidateLocales = listOf(
                        Locale.getDefault(),
                        Locale.US,
                        Locale.UK,
                        Locale.ENGLISH
                    )
                    for (loc in candidateLocales) {
                        val result = tts.setLanguage(loc)
                        if (result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED) {
                            Log.i(tag, "Offline TextToSpeech engine configured with locale: $loc")
                            break
                        }
                    }
                    tts.setPitch(1.0f)
                    tts.setSpeechRate(1.0f)
                    isTtsReady = true
                    Log.i(tag, "Offline TextToSpeech engine initialized.")
                }
            } else {
                Log.e(tag, "TextToSpeech initialization failed with status: $status")
            }
        }

        textToSpeech?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                _isSpeaking.value = true
            }

            override fun onDone(utteranceId: String?) {
                _isSpeaking.value = false
            }

            override fun onError(utteranceId: String?) {
                _isSpeaking.value = false
            }
        })
    }

    private fun initializeSpeechRecognizer() {
        if (SpeechRecognizer.isRecognitionAvailable(context)) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
                setRecognitionListener(createRecognitionListener())
            }
        } else {
            Log.w(tag, "SpeechRecognizer unavailable on this device configuration.")
        }
    }

    private fun createRecognitionListener(): RecognitionListener {
        return object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                _inputState.value = VoiceInputState.Listening
            }

            override fun onBeginningOfSpeech() {}

            override fun onRmsChanged(rmsdB: Float) {}

            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                _inputState.value = VoiceInputState.Idle
            }

            override fun onError(error: Int) {
                // If it's a silent timeout or no match, quietly return to Idle so the UI stays clean
                if (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                    _inputState.value = VoiceInputState.Idle
                    return
                }
                val errorMsg = when (error) {
                    SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
                    SpeechRecognizer.ERROR_CLIENT -> "Speech recognition not available or active"
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Audio permission required"
                    SpeechRecognizer.ERROR_NETWORK -> "Offline speech recognition used"
                    else -> "Speech recognition paused"
                }
                _inputState.value = VoiceInputState.Error(errorMsg)
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val transcript = matches?.firstOrNull() ?: ""
                if (transcript.isNotBlank()) {
                    _inputState.value = VoiceInputState.Decoded(transcript)
                    onTranscriptReady(transcript)
                } else {
                    _inputState.value = VoiceInputState.Idle
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}

            override fun onEvent(eventType: Int, params: Bundle?) {}
        }
    }

    fun startListening() {
        if (speechRecognizer == null) {
            initializeSpeechRecognizer()
        }

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault().toString())
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, Locale.getDefault().toString())
            // Instruct Android system to prefer local on-device acoustic recognition models
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }

        try {
            stopSpeaking()
            speechRecognizer?.startListening(intent)
            _inputState.value = VoiceInputState.Listening
        } catch (e: Exception) {
            Log.e(tag, "Error starting offline speech recognizer", e)
            _inputState.value = VoiceInputState.Error(e.message ?: "Failed to start listening")
        }
    }

    fun stopListening() {
        try {
            speechRecognizer?.stopListening()
            _inputState.value = VoiceInputState.Idle
        } catch (e: Exception) {
            Log.e(tag, "Error stopping recognizer", e)
        }
    }

    fun speak(text: String) {
        if (text.isBlank()) return

        // Clean out markdown symbols so speech sounds natural and fluid
        val cleanSpeech = text
            .replace(Regex("""\*\*([^*]+)\*\*"""), "$1")
            .replace(Regex("""\*([^*]+)\*"""), "$1")
            .replace(Regex("""`([^`]+)`"""), "$1")
            .replace(Regex("""```[\s\S]*?```"""), "code block")
            .replace(Regex("""[#•\-–—]"""), " ")
            .replace(Regex("""\s+"""), " ")
            .trim()

        if (cleanSpeech.isBlank()) return

        if (!isTtsReady) {
            initializeTts()
        }

        val utteranceId = "msg_${System.currentTimeMillis()}"
        textToSpeech?.speak(cleanSpeech, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }

    fun stopSpeaking() {
        textToSpeech?.stop()
        _isSpeaking.value = false
    }

    fun destroy() {
        speechRecognizer?.destroy()
        speechRecognizer = null

        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        isTtsReady = false
    }
}
