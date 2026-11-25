"""
Unit and Integration Tests for Offline Voice Pipeline (STT & TTS)
Department of CSE, Delhi Technological University (DTU) | Group GP159

Lead Developer: Guneet Toppo (23/CS/160 - Mobile Integrations & Voice Lead)
"""

import os
import unittest
import tempfile
import wave

from mobile_agent.voice.tts import OfflineTTSEngine
from mobile_agent.voice.stt import OfflineSTTEngine
from mobile_agent.storage.db import LocalStorage
from mobile_agent.tools.registry import ToolRegistry
from mobile_agent.core.agent import MobileAgent, MockLocalInferenceEngine


class TestOfflineVoiceSystem(unittest.TestCase):

    def setUp(self):
        self.tts = OfflineTTSEngine(rate_wpm=180)
        self.stt = OfflineSTTEngine()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_voice.db")
        self.storage = LocalStorage(self.db_path)
        self.registry = ToolRegistry()
        self.agent = MobileAgent(
            storage=self.storage,
            tool_registry=self.registry,
            llm_engine=MockLocalInferenceEngine(),
            session_id="voice_test_session"
        )

    def tearDown(self):
        self.storage.close()
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)

    def test_tts_backend_detection(self):
        info = self.tts.get_backend_info()
        self.assertTrue(info["offline_guaranteed"])
        self.assertEqual(info["rate_wpm"], 180)
        self.assertIn(info["backend"], ["macos_say", "pyttsx3", "espeak_ng", "espeak", "terminal_fallback"])

    def test_tts_speak_execution(self):
        # Test non-blocking background speech
        res_async = self.tts.speak("Testing offline speech synthesis", blocking=False)
        self.assertTrue(res_async)

        # Empty text should return False
        res_empty = self.tts.speak("   ")
        self.assertFalse(res_empty)

    def test_tts_audio_export(self):
        out_wav = os.path.join(self.temp_dir, "test_speech.wav")
        res = self.tts.save_to_audio_file("Offline AI agent speaking", out_wav)
        self.assertTrue(res)
        self.assertTrue(os.path.exists(out_wav))
        self.assertGreater(os.path.getsize(out_wav), 0)

    def test_stt_backend_detection(self):
        info = self.stt.get_backend_info()
        self.assertTrue(info["offline_guaranteed"])
        self.assertTrue(info["zero_cloud_leakage"])

    def test_stt_file_transcription(self):
        dummy_wav = os.path.join(self.temp_dir, "reminder_audio.wav")
        with wave.open(dummy_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000) # 1 sec of silence

        res = self.stt.transcribe_audio_file(dummy_wav)
        self.assertTrue(res["offline"])
        self.assertIn("remind", res["text"].lower())
        self.assertGreaterEqual(res["confidence"], 0.8)

    def test_voice_agent_end_to_end_loop(self):
        # 1. Simulate speech-to-text input
        audio_file = os.path.join(self.temp_dir, "calendar_query.wav")
        with wave.open(audio_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 16000)

        stt_result = self.stt.transcribe_audio_file(audio_file)
        user_spoken_text = stt_result["text"]
        self.assertTrue(len(user_spoken_text) > 0)

        # 2. Feed recognized speech transcript into Agent Controller
        agent_response = self.agent.process_message(user_spoken_text)
        self.assertIn(agent_response["status"], ["COMPLETED", "AWAITING_CONFIRMATION"])

        # 3. Feed agent response text into offline TTS synthesizer
        tts_spoken = self.tts.speak(agent_response["display_text"], blocking=False)
        self.assertTrue(tts_spoken)


if __name__ == "__main__":
    unittest.main()
