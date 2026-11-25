"""
Interactive Terminal CLI for Privacy-First Offline Mobile Agent (JARVIS).
Allows live interaction, testing mobile tools, inspecting SQLite persistence,
and verifying security confirmation gates.
"""

import sys
import os
import json
import argparse
from datetime import datetime

from mobile_agent.storage.db import LocalStorage
from mobile_agent.tools.registry import ToolRegistry
from mobile_agent.core.agent import MobileAgent, MockLocalInferenceEngine
from mobile_agent.voice.tts import OfflineTTSEngine
from mobile_agent.voice.stt import OfflineSTTEngine


# ANSI Color formatting
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner(voice_enabled: bool = False):
    voice_status = f"{GREEN}ON{RESET}" if voice_enabled else f"{RED}OFF{RESET}"
    banner = f"""{CYAN}{BOLD}
======================================================================
  JARVIS: Privacy-First Offline AI Mobile Agent (DTU GP159)
  Zero Cloud Reliance | 100% Local Inference & SQLite Storage
  Voice Mode: {voice_status} | Offline Audio Synthesizer Active
======================================================================{RESET}
Type any instruction (e.g. 'Remind me to call Rahul tomorrow at 10 am',
'Check my schedule for tomorrow', 'Save a note titled Project Ideas',
'Find my project proposal PDF', 'Draft a message to Rahul').
Commands: {YELLOW}/tools{RESET}, {YELLOW}/notes{RESET}, {YELLOW}/reminders{RESET}, {YELLOW}/audit{RESET}, {YELLOW}/voice [on|off]{RESET}, {YELLOW}/listen <file>{RESET}, {YELLOW}/clear{RESET}, {YELLOW}/exit{RESET}
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="Privacy-First Offline AI Mobile Agent CLI")
    parser.add_argument("--voice", action="store_true", help="Enable offline text-to-speech spoken responses")
    args = parser.parse_args()

    voice_mode = args.voice
    tts_engine = OfflineTTSEngine()
    stt_engine = OfflineSTTEngine()

    db_file = os.path.join(os.path.dirname(__file__), "..", "results", "local_mobile_sandbox.db")
    storage = LocalStorage(db_file)
    registry = ToolRegistry()
    agent = MobileAgent(
        storage=storage,
        tool_registry=registry,
        llm_engine=MockLocalInferenceEngine(),
        session_id="interactive_cli"
    )

    # Seed sample data if empty
    if not storage.search_documents("proposal"):
        storage.add_document(
            filename="project_proposal.pdf",
            title="BTP Project Proposal",
            content_text="Project Title: Privacy-First Offline AI Mobile Agent.\nResearch objectives: Deploy quantized LLMs on mobile smartphones.\nMemory budget: Under 4GB RAM.\nAuthors: Faqre Alam, Ekansh Agarwal, Guneet Toppo.\nSupervisor: Ms. Gull Kaur.",
            tags="proposal, btp, mobile"
        )
    if not storage.search_calendar("2026-11-27"):
        storage.add_calendar_event(
            title="BTP Progress Review with Ms. Gull Kaur",
            start_time_iso="2026-11-27T11:00:00",
            end_time_iso="2026-11-27T12:00:00",
            location="DTU Cabin 204"
        )

    print_banner(voice_mode)

    pending_confirmation = None

    while True:
        try:
            prompt_label = f"{YELLOW}[CONFIRM Y/N]{RESET} > " if pending_confirmation else f"{GREEN}JARVIS{RESET} > "
            user_input = input(prompt_label).strip()

            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                print(f"{CYAN}Shutting down JARVIS offline sandbox. Goodbye!{RESET}")
                break

            elif user_input.lower() == "/tools":
                print(f"\n{BOLD}Registered Mobile Tools:{RESET}")
                for tool in registry.list_tools():
                    sec_color = YELLOW if tool.security_level.value == "REQUIRES_CONFIRMATION" else GREEN
                    print(f"- {BOLD}{tool.name}{RESET} [{sec_color}{tool.security_level.value}{RESET}]: {tool.description}")
                print()
                continue

            elif user_input.lower() == "/notes":
                notes = storage.list_notes()
                print(f"\n{BOLD}Saved Local Notes ({len(notes)}):{RESET}")
                for n in notes:
                    print(f"[{n['id']}] {BOLD}{n['title']}{RESET} ({n['updated_at']}): {n['content']}")
                print()
                continue

            elif user_input.lower() == "/reminders":
                reminders = storage.list_reminders()
                print(f"\n{BOLD}Active Reminders ({len(reminders)}):{RESET}")
                for r in reminders:
                    print(f"[{r['id']}] {BOLD}{r['title']}{RESET} (Due: {r['due_time_iso']}, Priority: {r['priority']})")
                print()
                continue

            elif user_input.lower() == "/audit":
                logs = storage.get_audit_logs(limit=10)
                print(f"\n{BOLD}Security Audit Logs ({len(logs)} recent entries):{RESET}")
                for l in logs:
                    conf_str = "CONFIRMED" if l["user_confirmed"] else "AUTO/NONE"
                    print(f"[{l['timestamp']}] Tool: {BOLD}{l['tool_name']}{RESET} | Status: {l['status']} | User: {conf_str}")
                    print(f"    Params: {l['parameters_json']}")
                print()
                continue

            elif user_input.lower().startswith("/voice"):
                parts = user_input.strip().split()
                if len(parts) > 1 and parts[1].lower() in ["on", "enable", "true", "1"]:
                    voice_mode = True
                    print(f"{GREEN}Offline Voice Mode enabled. Spoken replies active.{RESET}")
                    tts_engine.speak("Voice output enabled.")
                elif len(parts) > 1 and parts[1].lower() in ["off", "disable", "false", "0"]:
                    voice_mode = False
                    print(f"{YELLOW}Offline Voice Mode disabled. Spoken replies muted.{RESET}")
                else:
                    voice_mode = not voice_mode
                    status_str = f"{GREEN}ON{RESET}" if voice_mode else f"{RED}OFF{RESET}"
                    print(f"Voice Mode toggled: {status_str}")
                continue

            elif user_input.lower().startswith("/speak"):
                text_to_speak = user_input[6:].strip()
                if text_to_speak:
                    print(f"{CYAN}[Speaking offline: '{text_to_speak}']{RESET}")
                    tts_engine.speak(text_to_speak)
                continue

            elif user_input.lower().startswith("/listen"):
                parts = user_input.strip().split(maxsplit=1)
                audio_path = parts[1].strip() if len(parts) > 1 else "sample_voice.wav"
                print(f"{CYAN}[Offline STT Processing: {audio_path}...]{RESET}")
                stt_res = stt_engine.transcribe_audio_file(audio_path)
                recognized = stt_res.get("text", "")
                if recognized:
                    print(f"{GREEN}[Recognized Audio]:{RESET} \"{recognized}\"")
                    user_input = recognized
                else:
                    print(f"{RED}Could not decode speech from {audio_path}{RESET}")
                    continue

            elif user_input.lower() == "/clear":
                os.system("clear" if os.name == "posix" else "cls")
                print_banner(voice_mode)
                continue

            # Handle pending confirmation response
            if pending_confirmation:
                if user_input.lower() in ["y", "yes"]:
                    print(f"{CYAN}[Executing with User Authorization...]{RESET}")
                    # Re-run with user_confirmed=True
                    response = agent.process_message(pending_confirmation["original_input"], user_confirmed=True)
                    pending_confirmation = None
                    print(f"\n{GREEN}{BOLD}Result:{RESET} {response['display_text']}\n")
                    if voice_mode:
                        tts_engine.speak(response['display_text'], blocking=False)
                else:
                    print(f"{RED}Action cancelled by user.{RESET}\n")
                    if voice_mode:
                        tts_engine.speak("Action cancelled.", blocking=False)
                    pending_confirmation = None
                continue

            # Normal message processing
            print(f"{CYAN}[Thinking... Checking Local Schemas & Context]{RESET}")
            response = agent.process_message(user_input, user_confirmed=False)

            if response.get("thought"):
                print(f"{CYAN}Thought:{RESET} {response['thought']}")

            if response["status"] == "AWAITING_CONFIRMATION":
                print(f"\n{YELLOW}{BOLD}SECURITY CONFIRMATION REQUIRED:{RESET}")
                print(response["display_text"])
                print(f"{YELLOW}Proceed with this action? (y/n){RESET}")
                if voice_mode:
                    tts_engine.speak("Confirmation required. Should I proceed?", blocking=False)
                pending_confirmation = {
                    "original_input": user_input,
                    "tool": response.get("tool"),
                    "params": response.get("parameters")
                }
            elif response["status"] == "COMPLETED":
                tool_used = response.get("tool")
                if tool_used:
                    print(f"{GREEN}[Tool Invoked: {tool_used}]{RESET}")
                print(f"\n{BOLD}{response['display_text']}{RESET}\n")
                if voice_mode:
                    tts_engine.speak(response['display_text'], blocking=False)
            else:
                print(f"\n{RED}{response['display_text']}{RESET}\n")
                if voice_mode:
                    tts_engine.speak(response['display_text'], blocking=False)

        except (KeyboardInterrupt, EOFError):
            print(f"\n{CYAN}Session ended.{RESET}")
            break


if __name__ == "__main__":
    main()
