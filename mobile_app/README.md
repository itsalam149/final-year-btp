# Offline Mobile Agent (Android Client)
**Privacy-First Offline AI Mobile Agent Using LLM Quantization (JARVIS-Like On-Device Assistant)**  
*Department of Computer Science & Engineering, Delhi Technological University (DTU)*  
**Group:** DTU/CSE/BTP/GP159 | **Supervisor:** Ms. Gull Kaur  
**Authors:** Faqre Alam (23/CS/150), Ekansh Agarwal (23/CS/149), Guneet Toppo (23/CS/160)

---

## 📱 Overview

The `mobile_app` module is the native Android implementation of our privacy-first on-device AI assistant. It couples **C++ JNI inference (`llama.cpp`)** with **GBNF grammar-constrained decoding**, an **offline Room SQLite data layer**, and a **Jetpack Compose UI** that guarantees zero cloud egress.

```
+-------------------------------------------------------------------------+
|                       Jetpack Compose UI (ChatScreen)                   |
|   - Real-time token streaming & reasoning thought display              |
|   - Dynamic Security Pill ("100% Offline | Zero Cloud Egress")           |
|   - Human-in-the-Loop Confirmation Dialog for external actions         |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  Kotlin Agent Controller & ReAct Loop                   |
|   - Prompt construction with strict context budget (2048 tokens)        |
|   - Function call parsing & tool safety gate evaluation                |
|   - Security audit logging for every planned and executed action        |
+-------------------------------------------------------------------------+
                    |                                   |
                    v                                   v
+-----------------------------------+   +---------------------------------+
|        Native C++ JNI Bridge      |   |       Local Room SQLite DB      |
|  (llama-bridge.cpp + CMakeLists)  |   | (Notes, Reminders, Calendar,    |
|   - GBNF Grammar logit masking    |   |  Documents, Audit Log, Messages)|
|   - ARMv8.2-A NEON dot-product    |   |                                 |
|   - 4 threads, Q4_K_M weights     |   |  *Zero external cloud sync*     |
+-----------------------------------+   +---------------------------------+
```

---

## 🔒 Zero-Internet Guarantee

Unlike conventional cloud assistants (Siri, Google Assistant, ChatGPT App), this application enforces offline privacy at the operating system kernel and sandbox level:

1. **Absence of Internet Permission:**
   The `AndroidManifest.xml` deliberately **omits** `android.permission.INTERNET`.
   ```xml
   <!-- AndroidManifest.xml -->
   <!-- Notice: NO android.permission.INTERNET -->
   <uses-permission android:name="android.permission.SCHEDULE_EXACT_ALARM" />
   <uses-permission android:name="android.permission.READ_CALENDAR" />
   ```
   Even if the LLM attempted an HTTP request or socket transmission, Android's SELinux sandbox drops the network syscall instantly.

2. **Cloud Backup Disabled:**
   `res/xml/data_extraction_rules.xml` excludes database and preference files from Android Auto Cloud Backup.

3. **Human-in-the-Loop Safety Gate:**
   Safe operations (`manage_notes`, `search_calendar`, `create_reminder`, `search_local_docs`, `open_app`, `read_screen`) execute autonomously. External operations or mutating UI actions (`draft_message`, `click`, `type`) trigger a modal `ConfirmationDialog` before any intent or accessibility click is dispatched.

---

## 📱 3-Tier Mobile App Interaction Architecture

Android sandboxes each app in its own Linux UID. To let the on-device agent control apps across the entire phone, we implemented 3 distinct interaction tiers:

1. **Level 1: System-Integrated Local Storage & Providers**
   - High-throughput direct database queries using Android Room SQLite (`NoteEntity`, `ReminderEntity`, `CalendarEventEntity`, `DocumentEntity`).
   - Android OS integration for alarms via `AlarmManager` and calendar via `CalendarProvider`.

2. **Level 2: Third-Party Inter-App Communication (Intents & Deep Links)**
   - Launch arbitrary installed applications by name using `PackageManager.getLaunchIntentForPackage()` (e.g. Spotify, Maps, YouTube, WhatsApp, Settings).
   - Pre-fill message drafts with recipients and content via `Intent.ACTION_VIEW` and `Intent.ACTION_SENDTO` (WhatsApp & SMS).

3. **Level 3: Arbitrary App UI Automation (Accessibility Service)**
   - Powered by `AgentAccessibilityService` extending Android's `AccessibilityService`.
   - Traverses the active window's `AccessibilityNodeInfo` hierarchy to:
     - Find and click buttons by label or description (`clickElementByText`)
     - Focus and enter text into input fields (`enterTextInField`)
     - Inspect all visible elements on screen (`dumpVisibleScreenText`)
     - Execute global system navigation (`performNavigation`: Back, Home, Recents)
   - Read operations (`read_screen`) execute safely without blocking; mutating operations (`click`, `type`) are gated by user confirmation.

---

## 📂 Architecture & Directory Structure

```
mobile_app/
├── build.gradle.kts                   # Top-level Gradle configuration
├── settings.gradle.kts                # Project settings & module inclusion
├── gradle.properties                  # JVM memory and AndroidX flags
├── gradle/
│   └── libs.versions.toml             # Gradle version catalog (AGP 8.3.1, Kotlin 1.9.22, Room 2.6.1)
└── app/
    ├── build.gradle.kts               # App module build script (Compose, Room KSP, CMake)
    ├── proguard-rules.pro             # Proguard keep rules for JNI and Room
    └── src/
        └── main/
            ├── AndroidManifest.xml    # Sandboxed manifest (zero internet permission)
            ├── assets/
            │   ├── grammars/
            │   │   └── tool_calling.gbnf   # GBNF grammar enforcing valid JSON tool syntax
            │   └── prompts/
            │       └── system_prompt.txt   # Offline agent system prompt
            ├── cpp/
            │   ├── CMakeLists.txt     # NDK build targeting ARM NEON & dotprod
            │   ├── llama-bridge.h     # Native engine header & performance metrics
            │   └── llama-bridge.cpp   # JNI exports and grammar-guided inference
            ├── java/com/dtu/mobileagent/
            │   ├── MainActivity.kt               # Main entrypoint with permission handling
            │   ├── MobileAgentApplication.kt     # App context & Room database singleton
            │   ├── data/local/
            │   │   ├── AppDatabase.kt            # Room DB with initial academic seed data
            │   │   ├── entities/                 # Note, Reminder, Calendar, Document, AuditLog
            │   │   └── dao/                      # Room DAOs with Coroutine Flows
            │   ├── engine/
            │   │   ├── LlamaBridge.kt            # JNI wrapper & fallback simulation
            │   │   ├── ToolSchemas.kt            # Tool definitions, safety levels, JSON models
            │   │   ├── ToolExecutor.kt           # Autonomous & confirmed tool dispatch (Levels 1-3)
            │   │   └── AgentController.kt        # ReAct orchestration & prompt budgeting
            │   ├── service/
            │   │   └── AgentAccessibilityService.kt # Level 3 UI Automation (NodeInfo tree clicking & typing)
            │   ├── voice/
            │   │   └── VoiceManager.kt           # Offline Android SpeechRecognizer & TextToSpeech
            │   └── ui/
            │       ├── Theme.kt                  # Material3 Dark Theme
            │       ├── Color.kt                  # DTU Crimson (#B91C1C) & Emerald Offline
            │       ├── Components.kt             # Security badge, metrics bar, confirmation modal
            │       └── ChatScreen.kt             # Chat interface with streaming & voice controls
            └── res/
                ├── values/                       # strings.xml, colors.xml, themes.xml
                └── xml/
                    ├── accessibility_service_config.xml # Accessibility permissions & event filters
                    └── data_extraction_rules.xml # Zero cloud backup rules
```

---

## 🛠️ Build and Setup Instructions

### Prerequisites
- **Android Studio:** Hedgehog (2023.1.1) or newer (Iguana / Jellyfish / Koala recommended).
- **Android SDK:** Platform 34 (Android 14).
- **Android NDK:** Version 26.1.10909125 or newer.
- **CMake:** Version 3.22.1.
- **Device:** Android 8.0+ (API 26+) physical device or emulator. Physical ARM64 device recommended for full NEON hardware acceleration.

### Step-by-Step Build
1. Open Android Studio.
2. Select **Open** and select the `/Users/faqrealam149/Desktop/final-year-btp/mobile_app` folder.
3. Allow Gradle to sync dependencies via `libs.versions.toml`.
4. Run:
   ```bash
   ./gradlew assembleDebug
   ```
5. Install on connected device:
   ```bash
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

### Deploying Quantized GGUF Model Weights
To run with real GGUF weights on a physical device:
```bash
# Push quantized GGUF model directly to app internal storage
adb push llama-3.2-1b-instruct-q4_k_m.gguf /sdcard/Android/data/com.dtu.mobileagent/files/
```
The native engine automatically detects the model file, allocates the 804 MB KV cache, and runs 4-thread inference using ARM NEON vector instructions.

---

## 📊 Benchmark Alignment

The Android engine implements the exact parameters proven in our **AAQB (Autonomous Agent Quantization Benchmark)**:

| Metric | Target Specification |
| :--- | :--- |
| **Model** | Llama-3.2-1B-Instruct (GGUF Q4_K_M) |
| **Grammar Constraint** | GBNF (`tool_calling.gbnf`) |
| **Syntax Adherence** | **100.0%** (vs 72.0% unconstrained naive 4-bit) |
| **RAM Consumption** | **804 MB** (67.3% reduction vs 2,460 MB FP16) |
| **Tokens / Second** | **28.4 tok/s** (ARM NEON, 4 threads, Snapdragon 8 Gen 2) |
| **Task Success Rate** | **86.0%** |
