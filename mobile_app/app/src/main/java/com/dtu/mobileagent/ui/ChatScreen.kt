package com.dtu.mobileagent.ui

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.automirrored.filled.FormatListBulleted
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dtu.mobileagent.data.local.entities.AuditLogEntity
import com.dtu.mobileagent.data.local.entities.ChatMessageEntity
import com.dtu.mobileagent.data.local.entities.NoteEntity
import com.dtu.mobileagent.data.local.entities.ReminderEntity
import com.dtu.mobileagent.data.local.entities.UserMemoryEntity
import com.dtu.mobileagent.data.local.entities.DocumentEntity
import com.dtu.mobileagent.engine.AgentController
import com.dtu.mobileagent.engine.ModelGalleryManager
import com.dtu.mobileagent.voice.VoiceInputState
import com.dtu.mobileagent.voice.VoiceManager
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

enum class AppScreen {
    HOME_GALLERY,
    AI_CHAT,
    AGENT_SKILLS,
    MODELS_CATALOG,
    NOTES_LEDGER,
    KNOWLEDGE_VAULT,
    REMINDERS_NOTIFICATIONS,
    HARDWARE_PROFILER,
    SECURITY_AUDIT,
    SETTINGS
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    controller: AgentController,
    voiceManager: VoiceManager? = null,
    galleryManager: ModelGalleryManager? = null,
    onImportLocalModel: () -> Unit = {},
    onRequestRecordAudio: ((() -> Unit) -> Unit) = { it() },
    modifier: Modifier = Modifier
) {
    val coroutineScope = rememberCoroutineScope()
    val uiState by controller.uiState.collectAsState()
    val messages by controller.messages.collectAsState(initial = emptyList())
    val allNotes by controller.allNotes.collectAsState(initial = emptyList())
    val allMemories by controller.allMemories.collectAsState(initial = emptyList())
    val allDocuments by controller.allDocuments.collectAsState(initial = emptyList())
    val activeReminders by controller.activeReminders.collectAsState(initial = emptyList())
    val recentAuditLogs by controller.recentAuditLogs.collectAsState(initial = emptyList())
    val telemetry by controller.telemetry.collectAsState()

    val voiceState by voiceManager?.inputState?.collectAsState() ?: remember { mutableStateOf(VoiceInputState.Idle) }
    val isListening = voiceState is VoiceInputState.Listening

    var currentScreen by remember { mutableStateOf(AppScreen.HOME_GALLERY) }
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)

    val listState = rememberLazyListState()
    var inputText by remember { mutableStateOf("") }

    // Intercept back navigation
    BackHandler(enabled = drawerState.isOpen || currentScreen != AppScreen.HOME_GALLERY) {
        if (drawerState.isOpen) {
            coroutineScope.launch { drawerState.close() }
        } else {
            currentScreen = AppScreen.HOME_GALLERY
        }
    }

    // Auto-scroll chat to bottom
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(
                drawerContainerColor = GoogleSurface,
                modifier = Modifier.width(320.dp)
            ) {
                EdgeAgentDrawerContent(
                    onNavigate = { screen ->
                        currentScreen = screen
                        coroutineScope.launch { drawerState.close() }
                    }
                )
            }
        }
    ) {
        Box(
            modifier = modifier
                .fillMaxSize()
                .background(GoogleBackground)
        ) {
            when (currentScreen) {
                AppScreen.HOME_GALLERY -> {
                    EdgeAgentHomeScreen(
                        activeModelName = uiState.activeModel.displayName,
                        isModelDownloaded = galleryManager?.isModelDownloaded(uiState.activeModel) == true,
                        onOpenDrawer = { coroutineScope.launch { drawerState.open() } },
                        onNavigate = { screen -> currentScreen = screen }
                    )
                }

                AppScreen.AI_CHAT -> {
                    EdgeAgentChatScreen(
                        isAgentMode = false,
                        controller = controller,
                        uiState = uiState,
                        messages = messages,
                        isListening = isListening,
                        voiceManager = voiceManager,
                        galleryManager = galleryManager,
                        onRequestRecordAudio = onRequestRecordAudio,
                        listState = listState,
                        inputText = inputText,
                        onInputTextChanged = { inputText = it },
                        onBack = { currentScreen = AppScreen.HOME_GALLERY },
                        onNavigateToModels = { currentScreen = AppScreen.MODELS_CATALOG },
                        onQuickPrompt = { prompt ->
                            coroutineScope.launch {
                                controller.handleUserMessage(prompt)
                            }
                        }
                    )
                }

                AppScreen.AGENT_SKILLS -> {
                    EdgeAgentChatScreen(
                        isAgentMode = true,
                        controller = controller,
                        uiState = uiState,
                        messages = messages,
                        isListening = isListening,
                        voiceManager = voiceManager,
                        galleryManager = galleryManager,
                        onRequestRecordAudio = onRequestRecordAudio,
                        listState = listState,
                        inputText = inputText,
                        onInputTextChanged = { inputText = it },
                        onBack = { currentScreen = AppScreen.HOME_GALLERY },
                        onNavigateToModels = { currentScreen = AppScreen.MODELS_CATALOG },
                        onQuickPrompt = { prompt ->
                            coroutineScope.launch {
                                controller.handleUserMessage(prompt)
                            }
                        }
                    )
                }

                AppScreen.MODELS_CATALOG -> {
                    if (galleryManager != null) {
                        EdgeModelGalleryView(
                            galleryManager = galleryManager,
                            isHybridEnabled = uiState.isHybridRoutingEnabled,
                            onToggleHybrid = { enable -> controller.setHybridRoutingEnabled(enable) },
                            onBack = { currentScreen = AppScreen.HOME_GALLERY },
                            onImportLocalFile = onImportLocalModel,
                            onModelActivated = { spec ->
                                coroutineScope.launch {
                                    controller.setHybridRoutingEnabled(false)
                                    controller.switchModel(spec)
                                    galleryManager.setActiveModel(spec)
                                }
                            },
                            onStartChat = { currentScreen = AppScreen.AI_CHAT }
                        )
                    }
                }

                AppScreen.NOTES_LEDGER -> {
                    EdgeNotesLedgerScreen(
                        notes = allNotes,
                        onBack = { currentScreen = AppScreen.HOME_GALLERY },
                        onDeleteNote = { noteId ->
                            coroutineScope.launch {
                                controller.deleteNote(noteId)
                            }
                        }
                    )
                }

                AppScreen.KNOWLEDGE_VAULT -> {
                    EdgeKnowledgeVaultScreen(
                        memories = allMemories,
                        documents = allDocuments,
                        onBack = { currentScreen = AppScreen.HOME_GALLERY },
                        onDeleteMemory = { memoryId ->
                            coroutineScope.launch {
                                controller.deleteMemoryById(memoryId)
                            }
                        },
                        onClearMemories = {
                            coroutineScope.launch {
                                controller.clearMemories()
                            }
                        },
                        onRescanDocuments = {
                            coroutineScope.launch {
                                controller.rescanDocuments()
                            }
                        }
                    )
                }

                AppScreen.REMINDERS_NOTIFICATIONS -> {
                    EdgeRemindersScreen(
                        reminders = activeReminders,
                        onBack = { currentScreen = AppScreen.HOME_GALLERY },
                        onMarkCompleted = { reminderId ->
                            coroutineScope.launch {
                                controller.markReminderCompleted(reminderId)
                            }
                        }
                    )
                }

                AppScreen.HARDWARE_PROFILER -> {
                    EdgeHardwareProfilerScreen(
                        telemetry = telemetry,
                        onBack = { currentScreen = AppScreen.HOME_GALLERY },
                        onRefresh = { controller.refreshTelemetry() }
                    )
                }

                AppScreen.SECURITY_AUDIT -> {
                    EdgeSecurityAuditScreen(
                        auditLogs = recentAuditLogs,
                        onBack = { currentScreen = AppScreen.HOME_GALLERY }
                    )
                }

                AppScreen.SETTINGS -> {
                    EdgeSettingsScreen(
                        onBack = { currentScreen = AppScreen.HOME_GALLERY }
                    )
                }
            }

            // Two-Tier Security Gate Modal Dialog
            uiState.pendingConfirmation?.let { pending ->
                ConfirmationDialog(
                    pending = pending,
                    onConfirm = { approved ->
                        coroutineScope.launch {
                            controller.confirmPendingAction(approved)
                        }
                    }
                )
            }
        }
    }
}

/**
 * EdgeAgent Home Screen
 */
@Composable
fun EdgeAgentHomeScreen(
    activeModelName: String,
    isModelDownloaded: Boolean,
    onOpenDrawer: () -> Unit,
    onNavigate: (AppScreen) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(EdgeBackground)
    ) {
        // Top App Bar
        Surface(
            color = EdgeSurface,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = onOpenDrawer) {
                        Icon(
                            imageVector = Icons.Default.Menu,
                            contentDescription = "Menu",
                            tint = TextPrimary
                        )
                    }

                    Spacer(modifier = Modifier.width(6.dp))

                    EdgeDotsLogo(dotSize = 7f, spacing = 3.5f)

                    Spacer(modifier = Modifier.width(10.dp))

                    Text(
                        text = "EdgeAgent",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                }

                Surface(
                    color = if (isModelDownloaded) EdgeGreenContainer else EdgeAmberContainer,
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.clickable { onNavigate(AppScreen.MODELS_CATALOG) }
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(6.dp)
                                .clip(CircleShape)
                                .background(if (isModelDownloaded) EdgeGreen else EdgeAmber)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = activeModelName.replace("-Instruct", "").take(10),
                            fontSize = 11.sp,
                            color = if (isModelDownloaded) EdgeGreen else EdgeAmber,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }
            }
        }

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 18.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Hero Title
            item {
                Spacer(modifier = Modifier.height(14.dp))
                Text(
                    text = "EdgeAgent",
                    fontSize = 26.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Autonomous on-device AI agent for privacy-first task execution. All inference runs locally on ARM — no cloud, no telemetry.",
                    fontSize = 14.sp,
                    color = TextSecondary,
                    lineHeight = 20.sp
                )
            }

            // Feature Banner: Voice capability
            item {
                Spacer(modifier = Modifier.height(8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.GraphicEq,
                        contentDescription = null,
                        tint = EdgeBlue,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Voice commands now live",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                }

                Spacer(modifier = Modifier.height(6.dp))

                Text(
                    text = "Speak to your agent — 100% offline, no internet required. Voice in, voice out.",
                    fontSize = 13.sp,
                    color = TextSecondary,
                    lineHeight = 18.sp
                )

                Spacer(modifier = Modifier.height(14.dp))

                // Feature Card 1: Autonomous Agent
                EdgeFeatureCard(
                    title = "Autonomous Agent",
                    subtitle = "Issue natural-language commands. The agent plans, selects tools, and executes tasks — all on-device.",
                    icon = Icons.Default.SmartToy,
                    badgeColor = EdgeBlueBadge,
                    isSquircle = true,
                    onClick = { onNavigate(AppScreen.AGENT_SKILLS) }
                )

                Spacer(modifier = Modifier.height(10.dp))

                // Feature Card 2: On-Device Chat
                EdgeFeatureCard(
                    title = "On-Device Chat",
                    subtitle = "Converse with a quantized LLM — GGUF Q4_K_M inference on ARM NEON, zero cloud dependency.",
                    icon = Icons.AutoMirrored.Filled.Chat,
                    badgeColor = EdgeRedBadge,
                    isSquircle = false,
                    onClick = { onNavigate(AppScreen.AI_CHAT) }
                )
            }

            // Section: Capabilities
            item {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "What can EdgeAgent do?",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(10.dp))
            }

            // 2-Column Grid of actual EdgeAgent features
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Chat
                    EdgeGridCard(
                        badgeText = "3 Models",
                        title = "Chat",
                        description = "Talk to a local quantized LLM",
                        icon = Icons.AutoMirrored.Filled.Chat,
                        badgeColor = EdgeRedBadge,
                        isSquircle = false,
                        onClick = { onNavigate(AppScreen.AI_CHAT) },
                        modifier = Modifier.weight(1f)
                    )

                    // Task Execution
                    EdgeGridCard(
                        badgeText = "30+ Tools",
                        title = "Task Agent",
                        description = "Execute reminders, notes, app launches",
                        icon = Icons.Default.RocketLaunch,
                        badgeColor = EdgeGreenBadge,
                        isSquircle = false,
                        onClick = { onNavigate(AppScreen.AGENT_SKILLS) },
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Notes
                    EdgeGridCard(
                        badgeText = "SQLite Room",
                        title = "Notes Ledger",
                        description = "Encrypted local notes — zero cloud",
                        icon = Icons.Default.Description,
                        badgeColor = EdgeBlueBadge,
                        isSquircle = true,
                        onClick = { onNavigate(AppScreen.NOTES_LEDGER) },
                        modifier = Modifier.weight(1f)
                    )

                    // Hardware Monitor
                    EdgeGridCard(
                        badgeText = "ARM NEON",
                        title = "HW Profiler",
                        description = "Live tok/s, RAM, battery, thermal",
                        icon = Icons.Default.Speed,
                        badgeColor = EdgeAmberBadge,
                        isSquircle = true,
                        onClick = { onNavigate(AppScreen.HARDWARE_PROFILER) },
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Reminders
                    EdgeGridCard(
                        badgeText = "AlarmManager",
                        title = "Reminders",
                        description = "Set and manage offline alarms",
                        icon = Icons.Default.Notifications,
                        badgeColor = EdgeRedBadge,
                        isSquircle = true,
                        onClick = { onNavigate(AppScreen.REMINDERS_NOTIFICATIONS) },
                        modifier = Modifier.weight(1f)
                    )

                    // Security Gate
                    EdgeGridCard(
                        badgeText = "Two-Tier Gate",
                        title = "Security Log",
                        description = "Audit autonomous action execution",
                        icon = Icons.Default.Shield,
                        badgeColor = EdgeAmberBadge,
                        isSquircle = true,
                        onClick = { onNavigate(AppScreen.SECURITY_AUDIT) },
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Model Gallery
                    EdgeGridCard(
                        badgeText = "GGUF Q4_K_M",
                        title = "Model Gallery",
                        description = "Download and switch quantized LLMs",
                        icon = Icons.Default.GridView,
                        badgeColor = EdgeBlueBadge,
                        isSquircle = true,
                        onClick = { onNavigate(AppScreen.MODELS_CATALOG) },
                        modifier = Modifier.weight(1f)
                    )

                    // Settings
                    EdgeGridCard(
                        badgeText = "Config",
                        title = "Settings",
                        description = "Agent behaviour and system config",
                        icon = Icons.Default.Settings,
                        badgeColor = EdgePurpleBadge,
                        isSquircle = true,
                        onClick = { onNavigate(AppScreen.SETTINGS) },
                        modifier = Modifier.weight(1f)
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                Row(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    // Memory & Document Vault
                    EdgeGridCard(
                        badgeText = "Local RAG Store",
                        title = "Memory & Document Vault",
                        description = "Personal knowledge memory & indexed device files",
                        icon = Icons.Default.Storage,
                        badgeColor = EdgeGreenBadge,
                        isSquircle = true,
                        onClick = { onNavigate(AppScreen.KNOWLEDGE_VAULT) },
                        modifier = Modifier.fillMaxWidth()
                    )
                }

                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
}

/**
 * Slide-out Navigation Drawer
 */
@Composable
fun EdgeAgentDrawerContent(
    onNavigate: (AppScreen) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(GoogleSurface)
            .padding(16.dp)
    ) {
        Spacer(modifier = Modifier.height(24.dp))

        Row(verticalAlignment = Alignment.CenterVertically) {
            EdgeDotsLogo(dotSize = 8f, spacing = 4f)
            Spacer(modifier = Modifier.width(10.dp))
            Column {
                Text(
                    text = "EdgeAgent",
                    fontSize = 17.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Text(
                    text = "DTU BTP GP159 • ARM On-Device",
                    fontSize = 11.sp,
                    color = TextSecondary
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        HorizontalDivider(color = GoogleDivider, thickness = 1.dp)

        Spacer(modifier = Modifier.height(16.dp))

        // Grid of Drawer items matching Screenshot 3
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Settings Card
            Surface(
                onClick = { onNavigate(AppScreen.SETTINGS) },
                color = GoogleCard,
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier
                    .weight(1f)
                    .height(120.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(14.dp),
                    verticalArrangement = Arrangement.SpaceBetween
                ) {
                    EdgeIconBadge(
                        icon = Icons.Default.Settings,
                        badgeColor = GoogleBlueBadge,
                        size = 36,
                        iconSize = 18,
                        isSquircle = true
                    )
                    Column {
                        Text("Settings", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        Text("Manage settings", fontSize = 11.sp, color = TextSecondary, maxLines = 1)
                    }
                }
            }

            // Models Card
            Surface(
                onClick = { onNavigate(AppScreen.MODELS_CATALOG) },
                color = GoogleCard,
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier
                    .weight(1f)
                    .height(120.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(14.dp),
                    verticalArrangement = Arrangement.SpaceBetween
                ) {
                    EdgeIconBadge(
                        icon = Icons.AutoMirrored.Filled.FormatListBulleted,
                        badgeColor = GoogleGreenBadge,
                        size = 36,
                        iconSize = 18,
                        isSquircle = true
                    )
                    Column {
                        Text("Models", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        Text("Browse & try", fontSize = 11.sp, color = TextSecondary, maxLines = 1)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Notifications / Reminders Card
            Surface(
                onClick = { onNavigate(AppScreen.REMINDERS_NOTIFICATIONS) },
                color = GoogleCard,
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier
                    .weight(1f)
                    .height(120.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(14.dp),
                    verticalArrangement = Arrangement.SpaceBetween
                ) {
                    EdgeIconBadge(
                        icon = Icons.Default.Notifications,
                        badgeColor = GoogleRedBadge,
                        size = 36,
                        iconSize = 18,
                        isSquircle = true
                    )
                    Column {
                        Text("Notifications", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        Text("Scheduled alarms", fontSize = 11.sp, color = TextSecondary, maxLines = 1)
                    }
                }
            }

            // Notes & Hisaab Card
            Surface(
                onClick = { onNavigate(AppScreen.NOTES_LEDGER) },
                color = GoogleCard,
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier
                    .weight(1f)
                    .height(120.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(14.dp),
                    verticalArrangement = Arrangement.SpaceBetween
                ) {
                    EdgeIconBadge(
                        icon = Icons.Default.Description,
                        badgeColor = GoogleYellowBadge,
                        size = 36,
                        iconSize = 18,
                        isSquircle = true
                    )
                    Column {
                        Text("Notes", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        Text("Encrypted store", fontSize = 11.sp, color = TextSecondary, maxLines = 1)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Direct Navigation items
        NavigationDrawerItem(
            label = { Text("AI Chat", color = TextPrimary, fontSize = 14.sp) },
            icon = { Icon(Icons.AutoMirrored.Filled.Chat, contentDescription = null, tint = GoogleRed) },
            selected = false,
            onClick = { onNavigate(AppScreen.AI_CHAT) },
            colors = NavigationDrawerItemDefaults.colors(unselectedContainerColor = Color.Transparent)
        )

        NavigationDrawerItem(
            label = { Text("Agent Skills", color = TextPrimary, fontSize = 14.sp) },
            icon = { Icon(Icons.Default.RocketLaunch, contentDescription = null, tint = GoogleGreen) },
            selected = false,
            onClick = { onNavigate(AppScreen.AGENT_SKILLS) },
            colors = NavigationDrawerItemDefaults.colors(unselectedContainerColor = Color.Transparent)
        )

        NavigationDrawerItem(
            label = { Text("Memory & Documents", color = TextPrimary, fontSize = 14.sp) },
            icon = { Icon(Icons.Default.Storage, contentDescription = null, tint = GoogleGreen) },
            selected = false,
            onClick = { onNavigate(AppScreen.KNOWLEDGE_VAULT) },
            colors = NavigationDrawerItemDefaults.colors(unselectedContainerColor = Color.Transparent)
        )

        NavigationDrawerItem(
            label = { Text("Hardware Telemetry", color = TextPrimary, fontSize = 14.sp) },
            icon = { Icon(Icons.Default.Speed, contentDescription = null, tint = GoogleYellow) },
            selected = false,
            onClick = { onNavigate(AppScreen.HARDWARE_PROFILER) },
            colors = NavigationDrawerItemDefaults.colors(unselectedContainerColor = Color.Transparent)
        )

        NavigationDrawerItem(
            label = { Text("Security Audit Logs", color = TextPrimary, fontSize = 14.sp) },
            icon = { Icon(Icons.Default.Shield, contentDescription = null, tint = GoogleBlue) },
            selected = false,
            onClick = { onNavigate(AppScreen.SECURITY_AUDIT) },
            colors = NavigationDrawerItemDefaults.colors(unselectedContainerColor = Color.Transparent)
        )

        Spacer(modifier = Modifier.weight(1f))

        Text(
            text = "DTU CSE • Minor Project GP159\nEdgeAgent v1.1.0 • ARM NEON SIMD",
            fontSize = 11.sp,
            color = TextMuted,
            lineHeight = 16.sp
        )
    }
}

/**
 * On-Device Chat Screen
 */
@Composable
fun EdgeAgentChatScreen(
    isAgentMode: Boolean,
    controller: AgentController,
    uiState: com.dtu.mobileagent.engine.AgentUiState,
    messages: List<ChatMessageEntity>,
    isListening: Boolean,
    voiceManager: VoiceManager?,
    galleryManager: ModelGalleryManager?,
    onRequestRecordAudio: ((() -> Unit) -> Unit),
    listState: androidx.compose.foundation.lazy.LazyListState,
    inputText: String,
    onInputTextChanged: (String) -> Unit,
    onBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    onQuickPrompt: (String) -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    val isModelDownloaded = galleryManager?.isModelDownloaded(uiState.activeModel) == true

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(GoogleBackground)
    ) {
        // Top App Bar with back navigation
        Surface(
            color = GoogleSurface,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = TextPrimary
                        )
                    }

                    Spacer(modifier = Modifier.width(4.dp))

                    Column {
                        Text(
                            text = if (isAgentMode) "Agent Skills" else "AI Chat",
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary
                        )
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.clickable { onNavigateToModels() }
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(if (uiState.isHybridRoutingEnabled || isModelDownloaded) GoogleGreen else GoogleYellow)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = if (uiState.isHybridRoutingEnabled) "⚡ Auto: " + uiState.activeModel.displayName.replace("-Instruct", "").replace("-Chat", "")
                                       else uiState.activeModel.displayName.replace("-Instruct", "").replace("-Chat", ""),
                                fontSize = 11.sp,
                                color = if (uiState.isHybridRoutingEnabled) GoogleGreen else TextSecondary,
                                fontWeight = if (uiState.isHybridRoutingEnabled) FontWeight.Bold else FontWeight.Normal
                            )
                            Icon(
                                imageVector = Icons.Default.ArrowDropDown,
                                contentDescription = null,
                                tint = TextSecondary,
                                modifier = Modifier.size(14.dp)
                            )
                        }
                    }
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    // Voice Output Toggle
                    IconButton(onClick = { controller.toggleVoiceOutput() }) {
                        Icon(
                            imageVector = if (uiState.isVoiceOutputEnabled) Icons.AutoMirrored.Filled.VolumeUp else Icons.AutoMirrored.Filled.VolumeOff,
                            contentDescription = "Voice Output",
                            tint = if (uiState.isVoiceOutputEnabled) GoogleGreen else TextSecondary
                        )
                    }

                    // Clear Chat History Button
                    IconButton(onClick = { coroutineScope.launch { controller.clearHistory() } }) {
                        Icon(
                            imageVector = Icons.Default.DeleteOutline,
                            contentDescription = "Clear Chat",
                            tint = TextSecondary
                        )
                    }
                }
            }
        }

        // Hardware Performance Meter Bar
        ModelMetricsBar(
            tokensPerSec = uiState.currentTokensPerSec,
            memoryMb = uiState.currentMemoryMb
        )

        // Missing Model Notification Banner
        if (!isModelDownloaded) {
            Surface(
                color = GoogleCard,
                shape = RoundedCornerShape(12.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, GoogleYellow.copy(alpha = 0.5f)),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 6.dp)
                    .clickable { onNavigateToModels() }
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                        Icon(
                            imageVector = Icons.Default.Download,
                            contentDescription = null,
                            tint = GoogleYellow,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "${uiState.activeModel.displayName} weights (${uiState.activeModel.sizeMb} MB) not on disk.",
                            fontSize = 11.sp,
                            color = TextPrimary
                        )
                    }
                    Text(
                        text = "DOWNLOAD",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = GoogleYellow
                    )
                }
            }
        }

        // Chat Message Thread
        if (messages.isEmpty()) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp),
                contentAlignment = Alignment.Center
            ) {
                EdgeAgentChatHero(
                    isAgentMode = isAgentMode,
                    activeModel = uiState.activeModel.displayName,
                    onPromptSelected = onQuickPrompt
                )
            }
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 4.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                items(messages) { msg ->
                    EdgeMessageBubble(msg)
                }

                if (uiState.isProcessing) {
                    item {
                        EdgeProcessingIndicator()
                    }
                }
            }
        }

        // Listening Indicator
        if (isListening) {
            Surface(
                color = GoogleCard,
                shape = RoundedCornerShape(16.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, GoogleGreen),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 4.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(GoogleGreen)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Listening... Speak your command",
                        color = GoogleGreen,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }

        // Quick Suggestion Chips Row
        EdgeQuickPromptChipsRow(
            onPromptSelected = onQuickPrompt
        )

        // Bottom Input Bar in floating pill shape
        Surface(
            color = GoogleBackground,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Input Container Pill
                Surface(
                    color = GoogleCard,
                    shape = RoundedCornerShape(26.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, GoogleBorder),
                    modifier = Modifier.weight(1f)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        TextField(
                            value = inputText,
                            onValueChange = onInputTextChanged,
                            placeholder = {
                                Text(
                                    if (isAgentMode) "Command device tasks..." else "Ask on-device LLM...",
                                    color = TextMuted,
                                    fontSize = 13.sp
                                )
                            },
                            colors = TextFieldDefaults.colors(
                                focusedContainerColor = Color.Transparent,
                                unfocusedContainerColor = Color.Transparent,
                                focusedTextColor = TextPrimary,
                                unfocusedTextColor = TextPrimary,
                                cursorColor = GoogleBlue,
                                focusedIndicatorColor = Color.Transparent,
                                unfocusedIndicatorColor = Color.Transparent
                            ),
                            modifier = Modifier
                                .weight(1f)
                                .heightIn(min = 44.dp)
                        )

                        // Voice Mic Button
                        IconButton(
                            onClick = {
                                if (isListening) {
                                    voiceManager?.stopListening()
                                } else {
                                    onRequestRecordAudio {
                                        voiceManager?.startListening()
                                    }
                                }
                            },
                            modifier = Modifier
                                .size(36.dp)
                                .clip(CircleShape)
                                .background(if (isListening) GoogleGreen else GoogleCardElevated)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Mic,
                                contentDescription = "Voice Input",
                                tint = if (isListening) Color.White else TextSecondary,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.width(8.dp))

                // Send Button Pill
                IconButton(
                    onClick = {
                        if (inputText.isNotBlank() && !uiState.isProcessing) {
                            val text = inputText
                            onInputTextChanged("")
                            coroutineScope.launch {
                                controller.handleUserMessage(text)
                            }
                        }
                    },
                    enabled = inputText.isNotBlank() && !uiState.isProcessing,
                    modifier = Modifier
                        .size(46.dp)
                        .clip(CircleShape)
                        .background(if (inputText.isNotBlank()) GoogleBlue else GoogleCard)
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Send",
                        tint = if (inputText.isNotBlank()) Color.White else TextMuted,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun EdgeAgentChatHero(
    isAgentMode: Boolean,
    activeModel: String,
    onPromptSelected: (String) -> Unit
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.fillMaxWidth()
    ) {
        EdgeIconBadge(
            icon = if (isAgentMode) Icons.Default.RocketLaunch else Icons.AutoMirrored.Filled.Chat,
            badgeColor = if (isAgentMode) GoogleGreenBadge else GoogleRedBadge,
            size = 56,
            iconSize = 28
        )

        Spacer(modifier = Modifier.height(14.dp))

        Text(
            text = if (isAgentMode) "Agent Skills" else "AI Chat",
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = TextPrimary
        )

        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = if (isAgentMode) "Autonomous on-device agent with tool calling" else "Chat with on-device large language models",
            fontSize = 13.sp,
            color = TextSecondary
        )

        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "Active: $activeModel",
            fontSize = 11.sp,
            color = TextMuted
        )

        Spacer(modifier = Modifier.height(8.dp))

        OfflineSecurityBadge()

        Spacer(modifier = Modifier.height(20.dp))

        val prompts = if (isAgentMode) {
            listOf(
                "Notes pr hisaab likho 200 sharukh, 500 kalli, 300 haaji aur total karo",
                "Split dinner bill of 1800 with 4 friends",
                "Remind me to study at 7 PM",
                "Open YouTube and play saas song"
            )
        } else {
            listOf(
                "Explain how 4-bit quantization reduces memory bandwidth",
                "What is the difference between perplexity and reasoning accuracy?",
                "Draft a technical summary of on-device LLM inference",
                "How does GBNF grammar constrain autoregressive decoding?"
            )
        }

        prompts.forEach { prompt ->
            Surface(
                onClick = { onPromptSelected(prompt) },
                color = GoogleCard,
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.ChevronRight,
                        contentDescription = null,
                        tint = GoogleBlue,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = prompt,
                        fontSize = 12.sp,
                        color = TextPrimary
                    )
                }
            }
        }
    }
}

@Composable
fun EdgeMessageBubble(message: ChatMessageEntity) {
    val isUser = message.sender == "user"
    val bubbleColor = if (isUser) GoogleCardElevated else GoogleCard
    val alignment = if (isUser) Alignment.End else Alignment.Start

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment
    ) {
        Surface(
            color = bubbleColor,
            shape = RoundedCornerShape(
                topStart = 18.dp,
                topEnd = 18.dp,
                bottomStart = if (isUser) 18.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 18.dp
            ),
            modifier = Modifier.widthIn(max = 310.dp)
        ) {
            Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
                if (!isUser) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(bottom = 4.dp)
                    ) {
                        EdgeDotsLogo(dotSize = 4f, spacing = 2f)
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "EdgeAgent",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = GoogleBlue
                        )
                    }

                    if (message.thought.isNotBlank()) {
                        Surface(
                            color = GoogleSurface,
                            shape = RoundedCornerShape(6.dp),
                            modifier = Modifier.padding(bottom = 6.dp)
                        ) {
                            Text(
                                text = message.thought,
                                fontSize = 10.sp,
                                color = TextSecondary,
                                fontWeight = FontWeight.Medium,
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                            )
                        }
                    }
                }

                Text(
                    text = message.content,
                    color = TextPrimary,
                    fontSize = 13.sp,
                    lineHeight = 19.sp
                )
            }
        }

        if (!message.toolName.isNullOrBlank()) {
            Spacer(modifier = Modifier.height(4.dp))
            Surface(
                color = GoogleGreenContainer,
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.padding(start = 4.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = GoogleGreen,
                        modifier = Modifier.size(11.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = when (message.toolName) {
                            "open_app" -> "Action: App Launched"
                            "manage_notes" -> "Action: Notes App"
                            "draft_message" -> "Action: Messaging Intent"
                            "create_reminder" -> "Action: Reminder Scheduled"
                            "search_calendar" -> "Action: Calendar Check"
                            "search_local_docs" -> "Action: Local Document Search"
                            else -> "Action: ${message.toolName}"
                        },
                        fontSize = 10.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = GoogleGreen
                    )
                }
            }
        }
    }
}

@Composable
fun EdgeProcessingIndicator() {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(vertical = 6.dp)
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(14.dp),
            strokeWidth = 1.8.dp,
            color = GoogleBlue
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "Generating via ARM NEON...",
            fontSize = 11.sp,
            color = TextSecondary
        )
    }
}

@Composable
fun EdgeQuickPromptChipsRow(
    onPromptSelected: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val prompts = listOf(
        "FILES" to "What files do I have?",
        "MEMORY" to "What do you remember about me?",
        "HISAAB" to "Notes pr hisaab likho 200 sharukh, 500 kalli, 300 haaji aur total karo",
        "SPLIT" to "Split dinner bill of 1800 with 4 friends",
        "ALARM" to "Remind me to study at 7 PM",
        "YOUTUBE" to "Open YouTube and play saas song",
        "NOTES" to "Show my notes",
        "KNOWLEDGE" to "What is the capital of France?"
    )

    LazyRow(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(prompts) { (badge, promptText) ->
            Surface(
                onClick = { onPromptSelected(promptText) },
                color = GoogleCard,
                shape = RoundedCornerShape(16.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Surface(
                        color = GoogleBlueContainer,
                        shape = RoundedCornerShape(6.dp)
                    ) {
                        Text(
                            text = badge,
                            fontSize = 8.sp,
                            fontWeight = FontWeight.Bold,
                            color = GoogleBlue,
                            modifier = Modifier.padding(horizontal = 4.dp, vertical = 1.dp)
                        )
                    }
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = promptText.take(22) + if (promptText.length > 22) "..." else "",
                        fontSize = 11.sp,
                        color = TextPrimary
                    )
                }
            }
        }
    }
}

/**
 * Local Notes & Ledgers Screen
 */
@Composable
fun EdgeNotesLedgerScreen(
    notes: List<NoteEntity>,
    onBack: () -> Unit,
    onDeleteNote: (Long) -> Unit
) {
    var searchQuery by remember { mutableStateOf("") }
    val filteredNotes = remember(notes, searchQuery) {
        if (searchQuery.isBlank()) notes
        else notes.filter { it.title.contains(searchQuery, ignoreCase = true) || it.content.contains(searchQuery, ignoreCase = true) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(GoogleBackground)
    ) {
        Surface(color = GoogleSurface, modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = TextPrimary)
                }
                Spacer(modifier = Modifier.width(4.dp))
                Text("Local Notes & Ledgers", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
            }
        }

        Column(modifier = Modifier.padding(16.dp)) {
            // Search Pill
            TextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                placeholder = { Text("Filter notes or hisaab entries...", color = TextMuted, fontSize = 12.sp) },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = null, tint = TextSecondary, modifier = Modifier.size(18.dp)) },
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = GoogleCard,
                    unfocusedContainerColor = GoogleCard,
                    focusedTextColor = TextPrimary,
                    unfocusedTextColor = TextPrimary,
                    cursorColor = GoogleBlue,
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent
                ),
                shape = RoundedCornerShape(22.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
            )

            Spacer(modifier = Modifier.height(14.dp))

            if (filteredNotes.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(bottom = 60.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        EdgeIconBadge(icon = Icons.Default.Description, badgeColor = GoogleYellowBadge, size = 52, iconSize = 26, isSquircle = true)
                        Spacer(modifier = Modifier.height(12.dp))
                        Text("No local notes recorded yet.", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text("Say: 'Notes pr hisaab likho 200 sharukh...'", color = TextSecondary, fontSize = 12.sp)
                    }
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    items(filteredNotes, key = { it.id }) { note ->
                        val isHisaab = note.title.contains("Hisaab", ignoreCase = true) || note.content.contains("₹")
                        Surface(
                            color = GoogleCard,
                            shape = RoundedCornerShape(18.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(modifier = Modifier.padding(14.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                                        Surface(
                                            color = if (isHisaab) GoogleGreenContainer else GoogleBlueContainer,
                                            shape = RoundedCornerShape(6.dp)
                                        ) {
                                            Text(
                                                text = if (isHisaab) "HISAAB" else "NOTE",
                                                fontSize = 9.sp,
                                                fontWeight = FontWeight.Bold,
                                                color = if (isHisaab) GoogleGreen else GoogleBlue,
                                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                            )
                                        }
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = note.title,
                                            fontSize = 14.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = TextPrimary,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                    }
                                    IconButton(
                                        onClick = { onDeleteNote(note.id) },
                                        modifier = Modifier.size(24.dp)
                                    ) {
                                        Icon(Icons.Default.DeleteOutline, contentDescription = "Delete", tint = TextMuted, modifier = Modifier.size(16.dp))
                                    }
                                }

                                Spacer(modifier = Modifier.height(8.dp))

                                Text(text = note.content, fontSize = 12.sp, color = TextSecondary, lineHeight = 18.sp)

                                Spacer(modifier = Modifier.height(8.dp))

                                val dateStr = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date(note.updatedAt))
                                Text("Saved: $dateStr • SQLite Room ORM", fontSize = 10.sp, color = TextMuted)
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Scheduled Reminders Screen
 */
@Composable
fun EdgeRemindersScreen(
    reminders: List<ReminderEntity>,
    onBack: () -> Unit,
    onMarkCompleted: (Long) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(GoogleBackground)
    ) {
        Surface(color = GoogleSurface, modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = TextPrimary)
                }
                Spacer(modifier = Modifier.width(4.dp))
                Text("Notifications & Reminders", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
            }
        }

        Column(modifier = Modifier.padding(16.dp)) {
            if (reminders.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        EdgeIconBadge(icon = Icons.Default.Notifications, badgeColor = GoogleRedBadge, size = 52, iconSize = 26, isSquircle = true)
                        Spacer(modifier = Modifier.height(12.dp))
                        Text("No pending reminders or alarms.", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text("Say: 'Remind me to study at 7 PM'", color = TextSecondary, fontSize = 12.sp)
                    }
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    items(reminders, key = { it.id }) { item ->
                        Surface(
                            color = GoogleCard,
                            shape = RoundedCornerShape(18.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(14.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(item.title, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = TextPrimary)
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text("Due: ${item.dueTimeIso} • ${item.priority.uppercase()}", fontSize = 11.sp, color = GoogleYellow)
                                }
                                Button(
                                    onClick = { onMarkCompleted(item.id) },
                                    colors = ButtonDefaults.buttonColors(containerColor = GoogleGreenContainer),
                                    shape = RoundedCornerShape(16.dp),
                                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
                                ) {
                                    Text("Done", fontSize = 11.sp, color = GoogleGreen, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Hardware Profiler Screen
 */
@Composable
fun EdgeHardwareProfilerScreen(
    telemetry: com.dtu.mobileagent.engine.HardwareTelemetry,
    onBack: () -> Unit,
    onRefresh: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(GoogleBackground)
    ) {
        Surface(color = GoogleSurface, modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = TextPrimary)
                }
                Spacer(modifier = Modifier.width(4.dp))
                Text("Hardware Telemetry", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            HardwareTelemetryCard(telemetry = telemetry)

            Surface(
                color = GoogleCard,
                shape = RoundedCornerShape(20.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("ARM NEON Kernel Architecture", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("• Target ABI: arm64-v8a + fp16 + dotprod", fontSize = 12.sp, color = TextSecondary)
                    Text("• Quantization Scheme: Q4_K_M Block Super-Weighting", fontSize = 12.sp, color = TextSecondary)
                    Text("• Grammar Engine: Deterministic GBNF AST Logit Masking", fontSize = 12.sp, color = TextSecondary)
                    Text("• Memory Hierarchy: L1/L2 Cache Tile Optimized Matrix Multiply", fontSize = 12.sp, color = TextSecondary)
                }
            }

            Button(
                onClick = onRefresh,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(46.dp),
                shape = RoundedCornerShape(24.dp),
                colors = ButtonDefaults.buttonColors(containerColor = GoogleBlue)
            ) {
                Text("Refresh Telemetry Counters", fontWeight = FontWeight.Bold, color = Color.White)
            }
        }
    }
}

/**
 * Security Audit Screen
 */
@Composable
fun EdgeSecurityAuditScreen(
    auditLogs: List<AuditLogEntity>,
    onBack: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(GoogleBackground)
    ) {
        Surface(color = GoogleSurface, modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = TextPrimary)
                }
                Spacer(modifier = Modifier.width(4.dp))
                Text("Security & Audit Logs", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
            }
        }

        Column(modifier = Modifier.padding(16.dp)) {
            OfflineSecurityBadge()

            Spacer(modifier = Modifier.height(14.dp))

            if (auditLogs.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No autonomous actions logged yet.", color = TextMuted, fontSize = 13.sp)
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(auditLogs, key = { it.id }) { log ->
                        Surface(
                            color = GoogleCard,
                            shape = RoundedCornerShape(14.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(log.toolName, fontSize = 13.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                                    val timeStr = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(log.timestamp))
                                    Text(timeStr, fontSize = 10.sp, color = TextMuted)
                                }
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(log.executionDetails, fontSize = 11.sp, color = TextSecondary)
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Application Settings Screen
 */
@Composable
fun EdgeSettingsScreen(
    onBack: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(GoogleBackground)
    ) {
        Surface(color = GoogleSurface, modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = TextPrimary)
                }
                Spacer(modifier = Modifier.width(4.dp))
                Text("Settings", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
            }
        }

        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Surface(color = GoogleCard, shape = RoundedCornerShape(18.dp), modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("About EdgeAgent", fontSize = 15.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                    Spacer(modifier = Modifier.height(4.dp))
                    Text("EdgeAgent: Autonomous On-Device Agent Architecture for Privacy-First ARM-Based Task Execution and Mobile Communication", fontSize = 12.sp, color = TextSecondary)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Delhi Technological University • Department of CSE", fontSize = 11.sp, color = TextMuted)
                    Text("Minor Project Session 2026-2027 • Group GP159", fontSize = 11.sp, color = TextMuted)
                }
            }

            Surface(color = GoogleCard, shape = RoundedCornerShape(18.dp), modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Privacy & Security Architecture", fontSize = 15.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                    Spacer(modifier = Modifier.height(4.dp))
                    Text("• Zero cloud egress: INTERNET permission omitted from manifest", fontSize = 12.sp, color = TextSecondary)
                    Text("• Local SQLite Room ORM with AES-256 encrypted storage", fontSize = 12.sp, color = TextSecondary)
                    Text("• Two-Tier Security Gate for all destructive system tools", fontSize = 12.sp, color = TextSecondary)
                }
            }
        }
    }
}

/**
 * On-Device Knowledge, Memory & Document Vault Screen
 */
@Composable
fun EdgeKnowledgeVaultScreen(
    memories: List<UserMemoryEntity>,
    documents: List<DocumentEntity>,
    onBack: () -> Unit,
    onDeleteMemory: (Long) -> Unit,
    onClearMemories: () -> Unit,
    onRescanDocuments: () -> Unit
) {
    var selectedTab by remember { mutableStateOf(0) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(GoogleBackground)
    ) {
        // Top App Bar
        Surface(color = GoogleSurface, modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = TextPrimary)
                }
                Spacer(modifier = Modifier.width(4.dp))
                Column {
                    Text("Memory & Document Vault", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                    Text("Persistent On-Device Knowledge & Local RAG", fontSize = 11.sp, color = TextSecondary)
                }
            }
        }

        // Tab Selector Row
        TabRow(
            selectedTabIndex = selectedTab,
            containerColor = GoogleSurface,
            contentColor = GoogleBlue
        ) {
            Tab(
                selected = selectedTab == 0,
                onClick = { selectedTab = 0 },
                text = { Text("Memories (${memories.size})", fontWeight = FontWeight.SemiBold, fontSize = 13.sp) }
            )
            Tab(
                selected = selectedTab == 1,
                onClick = { selectedTab = 1 },
                text = { Text("Indexed Files (${documents.size})", fontWeight = FontWeight.SemiBold, fontSize = 13.sp) }
            )
        }

        if (selectedTab == 0) {
            // Memories Tab Content
            Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("Stored Personal Facts", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                    if (memories.isNotEmpty()) {
                        TextButton(onClick = onClearMemories) {
                            Text("Clear All", color = GoogleRed, fontSize = 12.sp)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                if (memories.isEmpty()) {
                    Box(modifier = Modifier.fillMaxSize().padding(bottom = 60.dp), contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            EdgeIconBadge(icon = Icons.Default.Info, badgeColor = GoogleGreenBadge, size = 52, iconSize = 26, isSquircle = true)
                            Spacer(modifier = Modifier.height(12.dp))
                            Text("No persistent memories stored yet.", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                            Spacer(modifier = Modifier.height(4.dp))
                            Text("Tell EdgeAgent in chat: 'Remember that my car is DL4C1234'", color = TextSecondary, fontSize = 12.sp)
                        }
                    }
                } else {
                    LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        items(memories, key = { it.id }) { mem ->
                            Surface(
                                color = GoogleCard,
                                shape = RoundedCornerShape(16.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Row(
                                    modifier = Modifier.padding(14.dp),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column(modifier = Modifier.weight(1f)) {
                                        Surface(
                                            color = when (mem.category.lowercase()) {
                                                "project" -> GoogleBlueContainer
                                                "personal" -> GoogleGreenContainer
                                                else -> GoogleYellowBadge
                                            },
                                            shape = RoundedCornerShape(6.dp)
                                        ) {
                                            Text(
                                                text = mem.category.uppercase(),
                                                fontSize = 9.sp,
                                                fontWeight = FontWeight.Bold,
                                                color = when (mem.category.lowercase()) {
                                                    "project" -> GoogleBlue
                                                    "personal" -> GoogleGreen
                                                    else -> GoogleYellow
                                                },
                                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                            )
                                        }
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Text(text = mem.fact, fontSize = 13.sp, color = TextPrimary, fontWeight = FontWeight.Medium)
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(text = "Tag: #${mem.keyTag}", fontSize = 11.sp, color = TextMuted)
                                    }

                                    IconButton(onClick = { onDeleteMemory(mem.id) }) {
                                        Icon(imageVector = Icons.Default.Delete, contentDescription = "Delete", tint = TextMuted, modifier = Modifier.size(20.dp))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Documents Tab Content
            Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text("On-Device Index", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = TextPrimary)
                        Text("${documents.size} local files indexed for offline RAG", fontSize = 11.sp, color = TextSecondary)
                    }

                    Button(
                        onClick = onRescanDocuments,
                        colors = ButtonDefaults.buttonColors(containerColor = GoogleBlue),
                        shape = RoundedCornerShape(12.dp),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                    ) {
                        Icon(imageVector = Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Rescan Files", fontSize = 12.sp)
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                if (documents.isEmpty()) {
                    Box(modifier = Modifier.fillMaxSize().padding(bottom = 60.dp), contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            EdgeIconBadge(icon = Icons.Default.Folder, badgeColor = GoogleYellowBadge, size = 52, iconSize = 26, isSquircle = true)
                            Spacer(modifier = Modifier.height(12.dp))
                            Text("No indexed files found.", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                            Spacer(modifier = Modifier.height(4.dp))
                            Text("Tap 'Rescan Files' above to index documents from your phone.", color = TextSecondary, fontSize = 12.sp)
                        }
                    }
                } else {
                    LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        items(documents, key = { it.id }) { doc ->
                            Surface(
                                color = GoogleCard,
                                shape = RoundedCornerShape(16.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Column(modifier = Modifier.padding(14.dp)) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Surface(
                                            color = GoogleYellowBadge,
                                            shape = RoundedCornerShape(6.dp)
                                        ) {
                                            Text(
                                                text = doc.filename.substringAfterLast('.', "DOC").uppercase(),
                                                fontSize = 9.sp,
                                                fontWeight = FontWeight.Bold,
                                                color = GoogleYellow,
                                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                            )
                                        }
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = doc.filename,
                                            fontSize = 13.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = TextPrimary,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis,
                                            modifier = Modifier.weight(1f)
                                        )
                                    }

                                    Spacer(modifier = Modifier.height(6.dp))
                                    Text(
                                        text = doc.summary.ifEmpty { doc.content.take(120) },
                                        fontSize = 12.sp,
                                        color = TextSecondary,
                                        lineHeight = 16.sp
                                    )
                                    Spacer(modifier = Modifier.height(6.dp))
                                    Text(
                                        text = doc.filepath,
                                        fontSize = 10.sp,
                                        color = TextMuted,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
