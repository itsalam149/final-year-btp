package com.dtu.mobileagent.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
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
import com.dtu.mobileagent.engine.AgentController
import com.dtu.mobileagent.engine.ModelGalleryManager
import com.dtu.mobileagent.voice.VoiceInputState
import com.dtu.mobileagent.voice.VoiceManager
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

enum class MainNavTab(val title: String, val shortBadge: String, val icon: ImageVector) {
    ASSISTANT("Assistant", "AI", Icons.Default.Chat),
    NOTES("Notes & Hisaab", "NOTES", Icons.Default.Description),
    REMINDERS("Reminders", "ALARMS", Icons.Default.Alarm),
    SYSTEM("Audit & Stats", "SYS", Icons.Default.Analytics)
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
    val activeReminders by controller.activeReminders.collectAsState(initial = emptyList())
    val recentAuditLogs by controller.recentAuditLogs.collectAsState(initial = emptyList())
    val telemetry by controller.telemetry.collectAsState()

    val voiceState by voiceManager?.inputState?.collectAsState() ?: remember { mutableStateOf(VoiceInputState.Idle) }
    val isListening = voiceState is VoiceInputState.Listening

    var currentTab by remember { mutableStateOf(MainNavTab.ASSISTANT) }
    var showModelMenu by remember { mutableStateOf(false) }
    var showModelGallery by remember { mutableStateOf(false) }

    val listState = rememberLazyListState()
    var inputText by remember { mutableStateOf("") }

    // Auto-scroll chat to latest message
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .size(7.dp)
                                    .clip(CircleShape)
                                    .background(EmeraldGlow)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "EdgeAgent",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                color = TextPrimary
                            )
                        }
                        val isCurrentDownloaded = galleryManager?.isModelDownloaded(uiState.activeModel) == true
                        Text(
                            text = if (isCurrentDownloaded) {
                                "100% on-device • ${uiState.activeModel.displayName.replace("-Instruct", "").replace("-Chat", "")} • ${uiState.currentTokensPerSec} tps"
                            } else {
                                "Weights not on disk • Tap GET to download"
                            },
                            fontSize = 11.sp,
                            color = if (isCurrentDownloaded) TextSecondary else Color(0xFFFBBF24)
                        )
                    }
                },
                actions = {
                    // Quick Engine Switcher Dropdown Pill
                    Box {
                        Surface(
                            onClick = { showModelMenu = true },
                            color = CardBackground,
                            shape = RoundedCornerShape(14.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle)
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Bolt,
                                    contentDescription = null,
                                    tint = EmeraldGlow,
                                    modifier = Modifier.size(12.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    text = uiState.activeModel.displayName.take(11),
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.SemiBold,
                                    color = TextPrimary
                                )
                                Icon(
                                    imageVector = Icons.Default.ArrowDropDown,
                                    contentDescription = null,
                                    tint = TextSecondary,
                                    modifier = Modifier.size(14.dp)
                                )
                            }
                        }

                        DropdownMenu(
                            expanded = showModelMenu,
                            onDismissRequest = { showModelMenu = false },
                            modifier = Modifier.background(SurfaceDark)
                        ) {
                            ModelGalleryManager.CATALOG.forEach { spec ->
                                val isSelected = uiState.activeModel.id == spec.id
                                val isDownloaded = galleryManager?.isModelDownloaded(spec) == true
                                DropdownMenuItem(
                                    text = {
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Column(modifier = Modifier.weight(1f)) {
                                                Text(
                                                    text = spec.displayName,
                                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                                    color = if (isSelected) EmeraldGlow else TextPrimary,
                                                    fontSize = 13.sp
                                                )
                                                Text(
                                                    text = if (isDownloaded) "[Ready] (${spec.sizeMb} MB)" else "[Download Required] (${spec.sizeMb} MB)",
                                                    fontSize = 10.sp,
                                                    color = if (isDownloaded) EmeraldOffline else TextMuted
                                                )
                                            }
                                            if (!isDownloaded) {
                                                Surface(
                                                    color = BlueAccent.copy(alpha = 0.2f),
                                                    shape = RoundedCornerShape(6.dp)
                                                ) {
                                                    Text(
                                                        text = "GET",
                                                        fontSize = 9.sp,
                                                        fontWeight = FontWeight.Bold,
                                                        color = BlueAccent,
                                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                                    )
                                                }
                                            }
                                        }
                                    },
                                    onClick = {
                                        showModelMenu = false
                                        if (isDownloaded) {
                                            coroutineScope.launch {
                                                controller.switchModel(spec)
                                                galleryManager?.setActiveModel(spec)
                                            }
                                        } else {
                                            showModelGallery = true
                                        }
                                    }
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.width(4.dp))

                    // Voice Output Toggle
                    IconButton(onClick = { controller.toggleVoiceOutput() }) {
                        Icon(
                            imageVector = if (uiState.isVoiceOutputEnabled) Icons.Default.VolumeUp else Icons.Default.VolumeOff,
                            contentDescription = if (uiState.isVoiceOutputEnabled) "Mute Voice" else "Enable Voice",
                            tint = if (uiState.isVoiceOutputEnabled) EmeraldGlow else TextSecondary
                        )
                    }

                    // Model Gallery Hub Button
                    IconButton(onClick = { showModelGallery = true }) {
                        Icon(
                            imageVector = Icons.Default.CloudDownload,
                            contentDescription = "Model Gallery",
                            tint = TextSecondary
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
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = SurfaceDark
                )
            )
        },
        bottomBar = {
            NavigationBar(
                containerColor = SurfaceDark,
                contentColor = TextPrimary,
                tonalElevation = 8.dp
            ) {
                MainNavTab.values().forEach { tab ->
                    val isSelected = currentTab == tab
                    NavigationBarItem(
                        selected = isSelected,
                        onClick = { currentTab = tab },
                        icon = {
                            Icon(
                                imageVector = tab.icon,
                                contentDescription = tab.title,
                                tint = if (isSelected) EmeraldGlow else TextSecondary,
                                modifier = Modifier.size(20.dp)
                            )
                        },
                        label = {
                            Text(
                                text = tab.title,
                                fontSize = 10.sp,
                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                color = if (isSelected) EmeraldGlow else TextSecondary
                            )
                        },
                        colors = NavigationBarItemDefaults.colors(
                            indicatorColor = EmeraldDark.copy(alpha = 0.25f),
                            selectedIconColor = EmeraldGlow,
                            unselectedIconColor = TextSecondary,
                            selectedTextColor = EmeraldGlow,
                            unselectedTextColor = TextSecondary
                        )
                    )
                }
            }
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        Box(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            when (currentTab) {
                MainNavTab.ASSISTANT -> {
                    AssistantChatView(
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
                        onShowModelGallery = { showModelGallery = true },
                        onQuickPrompt = { prompt ->
                            coroutineScope.launch {
                                controller.handleUserMessage(prompt)
                            }
                        }
                    )
                }
                MainNavTab.NOTES -> {
                    NotesHisaabView(
                        notes = allNotes,
                        onDeleteNote = { noteId ->
                            coroutineScope.launch {
                                controller.deleteNote(noteId)
                            }
                        }
                    )
                }
                MainNavTab.REMINDERS -> {
                    RemindersView(
                        reminders = activeReminders,
                        onMarkCompleted = { reminderId ->
                            coroutineScope.launch {
                                controller.markReminderCompleted(reminderId)
                            }
                        }
                    )
                }
                MainNavTab.SYSTEM -> {
                    SystemTelemetryView(
                        telemetry = telemetry,
                        auditLogs = recentAuditLogs,
                        onRefreshTelemetry = { controller.refreshTelemetry() }
                    )
                }
            }
        }

        // Active Confirmation Dialog
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

        // Edge Model Gallery Sheet
        if (showModelGallery && galleryManager != null) {
            ModelGallerySheet(
                galleryManager = galleryManager,
                onDismiss = { showModelGallery = false },
                onImportLocalFile = {
                    showModelGallery = false
                    onImportLocalModel()
                },
                onModelActivated = { spec ->
                    showModelGallery = false
                    coroutineScope.launch {
                        controller.switchModel(spec)
                        galleryManager.setActiveModel(spec)
                    }
                }
            )
        }
    }
}

@Composable
fun AssistantChatView(
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
    onShowModelGallery: () -> Unit,
    onQuickPrompt: (String) -> Unit
) {
    val coroutineScope = rememberCoroutineScope()

    Column(modifier = Modifier.fillMaxSize()) {
        // Hardware Performance Meter Bar
        ModelMetricsBar(
            tokensPerSec = uiState.currentTokensPerSec,
            memoryMb = uiState.currentMemoryMb
        )

        // Missing Model Notification Banner
        if (galleryManager != null && !galleryManager.isModelDownloaded(uiState.activeModel)) {
            Surface(
                color = CardBackground,
                shape = RoundedCornerShape(10.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFFFBBF24).copy(alpha = 0.4f)),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 6.dp)
                    .clickable { onShowModelGallery() }
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                        Icon(
                            imageVector = Icons.Default.CloudDownload,
                            contentDescription = null,
                            tint = Color(0xFFFBBF24),
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "${uiState.activeModel.displayName.take(16)} weights (${uiState.activeModel.sizeMb} MB) not on disk.",
                            fontSize = 11.sp,
                            color = TextPrimary
                        )
                    }
                    Text(
                        text = "DOWNLOAD",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFFBBF24)
                    )
                }
            }
        }

        // Messages List or Clean Welcome
        if (messages.isEmpty()) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp),
                contentAlignment = Alignment.Center
            ) {
                CleanWelcomeCard(
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
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(messages) { msg ->
                    CleanMessageBubble(msg)
                }

                if (uiState.isProcessing) {
                    item {
                        ProcessingIndicator()
                    }
                }
            }
        }

        // Active Voice Listening Indicator Banner
        if (isListening) {
            Surface(
                color = CardBackground,
                shape = RoundedCornerShape(12.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, EmeraldOffline),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 4.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(EmeraldGlow)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Listening... Speak now",
                        color = EmeraldGlow,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }

        // Quick Prompt Action Chips Carousel
        QuickPromptChipsRow(
            onPromptSelected = onQuickPrompt
        )

        // Bottom Input Bar
        Surface(
            color = SurfaceDark,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                TextField(
                    value = inputText,
                    onValueChange = onInputTextChanged,
                    placeholder = {
                        Text(
                            "Ask offline agent...",
                            color = TextMuted,
                            fontSize = 13.sp
                        )
                    },
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = CardBackground,
                        unfocusedContainerColor = CardBackground,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                        cursorColor = EmeraldOffline,
                        focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent
                    ),
                    shape = RoundedCornerShape(22.dp),
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 46.dp)
                )

                Spacer(modifier = Modifier.width(6.dp))

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
                        .size(42.dp)
                        .clip(CircleShape)
                        .background(if (isListening) EmeraldOffline else CardBackground)
                ) {
                    Icon(
                        imageVector = Icons.Default.Mic,
                        contentDescription = "Voice Input",
                        tint = if (isListening) Color.White else TextSecondary,
                        modifier = Modifier.size(18.dp)
                    )
                }

                Spacer(modifier = Modifier.width(4.dp))

                // Send Button
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
                        .size(42.dp)
                        .clip(CircleShape)
                        .background(if (inputText.isNotBlank()) EmeraldOffline else BorderSubtle)
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Send",
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun QuickPromptChipsRow(
    onPromptSelected: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    val prompts = listOf(
        "HISAAB" to "Notes pr hisaab likho 200 sharukh, 500 kalli, 300 haaji aur total karo",
        "SPLIT" to "Split dinner bill of 1800 with 4 friends",
        "ALARM" to "Remind me to study at 7 PM",
        "YOUTUBE" to "Open YouTube and play saas song",
        "NOTES" to "Show my notes",
        "SCHEDULE" to "Check my schedule for today",
        "KNOWLEDGE" to "What is the capital of France?"
    )

    LazyRow(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 10.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        items(prompts) { (badge, promptText) ->
            Surface(
                color = CardBackground,
                shape = RoundedCornerShape(14.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle),
                modifier = Modifier.clickable { onPromptSelected(promptText) }
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Surface(
                        color = BlueAccent.copy(alpha = 0.2f),
                        shape = RoundedCornerShape(4.dp)
                    ) {
                        Text(
                            text = badge,
                            fontSize = 8.sp,
                            fontWeight = FontWeight.Bold,
                            color = BlueAccent,
                            modifier = Modifier.padding(horizontal = 4.dp, vertical = 1.dp)
                        )
                    }
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = promptText.take(24) + if (promptText.length > 24) "..." else "",
                        fontSize = 11.sp,
                        color = TextPrimary
                    )
                }
            }
        }
    }
}

@Composable
fun NotesHisaabView(
    notes: List<NoteEntity>,
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
            .padding(horizontal = 14.dp, vertical = 8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Local Encrypted Notes & Ledgers",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Text(
                    text = "${notes.size} total items stored in SQLite Room ORM (Zero Cloud)",
                    fontSize = 11.sp,
                    color = TextSecondary
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        TextField(
            value = searchQuery,
            onValueChange = { searchQuery = it },
            placeholder = { Text("Filter notes or hisaab entries...", color = TextMuted, fontSize = 12.sp) },
            leadingIcon = { Icon(Icons.Default.Search, contentDescription = null, tint = TextSecondary, modifier = Modifier.size(16.dp)) },
            colors = TextFieldDefaults.colors(
                focusedContainerColor = CardBackground,
                unfocusedContainerColor = CardBackground,
                focusedTextColor = TextPrimary,
                unfocusedTextColor = TextPrimary,
                cursorColor = EmeraldOffline,
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent
            ),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(46.dp)
        )

        Spacer(modifier = Modifier.height(10.dp))

        if (filteredNotes.isEmpty()) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = if (searchQuery.isBlank()) "No local notes recorded yet." else "No matching notes found.",
                        color = TextSecondary,
                        fontSize = 13.sp
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Say: 'Notes pr hisaab likho 200 sharukh, 500 kalli...'",
                        color = TextMuted,
                        fontSize = 11.sp
                    )
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(filteredNotes, key = { it.id }) { note ->
                    val isHisaab = note.title.contains("Hisaab", ignoreCase = true) || note.content.contains("₹")
                    Surface(
                        color = CardBackground,
                        shape = RoundedCornerShape(12.dp),
                        border = androidx.compose.foundation.BorderStroke(1.dp, if (isHisaab) EmeraldOffline.copy(alpha = 0.4f) else BorderSubtle),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                                    Surface(
                                        color = if (isHisaab) EmeraldDark.copy(alpha = 0.3f) else BlueAccent.copy(alpha = 0.2f),
                                        shape = RoundedCornerShape(4.dp)
                                    ) {
                                        Text(
                                            text = if (isHisaab) "[HISAAB]" else "[NOTE]",
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = if (isHisaab) EmeraldGlow else BlueAccent,
                                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                        )
                                    }
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = note.title,
                                        fontSize = 13.sp,
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
                                    Icon(
                                        Icons.Default.DeleteOutline,
                                        contentDescription = "Delete Note",
                                        tint = TextMuted,
                                        modifier = Modifier.size(15.dp)
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(6.dp))

                            Text(
                                text = note.content,
                                fontSize = 12.sp,
                                color = TextSecondary,
                                lineHeight = 18.sp
                            )

                            Spacer(modifier = Modifier.height(6.dp))

                            val dateStr = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date(note.updatedAt))
                            Text(
                                text = "Saved: $dateStr • Scoped Room Storage",
                                fontSize = 10.sp,
                                color = TextMuted
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun RemindersView(
    reminders: List<ReminderEntity>,
    onMarkCompleted: (Long) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 14.dp, vertical = 8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Scheduled Alarms & Reminders",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Text(
                    text = "${reminders.size} active reminders linked to Android AlarmManager",
                    fontSize = 11.sp,
                    color = TextSecondary
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        if (reminders.isEmpty()) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "No pending reminders or alarms.",
                        color = TextSecondary,
                        fontSize = 13.sp
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Say: 'Remind me to study at 7 PM'",
                        color = TextMuted,
                        fontSize = 11.sp
                    )
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(reminders, key = { it.id }) { item ->
                    Surface(
                        color = CardBackground,
                        shape = RoundedCornerShape(12.dp),
                        border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Surface(
                                        color = EmeraldDark.copy(alpha = 0.3f),
                                        shape = RoundedCornerShape(4.dp)
                                    ) {
                                        Text(
                                            text = "[ALARM]",
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = EmeraldGlow,
                                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                        )
                                    }
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = item.title,
                                        fontSize = 13.sp,
                                        fontWeight = FontWeight.SemiBold,
                                        color = TextPrimary
                                    )
                                }
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "Due: ${item.dueTimeIso} • Priority: ${item.priority.uppercase()}",
                                    fontSize = 11.sp,
                                    color = WarningAmber
                                )
                            }

                            Button(
                                onClick = { onMarkCompleted(item.id) },
                                colors = ButtonDefaults.buttonColors(containerColor = EmeraldDark),
                                shape = RoundedCornerShape(8.dp),
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                            ) {
                                Text("Mark Done", fontSize = 11.sp, color = Color.White)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun SystemTelemetryView(
    telemetry: com.dtu.mobileagent.engine.HardwareTelemetry,
    auditLogs: List<AuditLogEntity>,
    onRefreshTelemetry: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 14.dp, vertical = 8.dp)
    ) {
        // Live Hardware Profiler Card
        HardwareTelemetryCard(telemetry = telemetry)

        Spacer(modifier = Modifier.height(10.dp))

        // Immutable Audit Log Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Immutable Security Audit Log",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Text(
                    text = "Cryptographic local ledger of autonomous tool invocations",
                    fontSize = 11.sp,
                    color = TextSecondary
                )
            }
            TextButton(onClick = onRefreshTelemetry) {
                Text("Refresh", fontSize = 11.sp, color = BlueAccent)
            }
        }

        Spacer(modifier = Modifier.height(6.dp))

        if (auditLogs.isEmpty()) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "No autonomous actions logged yet.",
                    color = TextMuted,
                    fontSize = 12.sp
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                items(auditLogs, key = { it.id }) { log ->
                    Surface(
                        color = CardBackground,
                        shape = RoundedCornerShape(8.dp),
                        border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(10.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    val statusColor = if (log.status == "EXECUTED") EmeraldGlow else WarningAmber
                                    Surface(
                                        color = statusColor.copy(alpha = 0.2f),
                                        shape = RoundedCornerShape(4.dp)
                                    ) {
                                        Text(
                                            text = "[${log.status}]",
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = statusColor,
                                            modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp)
                                        )
                                    }
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(
                                        text = log.toolName,
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.SemiBold,
                                        color = TextPrimary
                                    )
                                }
                                val timeStr = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(log.timestamp))
                                Text(
                                    text = timeStr,
                                    fontSize = 10.sp,
                                    color = TextMuted
                                )
                            }
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = log.executionDetails,
                                fontSize = 11.sp,
                                color = TextSecondary,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                            if (log.parametersJson.isNotBlank()) {
                                Spacer(modifier = Modifier.height(2.dp))
                                Text(
                                    text = "Args: ${log.parametersJson}",
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

@Composable
fun CleanWelcomeCard(
    activeModel: String,
    onPromptSelected: (String) -> Unit
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(
            text = "EdgeAgent",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = TextPrimary
        )

        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "DTU Dept of CSE • GP159 | $activeModel",
            fontSize = 12.sp,
            color = TextSecondary
        )

        Spacer(modifier = Modifier.height(18.dp))

        data class WelcomePrompt(val badge: String, val text: String)

        val prompts = listOf(
            WelcomePrompt("[HISAAB]", "Notes pr hisaab likho 200 sharukh, 500 kalli, 300 haaji aur total karo"),
            WelcomePrompt("[ALARM]", "Remind me to study at 7 PM"),
            WelcomePrompt("[MEDIA]", "Open YouTube and play saas song"),
            WelcomePrompt("[CALENDAR]", "What's my schedule for today?")
        )

        prompts.forEach { item ->
            Surface(
                color = CardBackground,
                shape = RoundedCornerShape(12.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
                    .clickable { onPromptSelected(item.text) }
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 11.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Surface(
                        color = BlueAccent.copy(alpha = 0.2f),
                        shape = RoundedCornerShape(4.dp)
                    ) {
                        Text(
                            text = item.badge,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold,
                            color = BlueAccent,
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = item.text,
                        fontSize = 12.sp,
                        color = TextPrimary
                    )
                }
            }
        }
    }
}

@Composable
fun CleanMessageBubble(message: ChatMessageEntity) {
    val isUser = message.sender == "user"
    val bubbleColor = if (isUser) Color(0xFF1D4ED8) else CardBackground
    val alignment = if (isUser) Alignment.End else Alignment.Start

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment
    ) {
        Surface(
            color = bubbleColor,
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (isUser) 16.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 16.dp
            ),
            modifier = Modifier.widthIn(max = 300.dp)
        ) {
            Text(
                text = message.content,
                color = TextPrimary,
                fontSize = 14.sp,
                lineHeight = 20.sp,
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 9.dp)
            )
        }

        if (!message.toolName.isNullOrBlank()) {
            Spacer(modifier = Modifier.height(3.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "[Executed: ${message.toolName}]",
                    fontSize = 10.sp,
                    color = EmeraldGlow
                )
            }
        }
    }
}

@Composable
fun ProcessingIndicator() {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(vertical = 4.dp)
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(13.dp),
            strokeWidth = 1.5.dp,
            color = EmeraldOffline
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = "Thinking...",
            fontSize = 11.sp,
            color = TextSecondary
        )
    }
}
