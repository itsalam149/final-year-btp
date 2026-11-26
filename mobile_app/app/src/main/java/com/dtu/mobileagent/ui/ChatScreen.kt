package com.dtu.mobileagent.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dtu.mobileagent.data.local.entities.ChatMessageEntity
import com.dtu.mobileagent.engine.AgentController
import com.dtu.mobileagent.engine.ModelGalleryManager
import com.dtu.mobileagent.voice.VoiceInputState
import com.dtu.mobileagent.voice.VoiceManager
import kotlinx.coroutines.launch

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
    val voiceState by voiceManager?.inputState?.collectAsState() ?: remember { mutableStateOf(VoiceInputState.Idle) }
    val isListening = voiceState is VoiceInputState.Listening

    var showModelMenu by remember { mutableStateOf(false) }
    var showModelGallery by remember { mutableStateOf(false) }

    val listState = rememberLazyListState()
    var inputText by remember { mutableStateOf("") }

    // Scroll to bottom when message arrives
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
                                text = "Offline Mobile Agent",
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
                                "Weights not downloaded • Tap ⬇ to download"
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
                                                    text = if (isDownloaded) "✓ Downloaded (${spec.sizeMb} MB)" else "⬇ Download Required (${spec.sizeMb} MB)",
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
                                            // Model not downloaded! Open gallery sheet to download it
                                            showModelGallery = true
                                        }
                                    }
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.width(4.dp))

                    // Voice Output (Speaker) Toggle
                    IconButton(onClick = { controller.toggleVoiceOutput() }) {
                        Icon(
                            imageVector = if (uiState.isVoiceOutputEnabled) Icons.Default.VolumeUp else Icons.Default.VolumeOff,
                            contentDescription = if (uiState.isVoiceOutputEnabled) "Mute Voice" else "Enable Voice",
                            tint = if (uiState.isVoiceOutputEnabled) EmeraldGlow else TextSecondary
                        )
                    }

                    // Model Hub Button
                    IconButton(onClick = { showModelGallery = true }) {
                        Icon(
                            imageVector = Icons.Default.CloudDownload,
                            contentDescription = "Model Gallery",
                            tint = TextSecondary
                        )
                    }

                    // Clear History
                    IconButton(onClick = { coroutineScope.launch { controller.clearHistory() } }) {
                        Icon(
                            imageVector = Icons.Default.DeleteOutline,
                            contentDescription = "Clear History",
                            tint = TextSecondary
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = SurfaceDark
                )
            )
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Download Model Notice Banner
            if (galleryManager != null && !galleryManager.isModelDownloaded(uiState.activeModel)) {
                Surface(
                    color = CardBackground,
                    shape = RoundedCornerShape(10.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFFFBBF24).copy(alpha = 0.4f)),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 14.dp, vertical = 6.dp)
                        .clickable { showModelGallery = true }
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

            // Message Feed or Clean Welcome
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
                        onPromptSelected = { prompt ->
                            coroutineScope.launch {
                                controller.handleUserMessage(prompt)
                            }
                        }
                    )
                }
            } else {
                LazyColumn(
                    state = listState,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(horizontal = 14.dp, vertical = 6.dp),
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

            // Input Bar
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
                        onValueChange = { inputText = it },
                        placeholder = {
                            Text(
                                "Type a request...",
                                color = TextMuted,
                                fontSize = 14.sp
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

                    // Voice Input Button
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
                                inputText = ""
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
fun CleanWelcomeCard(
    activeModel: String,
    onPromptSelected: (String) -> Unit
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(
            text = "Offline Mobile Agent",
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

        data class WelcomePrompt(val icon: String, val text: String)

        val prompts = listOf(
            WelcomePrompt("⏰", "Remind me to study at 7 PM"),
            WelcomePrompt("🗓️", "What's my schedule for today?"),
            WelcomePrompt("📝", "Take a note: BTP review on Monday"),
            WelcomePrompt("📱", "Open WhatsApp")
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
                Text(
                    text = "${item.icon}  ${item.text}",
                    fontSize = 13.sp,
                    color = TextPrimary,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 11.dp)
                )
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
        // Message Content Bubble
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

        // Clean tool badge if action was performed
        if (!message.toolName.isNullOrBlank()) {
            Spacer(modifier = Modifier.height(3.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "✓ Executed: ${message.toolName}",
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
