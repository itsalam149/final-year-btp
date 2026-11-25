package com.dtu.mobileagent.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
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
import com.dtu.mobileagent.voice.VoiceInputState
import com.dtu.mobileagent.voice.VoiceManager
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    controller: AgentController,
    voiceManager: VoiceManager? = null,
    modifier: Modifier = Modifier
) {
    val coroutineScope = rememberCoroutineScope()
    val uiState by controller.uiState.collectAsState()
    val messages by controller.messages.collectAsState(initial = emptyList())
    val voiceState by voiceManager?.inputState?.collectAsState() ?: remember { mutableStateOf(VoiceInputState.Idle) }
    val isListening = voiceState is VoiceInputState.Listening

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
                        Text(
                            text = "Offline Mobile AI Agent",
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary
                        )
                        Text(
                            text = "DTU GP159 • Llama-3.2-1B (Q4_K_M GBNF)",
                            fontSize = 11.sp,
                            color = TextSecondary
                        )
                    }
                },
                actions = {
                    // Offline Text-to-Speech Toggle
                    IconButton(onClick = { controller.toggleVoiceOutput() }) {
                        Icon(
                            imageVector = if (uiState.isVoiceOutputEnabled) Icons.Default.VolumeUp else Icons.Default.VolumeOff,
                            contentDescription = "Toggle Spoken Replies",
                            tint = if (uiState.isVoiceOutputEnabled) EmeraldGlow else TextSecondary
                        )
                    }

                    // Clear Chat History
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
            // Security badge & Performance Metrics Bar
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 8.dp)
            ) {
                OfflineSecurityBadge()
                Spacer(modifier = Modifier.height(6.dp))
                ModelMetricsBar(
                    tokensPerSec = uiState.currentTokensPerSec,
                    memoryMb = uiState.currentMemoryMb
                )
            }

            // Message stream
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                items(messages) { msg ->
                    MessageBubble(msg)
                }

                if (uiState.isProcessing) {
                    item {
                        AgentProcessingIndicator()
                    }
                }
            }

            // Input bar
            Surface(
                color = SurfaceDark,
                modifier = Modifier.fillMaxWidth(),
                tonalElevation = 8.dp
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        placeholder = {
                            Text(
                                "Ask your offline agent...",
                                color = TextMuted,
                                fontSize = 14.sp
                            )
                        },
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = CardBackground,
                            unfocusedContainerColor = CardBackground,
                            focusedTextColor = TextPrimary,
                            unfocusedTextColor = TextPrimary,
                            cursorColor = DtuRedLight,
                            focusedIndicatorColor = Color.Transparent,
                            unfocusedIndicatorColor = Color.Transparent
                        ),
                        shape = RoundedCornerShape(24.dp),
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(min = 48.dp)
                    )

                    Spacer(modifier = Modifier.width(6.dp))

                    // Offline Voice Input Button (Microphone)
                    IconButton(
                        onClick = {
                            if (isListening) {
                                voiceManager?.stopListening()
                            } else {
                                voiceManager?.startListening()
                            }
                        },
                        modifier = Modifier
                            .size(44.dp)
                            .clip(CircleShape)
                            .background(if (isListening) EmeraldOffline else CardBackground)
                            .border(1.dp, if (isListening) EmeraldGlow else BorderSubtle, CircleShape)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Mic,
                            contentDescription = "Voice Input",
                            tint = if (isListening) Color.White else TextSecondary,
                            modifier = Modifier.size(20.dp)
                        )
                    }

                    Spacer(modifier = Modifier.width(6.dp))

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
                            .size(44.dp)
                            .clip(CircleShape)
                            .background(if (inputText.isNotBlank()) DtuRedPrimary else BorderSubtle)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Send,
                            contentDescription = "Send",
                            tint = Color.White,
                            modifier = Modifier.size(20.dp)
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
    }
}

@Composable
fun MessageBubble(message: ChatMessageEntity) {
    val isUser = message.sender == "user"
    val bubbleColor = if (isUser) DtuRedPrimary else CardBackground
    val alignment = if (isUser) Alignment.End else Alignment.Start

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment
    ) {
        if (!isUser && message.thought.isNotBlank()) {
            // Display model reasoning thought
            Surface(
                color = SurfaceDark,
                shape = RoundedCornerShape(8.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle),
                modifier = Modifier
                    .padding(bottom = 4.dp)
                    .widthIn(max = 300.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Psychology,
                        contentDescription = "Reasoning",
                        tint = BlueAccent,
                        modifier = Modifier.size(13.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Thought: ${message.thought}",
                        fontSize = 11.sp,
                        color = TextSecondary,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }

        // Tool Badge if action was taken
        if (!message.toolName.isNullOrBlank()) {
            Surface(
                color = EmeraldDark.copy(alpha = 0.2f),
                shape = RoundedCornerShape(6.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, EmeraldOffline.copy(alpha = 0.5f)),
                modifier = Modifier.padding(bottom = 4.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Build,
                        contentDescription = "Tool",
                        tint = EmeraldOffline,
                        modifier = Modifier.size(11.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Tool Executed: ${message.toolName}",
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = EmeraldGlow
                    )
                }
            }
        }

        // Core Message Content Bubble
        Surface(
            color = bubbleColor,
            shape = RoundedCornerShape(
                topStart = 14.dp,
                topEnd = 14.dp,
                bottomStart = if (isUser) 14.dp else 2.dp,
                bottomEnd = if (isUser) 2.dp else 14.dp
            ),
            modifier = Modifier.widthIn(max = 310.dp)
        ) {
            Text(
                text = message.content,
                color = TextPrimary,
                fontSize = 14.sp,
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)
            )
        }
    }
}

@Composable
fun AgentProcessingIndicator() {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(vertical = 6.dp)
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(16.dp),
            strokeWidth = 2.dp,
            color = DtuRedLight
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "Local LLM reasoning with GBNF grammar...",
            fontSize = 12.sp,
            color = TextSecondary,
            fontFamily = FontFamily.Monospace
        )
    }
}
