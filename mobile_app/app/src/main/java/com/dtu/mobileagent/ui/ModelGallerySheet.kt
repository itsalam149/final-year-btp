package com.dtu.mobileagent.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.automirrored.filled.HelpOutline
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dtu.mobileagent.engine.ModelDownloadProgress
import com.dtu.mobileagent.engine.ModelGalleryManager
import com.dtu.mobileagent.engine.ModelSpec
import kotlinx.coroutines.launch

@Composable
fun EdgeModelGalleryView(
    galleryManager: ModelGalleryManager,
    onBack: () -> Unit,
    onImportLocalFile: () -> Unit,
    onModelActivated: (ModelSpec) -> Unit,
    onStartChat: () -> Unit,
    modifier: Modifier = Modifier
) {
    val coroutineScope = rememberCoroutineScope()
    val activeModel by galleryManager.activeModel.collectAsState()
    val downloadStates by galleryManager.downloadStates.collectAsState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(GoogleBackground)
    ) {
        // Top App Bar
        Surface(
            color = GoogleSurface,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onBack) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = TextPrimary
                    )
                }
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "On-Device Models",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
            }
        }

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Hero Header matching Screenshot 5
            item {
                Spacer(modifier = Modifier.height(16.dp))
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    EdgeIconBadge(
                        icon = Icons.AutoMirrored.Filled.Chat,
                        badgeColor = GoogleRedBadge,
                        size = 64,
                        iconSize = 32
                    )

                    Spacer(modifier = Modifier.height(14.dp))

                    Text(
                        text = "AI Chat & Models",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )

                    Spacer(modifier = Modifier.height(6.dp))

                    Text(
                        text = "Chat with on-device large language models",
                        fontSize = 13.sp,
                        color = TextSecondary
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Description,
                                contentDescription = null,
                                tint = GoogleBlue,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "Zero Cloud Spec",
                                fontSize = 12.sp,
                                color = GoogleBlue
                            )
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Code,
                                contentDescription = null,
                                tint = GoogleBlue,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "ARM NEON GBNF",
                                fontSize = 12.sp,
                                color = GoogleBlue
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    Text(
                        text = "${ModelGalleryManager.CATALOG.size} models available",
                        fontSize = 12.sp,
                        color = TextMuted
                    )
                }

                Spacer(modifier = Modifier.height(20.dp))

                Text(
                    text = "Recommended models",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )
            }

            // Model Cards matching Screenshot 5
            items(ModelGalleryManager.CATALOG) { spec ->
                val isDownloaded = galleryManager.isModelDownloaded(spec)
                val isActive = activeModel.id == spec.id
                val downloadProgress = downloadStates[spec.id]

                EdgeModelCardItem(
                    spec = spec,
                    isDownloaded = isDownloaded,
                    isActive = isActive,
                    downloadProgress = downloadProgress,
                    onActivate = {
                        coroutineScope.launch {
                            galleryManager.setActiveModel(spec)
                            onModelActivated(spec)
                        }
                    },
                    onDownload = {
                        galleryManager.startDownload(spec)
                    },
                    onCancelDownload = {
                        galleryManager.cancelDownload(spec)
                    },
                    onStartChat = onStartChat
                )
            }

            // Import .GGUF option
            item {
                Spacer(modifier = Modifier.height(4.dp))
                OutlinedButton(
                    onClick = onImportLocalFile,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp),
                    shape = RoundedCornerShape(24.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = GoogleBlue),
                    border = androidx.compose.foundation.BorderStroke(1.dp, GoogleBorder)
                ) {
                    Icon(
                        imageVector = Icons.Default.FileOpen,
                        contentDescription = "Import",
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Import .GGUF from Storage",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
}

@Composable
fun EdgeModelCardItem(
    spec: ModelSpec,
    isDownloaded: Boolean,
    isActive: Boolean,
    downloadProgress: ModelDownloadProgress?,
    onActivate: () -> Unit,
    onDownload: () -> Unit,
    onCancelDownload: () -> Unit,
    onStartChat: () -> Unit
) {
    val isDownloading = downloadProgress?.isDownloading == true
    val animatedProgress by animateFloatAsState(
        targetValue = downloadProgress?.progressPct ?: 0f,
        label = "DownloadProgress"
    )
    var isExpanded by remember { mutableStateOf(false) }

    Surface(
        color = GoogleCard,
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(18.dp)
        ) {
            // Badge tag row (e.g. "Best overall" or "Fastest TTFT")
            if (spec.id == "gemma-2-2b") {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(bottom = 6.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Star,
                        contentDescription = null,
                        tint = GoogleYellow,
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Best overall",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium,
                        color = TextSecondary
                    )
                }
            } else if (spec.id == "llama-3.2-1b") {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(bottom = 6.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Bolt,
                        contentDescription = null,
                        tint = GoogleGreen,
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Fastest TTFT • Low Memory",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium,
                        color = TextSecondary
                    )
                }
            }

            // Title and expand chevron
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { isExpanded = !isExpanded },
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = spec.displayName,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Icon(
                    imageVector = if (isExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                    contentDescription = null,
                    tint = TextSecondary,
                    modifier = Modifier.size(20.dp)
                )
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Specs with Info icon: (?) size, RAM
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.HelpOutline,
                    contentDescription = null,
                    tint = TextSecondary,
                    modifier = Modifier.size(13.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "${spec.sizeMb} MB • ~${spec.ramMb} MB RAM • ${spec.precision}",
                    fontSize = 12.sp,
                    color = TextSecondary
                )
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Link: Learn more and see model licence
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.OpenInNew,
                    contentDescription = null,
                    tint = GoogleBlue,
                    modifier = Modifier.size(12.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "Learn more and see model license",
                    fontSize = 12.sp,
                    color = GoogleBlue
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Description
            Text(
                text = spec.description,
                fontSize = 12.sp,
                color = TextSecondary,
                lineHeight = 17.sp
            )

            // Expanded technical details
            AnimatedVisibility(visible = isExpanded) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 10.dp)
                ) {
                    HorizontalDivider(color = GoogleDivider, thickness = 1.dp)
                    Spacer(modifier = Modifier.height(10.dp))
                    Text("Architecture: 4-bit ARM NEON SIMD", fontSize = 11.sp, color = TextMuted)
                    Text("Inference Kernel: llama.cpp GBNF Grammar Engine", fontSize = 11.sp, color = TextMuted)
                    Text("File Name: ${spec.filename}", fontSize = 11.sp, color = TextMuted)
                    Spacer(modifier = Modifier.height(6.dp))
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            // Action Button Area
            when {
                isDownloading -> {
                    val dlMb = (downloadProgress?.downloadedBytes ?: 0L) / (1024 * 1024)
                    val totMb = (downloadProgress?.totalBytes ?: 0L) / (1024 * 1024)
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Downloading: $dlMb MB / $totMb MB (${((downloadProgress?.progressPct ?: 0f) * 100).toInt()}%)",
                                fontSize = 12.sp,
                                color = GoogleBlue,
                                fontWeight = FontWeight.Medium
                            )
                            TextButton(onClick = onCancelDownload) {
                                Text("Cancel", color = GoogleRed, fontSize = 12.sp)
                            }
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        LinearProgressIndicator(
                            progress = { animatedProgress },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(6.dp)
                                .clip(RoundedCornerShape(3.dp)),
                            color = GoogleBlue,
                            trackColor = GoogleCardElevated
                        )
                    }
                }

                isDownloaded -> {
                    if (isActive) {
                        Button(
                            onClick = onStartChat,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(46.dp),
                            shape = RoundedCornerShape(24.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = GoogleGreen)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Check,
                                contentDescription = null,
                                modifier = Modifier.size(16.dp),
                                tint = Color.White
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Active Engine • Start Chat",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White
                            )
                        }
                    } else {
                        Button(
                            onClick = onActivate,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(46.dp),
                            shape = RoundedCornerShape(24.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = GoogleCardElevated)
                        ) {
                            Text(
                                text = "Select Active Engine",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Medium,
                                color = TextPrimary
                            )
                        }
                    }
                }

                else -> {
                    Button(
                        onClick = onDownload,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(46.dp),
                        shape = RoundedCornerShape(24.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = GoogleBlue)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Download,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp),
                            tint = Color.White
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Download GGUF (${spec.sizeMb} MB)",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }
                }
            }
        }
    }
}
