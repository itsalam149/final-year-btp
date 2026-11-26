package com.dtu.mobileagent.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dtu.mobileagent.engine.ModelDownloadProgress
import com.dtu.mobileagent.engine.ModelGalleryManager
import com.dtu.mobileagent.engine.ModelSpec
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelGallerySheet(
    galleryManager: ModelGalleryManager,
    onDismiss: () -> Unit,
    onImportLocalFile: () -> Unit,
    onModelActivated: (ModelSpec) -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    val activeModel by galleryManager.activeModel.collectAsState()
    val downloadStates by galleryManager.downloadStates.collectAsState()

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = SurfaceDark,
        dragHandle = { BottomSheetDefaults.DragHandle(color = TextSecondary.copy(alpha = 0.4f)) }
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp)
                .padding(bottom = 32.dp)
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(EmeraldOffline)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "On-Device Model Gallery",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary
                        )
                    }
                    Text(
                        text = "Switch or download offline LLM engines • 100% private",
                        fontSize = 12.sp,
                        color = TextSecondary
                    )
                }
                IconButton(onClick = onDismiss) {
                    Icon(imageVector = Icons.Default.Close, contentDescription = "Close", tint = TextSecondary)
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                items(ModelGalleryManager.CATALOG) { spec ->
                    val isDownloaded = galleryManager.isModelDownloaded(spec)
                    val isActive = activeModel.id == spec.id
                    val downloadProgress = downloadStates[spec.id]

                    ModelCardItem(
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
                        }
                    )
                }

                item {
                    Spacer(modifier = Modifier.height(6.dp))
                    OutlinedButton(
                        onClick = onImportLocalFile,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = BlueAccent),
                        border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle)
                    ) {
                        Icon(imageVector = Icons.Default.FileOpen, contentDescription = "Import GGUF", modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Import .GGUF from Storage", fontSize = 13.sp, fontWeight = FontWeight.Medium)
                    }
                }
            }
        }
    }
}

@Composable
fun ModelCardItem(
    spec: ModelSpec,
    isDownloaded: Boolean,
    isActive: Boolean,
    downloadProgress: ModelDownloadProgress?,
    onActivate: () -> Unit,
    onDownload: () -> Unit,
    onCancelDownload: () -> Unit
) {
    val isDownloading = downloadProgress?.isDownloading == true
    val animatedProgress by animateFloatAsState(
        targetValue = downloadProgress?.progressPct ?: 0f,
        label = "DownloadProgress"
    )

    // Visual model identity mapping
    val (icon, iconColor) = when {
        spec.id.contains("gemma") -> Pair(Icons.Default.AutoAwesome, Color(0xFF60A5FA))
        spec.id.contains("3b") -> Pair(Icons.Default.Psychology, Color(0xFFA78BFA))
        spec.id.contains("1b") -> Pair(Icons.Default.Bolt, Color(0xFF38BDF8))
        spec.id.contains("qwen") -> Pair(Icons.Default.Speed, Color(0xFF34D399))
        else -> Pair(Icons.Default.ChatBubbleOutline, Color(0xFFFBBF24))
    }

    Surface(
        color = CardBackground,
        shape = RoundedCornerShape(14.dp),
        border = androidx.compose.foundation.BorderStroke(
            width = if (isActive) 1.5.dp else 1.dp,
            color = if (isActive) EmeraldOffline else BorderSubtle
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            // Header with Engine Icon & Active Badge
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f)
                ) {
                    Box(
                        modifier = Modifier
                            .size(34.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(iconColor.copy(alpha = 0.15f))
                            .border(1.dp, iconColor.copy(alpha = 0.3f), RoundedCornerShape(8.dp)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = icon,
                            contentDescription = null,
                            tint = iconColor,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                    Spacer(modifier = Modifier.width(10.dp))
                    Column {
                        Text(
                            text = spec.displayName,
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary
                        )
                        Text(
                            text = "${spec.precision} • ${spec.sizeMb} MB • ~${spec.ramMb} MB RAM",
                            fontSize = 11.sp,
                            color = TextSecondary
                        )
                    }
                }

                if (isActive) {
                    Surface(
                        color = EmeraldDark.copy(alpha = 0.35f),
                        shape = RoundedCornerShape(8.dp),
                        border = androidx.compose.foundation.BorderStroke(1.dp, EmeraldOffline.copy(alpha = 0.5f))
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(EmeraldGlow)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "ACTIVE",
                                color = EmeraldGlow,
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = spec.description,
                fontSize = 12.sp,
                color = TextSecondary,
                lineHeight = 16.sp
            )

            Spacer(modifier = Modifier.height(12.dp))

            if (isDownloading) {
                // Live Download Progress Bar & Statistics
                Column(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        val mbDownloaded = (downloadProgress?.downloadedBytes ?: 0L).toDouble() / (1024.0 * 1024.0)
                        val pctInt = ((downloadProgress?.progressPct ?: 0f) * 100).toInt()

                        Text(
                            text = "Downloading: ${String.format("%.1f", mbDownloaded)} MB / ${spec.sizeMb} MB ($pctInt%)",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            color = BlueAccent,
                            fontFamily = FontFamily.Monospace
                        )

                        TextButton(
                            onClick = onCancelDownload,
                            contentPadding = PaddingValues(horizontal = 6.dp, vertical = 2.dp)
                        ) {
                            Icon(imageVector = Icons.Default.Close, contentDescription = "Cancel", tint = DtuRedLight, modifier = Modifier.size(13.dp))
                            Spacer(modifier = Modifier.width(2.dp))
                            Text("Cancel", color = DtuRedLight, fontSize = 11.sp)
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    LinearProgressIndicator(
                        progress = { animatedProgress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(5.dp)
                            .clip(RoundedCornerShape(3.dp)),
                        color = EmeraldOffline,
                        trackColor = BorderSubtle
                    )
                }
            } else {
                // Actions: Allow instant switching to any engine + optional offline weights download
                Column(modifier = Modifier.fillMaxWidth()) {
                    if (downloadProgress?.errorMessage != null) {
                        Surface(
                            color = DtuRedDark.copy(alpha = 0.3f),
                            shape = RoundedCornerShape(6.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(bottom = 8.dp)
                        ) {
                            Text(
                                text = "Notice: ${downloadProgress.errorMessage}. You can still use the active engine profile.",
                                color = DtuRedLight,
                                fontSize = 11.sp,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                            )
                        }
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        if (isDownloaded) {
                            // Offline weights status indicator
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Default.CheckCircle,
                                    contentDescription = null,
                                    tint = EmeraldOffline,
                                    modifier = Modifier.size(14.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    text = "Weights on disk",
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.Medium,
                                    color = EmeraldOffline
                                )
                            }

                            // Activate / Active Button
                            if (!isActive) {
                                Button(
                                    onClick = onActivate,
                                    shape = RoundedCornerShape(8.dp),
                                    colors = ButtonDefaults.buttonColors(containerColor = EmeraldOffline),
                                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)
                                ) {
                                    Icon(imageVector = Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(14.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Activate Model", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                                }
                            } else {
                                Surface(
                                    color = EmeraldDark.copy(alpha = 0.25f),
                                    shape = RoundedCornerShape(8.dp),
                                    border = androidx.compose.foundation.BorderStroke(1.dp, EmeraldOffline.copy(alpha = 0.4f))
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Check,
                                            contentDescription = null,
                                            tint = EmeraldGlow,
                                            modifier = Modifier.size(13.dp)
                                        )
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text(
                                            text = "Active Engine",
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.SemiBold,
                                            color = EmeraldGlow
                                        )
                                    }
                                }
                            }
                        } else {
                            // NOT DOWNLOADED: Strictly only allow downloading! Cannot switch without downloading!
                            Text(
                                text = "Not downloaded",
                                fontSize = 11.sp,
                                color = TextMuted
                            )
                            Button(
                                onClick = onDownload,
                                shape = RoundedCornerShape(8.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1D4ED8)),
                                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                            ) {
                                Icon(imageVector = Icons.Default.Download, contentDescription = null, modifier = Modifier.size(14.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Download GGUF (${spec.sizeMb} MB)", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                            }
                        }
                    }
                }
            }
        }
    }
}
