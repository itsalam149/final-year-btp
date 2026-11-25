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
        dragHandle = { BottomSheetDefaults.DragHandle(color = TextSecondary.copy(alpha = 0.5f)) }
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
                            text = "Edge Model Gallery",
                            fontSize = 19.sp,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary
                        )
                    }
                    Text(
                        text = "Download once directly on-device • Runs 100% offline",
                        fontSize = 12.sp,
                        color = EmeraldGlow
                    )
                }
                IconButton(onClick = onDismiss) {
                    Icon(imageVector = Icons.Default.Close, contentDescription = "Close", tint = TextSecondary)
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            // Info Card
            Surface(
                color = CardBackground.copy(alpha = 0.6f),
                shape = RoundedCornerShape(10.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle.copy(alpha = 0.5f)),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Info,
                        contentDescription = "Info",
                        tint = BlueAccent,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Models are stored in private internal storage. High-speed direct streaming with zero cloud telemetry.",
                        fontSize = 11.sp,
                        color = TextSecondary,
                        lineHeight = 15.sp
                    )
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
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedButton(
                        onClick = onImportLocalFile,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = BlueAccent),
                        border = androidx.compose.foundation.BorderStroke(1.dp, BlueAccent.copy(alpha = 0.5f))
                    ) {
                        Icon(imageVector = Icons.Default.FileOpen, contentDescription = "Import GGUF", modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Import .GGUF from Phone Storage", fontWeight = FontWeight.SemiBold)
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
            // Title & Active Badge
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = spec.displayName,
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "${spec.precision} • ${spec.sizeMb} MB • ~${spec.ramMb} MB RAM",
                        fontSize = 11.sp,
                        color = TextSecondary
                    )
                }

                if (isActive) {
                    Surface(
                        color = EmeraldDark.copy(alpha = 0.35f),
                        shape = RoundedCornerShape(8.dp),
                        border = androidx.compose.foundation.BorderStroke(1.dp, EmeraldOffline.copy(alpha = 0.6f))
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
                        val totalMb = spec.sizeMb.toDouble()
                        val pctInt = ((downloadProgress?.progressPct ?: 0f) * 100).toInt()

                        Text(
                            text = "Streaming: ${String.format("%.1f", mbDownloaded)} MB / ${spec.sizeMb} MB ($pctInt%)",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            color = BlueAccent,
                            fontFamily = FontFamily.Monospace
                        )

                        TextButton(
                            onClick = onCancelDownload,
                            contentPadding = PaddingValues(horizontal = 6.dp, vertical = 2.dp)
                        ) {
                            Icon(imageVector = Icons.Default.Close, contentDescription = "Cancel", tint = DtuRedLight, modifier = Modifier.size(14.dp))
                            Spacer(modifier = Modifier.width(2.dp))
                            Text("Cancel", color = DtuRedLight, fontSize = 11.sp)
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    LinearProgressIndicator(
                        progress = { animatedProgress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(RoundedCornerShape(3.dp)),
                        color = EmeraldOffline,
                        trackColor = BorderSubtle
                    )
                }
            } else {
                // Not downloading: Show Action Buttons or Error
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
                                text = "Download issue: ${downloadProgress.errorMessage}. Tap to retry.",
                                color = DtuRedLight,
                                fontSize = 11.sp,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                            )
                        }
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.End
                    ) {
                        if (isDownloaded) {
                            if (!isActive) {
                                Button(
                                    onClick = onActivate,
                                    shape = RoundedCornerShape(8.dp),
                                    colors = ButtonDefaults.buttonColors(containerColor = BlueAccent),
                                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)
                                ) {
                                    Icon(imageVector = Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(14.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Load Model", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                                }
                            } else {
                                OutlinedButton(
                                    onClick = {},
                                    enabled = false,
                                    shape = RoundedCornerShape(8.dp),
                                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)
                                ) {
                                    Icon(imageVector = Icons.Default.Check, contentDescription = null, tint = EmeraldOffline, modifier = Modifier.size(14.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Active Engine", fontSize = 12.sp, color = EmeraldOffline)
                                }
                            }
                        } else {
                            Button(
                                onClick = onDownload,
                                shape = RoundedCornerShape(8.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = DtuRedPrimary),
                                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)
                            ) {
                                Icon(imageVector = Icons.Default.Download, contentDescription = null, modifier = Modifier.size(14.dp))
                                Spacer(modifier = Modifier.width(6.dp))
                                Text("Download (${spec.sizeMb} MB)", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                            }
                        }
                    }
                }
            }
        }
    }
}
