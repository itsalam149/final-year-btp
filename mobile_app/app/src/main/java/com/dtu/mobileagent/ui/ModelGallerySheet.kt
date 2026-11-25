package com.dtu.mobileagent.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
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
        dragHandle = { BottomSheetDefaults.DragHandle(color = TextSecondary) }
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp)
                .padding(bottom = 32.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "On-Device Model Gallery",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                    Text(
                        text = "Edge Gallery: Download once, run 100% offline",
                        fontSize = 12.sp,
                        color = EmeraldGlow
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
                    val downloadState = downloadStates[spec.id]
                    val isDownloading = downloadState?.isDownloading == true

                    ModelCardItem(
                        spec = spec,
                        isDownloaded = isDownloaded,
                        isActive = isActive,
                        isDownloading = isDownloading,
                        onActivate = {
                            coroutineScope.launch {
                                galleryManager.setActiveModel(spec)
                                onModelActivated(spec)
                            }
                        },
                        onDownload = {
                            galleryManager.startDownload(spec)
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
    isDownloading: Boolean,
    onActivate: () -> Unit,
    onDownload: () -> Unit
) {
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
                    Text(
                        text = "${spec.precision} • ${spec.sizeMb} MB • ~${spec.ramMb} MB RAM",
                        fontSize = 11.sp,
                        color = TextSecondary
                    )
                }

                if (isActive) {
                    Surface(
                        color = EmeraldDark.copy(alpha = 0.4f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            text = "ACTIVE",
                            color = EmeraldGlow,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = spec.description,
                fontSize = 12.sp,
                color = TextSecondary,
                lineHeight = 16.sp
            )

            Spacer(modifier = Modifier.height(10.dp))

            if (isDownloading) {
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp)),
                    color = EmeraldOffline,
                    trackColor = BorderSubtle
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Downloading from HuggingFace...",
                    fontSize = 11.sp,
                    color = BlueAccent
                )
            } else {
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
                                Text("Loaded", fontSize = 12.sp, color = EmeraldOffline)
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
