package com.dtu.mobileagent.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dtu.mobileagent.engine.PendingAction

@Composable
fun EdgeDotsLogo(
    modifier: Modifier = Modifier,
    dotSize: Float = 6f,
    spacing: Float = 3f
) {
    Row(
        modifier = modifier,
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(spacing.dp)
    ) {
        Box(
            modifier = Modifier
                .size(dotSize.dp)
                .clip(CircleShape)
                .background(GoogleBlue)
        )
        Box(
            modifier = Modifier
                .size(dotSize.dp)
                .clip(CircleShape)
                .background(GoogleRed)
        )
        Box(
            modifier = Modifier
                .size(dotSize.dp)
                .clip(CircleShape)
                .background(GoogleYellow)
        )
        Box(
            modifier = Modifier
                .size(dotSize.dp)
                .clip(CircleShape)
                .background(GoogleGreen)
        )
    }
}

@Composable
fun EdgeIconBadge(
    icon: ImageVector,
    badgeColor: Color,
    modifier: Modifier = Modifier,
    size: Int = 40,
    iconSize: Int = 20,
    isSquircle: Boolean = false
) {
    Box(
        modifier = modifier
            .size(size.dp)
            .clip(if (isSquircle) RoundedCornerShape(12.dp) else CircleShape)
            .background(badgeColor),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier.size(iconSize.dp)
        )
    }
}

@Composable
fun EdgeFeatureCard(
    title: String,
    subtitle: String,
    icon: ImageVector,
    badgeColor: Color,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    isSquircle: Boolean = false
) {
    Surface(
        onClick = onClick,
        color = GoogleCard,
        shape = RoundedCornerShape(20.dp),
        modifier = modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            EdgeIconBadge(
                icon = icon,
                badgeColor = badgeColor,
                size = 46,
                iconSize = 22,
                isSquircle = isSquircle
            )

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(3.dp))
                Text(
                    text = subtitle,
                    fontSize = 12.sp,
                    color = TextSecondary,
                    lineHeight = 16.sp,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun EdgeGridCard(
    badgeText: String,
    title: String,
    description: String,
    icon: ImageVector,
    badgeColor: Color,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    isSquircle: Boolean = false
) {
    Surface(
        onClick = onClick,
        color = GoogleCard,
        shape = RoundedCornerShape(22.dp),
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 170.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            EdgeIconBadge(
                icon = icon,
                badgeColor = badgeColor,
                size = 42,
                iconSize = 22,
                isSquircle = isSquircle
            )

            Spacer(modifier = Modifier.height(14.dp))

            Column {
                Text(
                    text = badgeText,
                    fontSize = 11.sp,
                    color = TextMuted,
                    fontWeight = FontWeight.Medium
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = title,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = description,
                    fontSize = 12.sp,
                    color = TextSecondary,
                    lineHeight = 16.sp,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
fun OfflineSecurityBadge(
    modifier: Modifier = Modifier
) {
    Surface(
        color = GoogleGreenContainer.copy(alpha = 0.4f),
        shape = RoundedCornerShape(20.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, GoogleGreen.copy(alpha = 0.5f)),
        modifier = modifier
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(7.dp)
                    .clip(CircleShape)
                    .background(GoogleGreen)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Icon(
                imageVector = Icons.Default.Lock,
                contentDescription = null,
                tint = GoogleGreen,
                modifier = Modifier.size(13.dp)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = "100% Offline • Zero Cloud Egress",
                color = GoogleGreen,
                fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
fun ModelMetricsBar(
    tokensPerSec: Double,
    memoryMb: Double,
    modifier: Modifier = Modifier
) {
    Surface(
        color = GoogleSurface,
        shape = RoundedCornerShape(0.dp),
        modifier = modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .background(GoogleGreen)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "${String.format("%.1f", tokensPerSec)} tok/s • ARM NEON",
                    color = TextSecondary,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium
                )
            }

            Text(
                text = "${String.format("%.0f", memoryMb)} MB RAM • Q4_K_M",
                color = TextSecondary,
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
fun HardwareTelemetryCard(
    telemetry: com.dtu.mobileagent.engine.HardwareTelemetry,
    modifier: Modifier = Modifier
) {
    Surface(
        color = GoogleCard,
        shape = RoundedCornerShape(20.dp),
        modifier = modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    EdgeIconBadge(
                        icon = Icons.Default.Memory,
                        badgeColor = GoogleYellow,
                        size = 32,
                        iconSize = 16,
                        isSquircle = true
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = "Hardware Telemetry",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary
                    )
                }
                Surface(
                    color = if (telemetry.thermalState == "OPTIMAL_NORMAL") GoogleGreenContainer else GoogleYellowContainer,
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = telemetry.thermalState,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = if (telemetry.thermalState == "OPTIMAL_NORMAL") GoogleGreen else GoogleYellow,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("RAM Footprint", fontSize = 11.sp, color = TextMuted)
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "${String.format("%.0f", telemetry.totalRamMb)} MB / ${String.format("%.0f", telemetry.availableDeviceRamMb)} MB",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }

                Column(horizontalAlignment = Alignment.End) {
                    Text("Device Battery", fontSize = 11.sp, color = TextMuted)
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "${telemetry.batteryPct}% • ${String.format("%.1f", telemetry.batteryTempC)}°C",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Throughput & Latency", fontSize = 11.sp, color = TextMuted)
                Text("${String.format("%.1f", telemetry.tokensPerSecond)} tok/s • ${String.format("%.1f", telemetry.timeToFirstTokenMs)}ms TTFT", fontSize = 11.sp, color = TextSecondary)
            }
        }
    }
}

@Composable
fun ConfirmationDialog(
    pending: PendingAction,
    onConfirm: (Boolean) -> Unit
) {
    AlertDialog(
        onDismissRequest = { onConfirm(false) },
        icon = {
            EdgeIconBadge(
                icon = Icons.Default.Warning,
                badgeColor = GoogleRed,
                size = 46,
                iconSize = 24
            )
        },
        title = {
            Text(
                text = "Two-Tier Security Gate",
                fontSize = 17.sp,
                fontWeight = FontWeight.Bold,
                color = TextPrimary
            )
        },
        text = {
            Column {
                Text(
                    text = "EdgeAgent requested an elevated action requiring human consent before execution:",
                    fontSize = 13.sp,
                    color = TextSecondary
                )
                Spacer(modifier = Modifier.height(10.dp))
                Surface(
                    color = GoogleCardElevated,
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text = "Tool: ${pending.toolName}",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = GoogleRed
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = pending.description,
                            fontSize = 12.sp,
                            color = TextPrimary
                        )
                    }
                }
            }
        },
        confirmButton = {
            Button(
                onClick = { onConfirm(true) },
                colors = ButtonDefaults.buttonColors(containerColor = GoogleGreen),
                shape = RoundedCornerShape(20.dp)
            ) {
                Text("Approve Execution", fontWeight = FontWeight.Bold, color = Color.White)
            }
        },
        dismissButton = {
            TextButton(
                onClick = { onConfirm(false) }
            ) {
                Text("Block & Deny", color = TextSecondary)
            }
        },
        containerColor = GoogleCard,
        shape = RoundedCornerShape(24.dp)
    )
}
