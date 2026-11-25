package com.dtu.mobileagent.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val DarkColorScheme = darkColorScheme(
    primary = DtuRedLight,
    onPrimary = TextPrimary,
    primaryContainer = DtuRedDark,
    onPrimaryContainer = TextPrimary,
    secondary = EmeraldOffline,
    onSecondary = DarkBackground,
    secondaryContainer = EmeraldDark,
    onSecondaryContainer = TextPrimary,
    background = DarkBackground,
    onBackground = TextPrimary,
    surface = SurfaceDark,
    onSurface = TextPrimary,
    surfaceVariant = CardBackground,
    onSurfaceVariant = TextSecondary,
    outline = BorderSubtle
)

@Composable
fun OfflineMobileAgentTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        content = content
    )
}
