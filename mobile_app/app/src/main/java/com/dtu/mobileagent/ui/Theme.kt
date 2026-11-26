package com.dtu.mobileagent.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val GoogleAIEdgeColorScheme = darkColorScheme(
    primary = GoogleBlue,
    onPrimary = TextPrimary,
    primaryContainer = GoogleBlueContainer,
    onPrimaryContainer = TextPrimary,
    secondary = GoogleGreen,
    onSecondary = GoogleBackground,
    secondaryContainer = GoogleGreenContainer,
    onSecondaryContainer = TextPrimary,
    tertiary = GoogleYellow,
    onTertiary = GoogleBackground,
    background = GoogleBackground,
    onBackground = TextPrimary,
    surface = GoogleSurface,
    onSurface = TextPrimary,
    surfaceVariant = GoogleCard,
    onSurfaceVariant = TextSecondary,
    outline = GoogleBorder,
    outlineVariant = GoogleDivider
)

@Composable
fun OfflineMobileAgentTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = GoogleAIEdgeColorScheme,
        content = content
    )
}
