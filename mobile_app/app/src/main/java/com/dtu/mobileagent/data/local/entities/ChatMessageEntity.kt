package com.dtu.mobileagent.data.local.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "chat_messages")
data class ChatMessageEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val sender: String, // "user", "agent", "system", "tool"
    val content: String,
    val thought: String = "",
    val toolName: String? = null,
    val toolParamsJson: String? = null,
    val requiresConfirmation: Boolean = false,
    val isConfirmed: Boolean = false,
    val timestamp: Long = System.currentTimeMillis()
)
