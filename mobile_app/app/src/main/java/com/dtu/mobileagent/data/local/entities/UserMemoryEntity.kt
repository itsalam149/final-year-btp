package com.dtu.mobileagent.data.local.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "user_memories")
data class UserMemoryEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val category: String = "general", // "personal", "project", "preference", "general"
    val keyTag: String, // e.g. "guide", "roll_number", "name", "project"
    val fact: String, // e.g. "Project supervisor is Ms. Gull Kaur"
    val createdAt: Long = System.currentTimeMillis()
)
