package com.dtu.mobileagent.data.local.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "reminders")
data class ReminderEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val title: String,
    val dueTimeIso: String,
    val priority: String = "normal", // low, normal, high
    val isCompleted: Boolean = false,
    val createdAt: Long = System.currentTimeMillis()
)
