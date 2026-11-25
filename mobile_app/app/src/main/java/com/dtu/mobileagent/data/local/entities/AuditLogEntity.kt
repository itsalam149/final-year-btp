package com.dtu.mobileagent.data.local.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "audit_log")
data class AuditLogEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val toolName: String,
    val parametersJson: String,
    val status: String, // EXECUTED, CONFIRMATION_REQUIRED, REJECTED, FAILED
    val executionDetails: String,
    val timestamp: Long = System.currentTimeMillis()
)
