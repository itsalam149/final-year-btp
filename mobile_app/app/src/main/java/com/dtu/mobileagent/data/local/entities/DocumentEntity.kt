package com.dtu.mobileagent.data.local.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "documents")
data class DocumentEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val filename: String,
    val filepath: String,
    val content: String,
    val summary: String = "",
    val indexedAt: Long = System.currentTimeMillis()
)
