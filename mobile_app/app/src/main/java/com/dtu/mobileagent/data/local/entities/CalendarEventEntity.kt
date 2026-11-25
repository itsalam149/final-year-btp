package com.dtu.mobileagent.data.local.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "calendar_events")
data class CalendarEventEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val title: String,
    val dateIso: String,
    val startTime: String,
    val endTime: String,
    val location: String = "Campus / Online",
    val description: String = ""
)
