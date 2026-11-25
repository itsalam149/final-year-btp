package com.dtu.mobileagent.data.local.dao

import androidx.room.*
import com.dtu.mobileagent.data.local.entities.CalendarEventEntity

@Dao
interface CalendarDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEvent(event: CalendarEventEntity): Long

    @Query("SELECT * FROM calendar_events WHERE dateIso = :dateIso AND (title LIKE '%' || :query || '%' OR description LIKE '%' || :query || '%') ORDER BY startTime ASC")
    suspend fun searchEvents(dateIso: String, query: String): List<CalendarEventEntity>

    @Query("SELECT * FROM calendar_events WHERE dateIso = :dateIso ORDER BY startTime ASC")
    suspend fun getEventsForDate(dateIso: String): List<CalendarEventEntity>
}
