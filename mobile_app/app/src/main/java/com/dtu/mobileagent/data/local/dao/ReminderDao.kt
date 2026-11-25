package com.dtu.mobileagent.data.local.dao

import androidx.room.*
import com.dtu.mobileagent.data.local.entities.ReminderEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface ReminderDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertReminder(reminder: ReminderEntity): Long

    @Query("SELECT * FROM reminders WHERE isCompleted = 0 ORDER BY dueTimeIso ASC")
    fun getActiveReminders(): Flow<List<ReminderEntity>>

    @Query("SELECT * FROM reminders ORDER BY dueTimeIso DESC")
    suspend fun getAllRemindersList(): List<ReminderEntity>

    @Query("UPDATE reminders SET isCompleted = 1 WHERE id = :id")
    suspend fun markCompleted(id: Long)
}
