package com.dtu.mobileagent.data.local.dao

import androidx.room.*
import com.dtu.mobileagent.data.local.entities.ChatMessageEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface ChatMessageDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMessage(message: ChatMessageEntity): Long

    @Query("SELECT * FROM chat_messages ORDER BY timestamp ASC")
    fun getAllMessages(): Flow<List<ChatMessageEntity>>

    @Query("SELECT * FROM chat_messages ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentMessages(limit: Int = 20): List<ChatMessageEntity>

    @Query("UPDATE chat_messages SET isConfirmed = 1 WHERE id = :id")
    suspend fun markConfirmed(id: Long)

    @Query("DELETE FROM chat_messages")
    suspend fun clearHistory()
}
