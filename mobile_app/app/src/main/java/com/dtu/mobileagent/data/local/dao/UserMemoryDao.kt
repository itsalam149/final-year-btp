package com.dtu.mobileagent.data.local.dao

import androidx.room.*
import com.dtu.mobileagent.data.local.entities.UserMemoryEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface UserMemoryDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMemory(memory: UserMemoryEntity): Long

    @Query("SELECT * FROM user_memories ORDER BY id DESC")
    fun getAllMemoriesFlow(): Flow<List<UserMemoryEntity>>

    @Query("SELECT * FROM user_memories ORDER BY id DESC")
    suspend fun getAllMemories(): List<UserMemoryEntity>

    @Query("SELECT * FROM user_memories WHERE fact LIKE '%' || :query || '%' OR keyTag LIKE '%' || :query || '%' LIMIT :limit")
    suspend fun searchMemories(query: String, limit: Int = 5): List<UserMemoryEntity>

    @Query("DELETE FROM user_memories WHERE keyTag = :keyTag OR fact LIKE '%' || :keyTag || '%'")
    suspend fun deleteMemoryByKey(keyTag: String): Int

    @Query("DELETE FROM user_memories WHERE id = :id")
    suspend fun deleteMemoryById(id: Long): Int

    @Query("DELETE FROM user_memories")
    suspend fun clearMemories()
}
