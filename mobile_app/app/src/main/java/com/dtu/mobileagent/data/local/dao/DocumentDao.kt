package com.dtu.mobileagent.data.local.dao

import androidx.room.*
import com.dtu.mobileagent.data.local.entities.DocumentEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface DocumentDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertDocument(doc: DocumentEntity): Long

    @Query("SELECT * FROM documents WHERE filename LIKE '%' || :query || '%' OR content LIKE '%' || :query || '%' LIMIT :limit")
    suspend fun searchDocuments(query: String, limit: Int = 5): List<DocumentEntity>

    @Query("SELECT * FROM documents WHERE filename LIKE '%' || :filename || '%' LIMIT 1")
    suspend fun findByFilename(filename: String): DocumentEntity?

    @Query("SELECT * FROM documents ORDER BY indexedAt DESC LIMIT :limit")
    suspend fun getRecentDocuments(limit: Int = 10): List<DocumentEntity>

    @Query("SELECT * FROM documents ORDER BY indexedAt DESC")
    fun getAllDocumentsFlow(): Flow<List<DocumentEntity>>

    @Query("SELECT COUNT(*) FROM documents")
    suspend fun getDocumentCount(): Int

    @Query("DELETE FROM documents")
    suspend fun clearDocuments()
}
