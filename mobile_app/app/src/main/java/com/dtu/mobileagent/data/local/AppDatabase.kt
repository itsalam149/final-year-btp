package com.dtu.mobileagent.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.sqlite.db.SupportSQLiteDatabase
import com.dtu.mobileagent.data.local.dao.*
import com.dtu.mobileagent.data.local.entities.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@Database(
    entities = [
        NoteEntity::class,
        ReminderEntity::class,
        CalendarEventEntity::class,
        DocumentEntity::class,
        AuditLogEntity::class,
        ChatMessageEntity::class,
        UserMemoryEntity::class
    ],
    version = 2,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun noteDao(): NoteDao
    abstract fun reminderDao(): ReminderDao
    abstract fun calendarDao(): CalendarDao
    abstract fun documentDao(): DocumentDao
    abstract fun auditLogDao(): AuditLogDao
    abstract fun chatMessageDao(): ChatMessageDao
    abstract fun userMemoryDao(): UserMemoryDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context, scope: CoroutineScope): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "offline_mobile_agent.db"
                )
                .addCallback(DatabaseCallback(scope))
                .fallbackToDestructiveMigration()
                .build()
                INSTANCE = instance
                instance
            }
        }

        private class DatabaseCallback(
            private val scope: CoroutineScope
        ) : RoomDatabase.Callback() {
            override fun onCreate(db: SupportSQLiteDatabase) {
                super.onCreate(db)
                INSTANCE?.let { database ->
                    scope.launch(Dispatchers.IO) {
                        populateInitialData(database)
                    }
                }
            }

            suspend fun populateInitialData(database: AppDatabase) {
                // Populate sample academic calendar events
                database.calendarDao().insertEvent(
                    CalendarEventEntity(
                        title = "Major Project Review (DTU CSE)",
                        dateIso = "2026-11-26",
                        startTime = "10:00 AM",
                        endTime = "11:30 AM",
                        location = "Seminar Hall 2 / Online",
                        description = "Presentation with Ms. Gull Kaur. Team: Faqre, Ekansh, Guneet."
                    )
                )
                database.calendarDao().insertEvent(
                    CalendarEventEntity(
                        title = "Hardware Lab Evaluation",
                        dateIso = "2026-11-26",
                        startTime = "02:00 PM",
                        endTime = "03:30 PM",
                        location = "CSE Lab 3",
                        description = "NPU and ARM NEON on-device latency benchmark check."
                    )
                )

                // Populate sample note
                database.noteDao().insertNote(
                    NoteEntity(
                        title = "BTP Research Notes",
                        content = "Quantization findings: 4-bit naive causes syntax collapse (28% error). GBNF grammar constraints restore 100% adherence with 67% memory reduction."
                    )
                )

                // Populate sample reminder
                database.reminderDao().insertReminder(
                    ReminderEntity(
                        title = "Submit Final Year BTP Synopsis and Architecture Document",
                        dueTimeIso = "2026-11-26T17:00:00",
                        priority = "high"
                    )
                )

                // Populate sample indexed documents for local RAG
                database.documentDao().insertDocument(
                    DocumentEntity(
                        filename = "btp_project_overview.txt",
                        filepath = "/sdcard/Documents/btp_project_overview.txt",
                        content = "Project Title: Privacy-First Offline AI Mobile Agent Using LLM Quantization. Goal: Deploy an offline JARVIS-like assistant on Android smartphones with zero internet dependency, utilizing GBNF grammar-guided quantized models (Llama-3.2-1B, Qwen2.5-0.5B) and Room SQLite storage.",
                        summary = "Overview of DTU Major Project GP159 on privacy-first on-device mobile AI assistant."
                    )
                )

                database.documentDao().insertDocument(
                    DocumentEntity(
                        filename = "quantization_specs.txt",
                        filepath = "/sdcard/Documents/quantization_specs.txt",
                        content = "Memory specifications: Llama-3.2-1B FP16 requires 2,460 MB RAM. Q4_K_M requires 804 MB RAM (67.3% reduction). ARM NEON 4-thread execution achieves 28.4 tokens/sec on Snapdragon 8 Gen 2.",
                        summary = "Empirical memory footprint and inference throughput for on-device quantized models."
                    )
                )

                // Populate core user facts and memories
                database.userMemoryDao().insertMemory(
                    UserMemoryEntity(
                        category = "project",
                        keyTag = "guide",
                        fact = "Project Supervisor is Ms. Gull Kaur (DTU Computer Science and Engineering Department)."
                    )
                )
                database.userMemoryDao().insertMemory(
                    UserMemoryEntity(
                        category = "project",
                        keyTag = "group",
                        fact = "BTP Group ID is GP159. Team members: Faqre, Ekansh, Guneet, Bhawishya."
                    )
                )
                database.userMemoryDao().insertMemory(
                    UserMemoryEntity(
                        category = "personal",
                        keyTag = "institution",
                        fact = "Institution: Delhi Technological University (DTU), B.Tech CSE Final Year 2026."
                    )
                )
                database.userMemoryDao().insertMemory(
                    UserMemoryEntity(
                        category = "project",
                        keyTag = "topic",
                        fact = "Major Project Topic: Privacy-First Offline AI Mobile Agent Using LLM Quantization."
                    )
                )
            }
        }
    }
}
