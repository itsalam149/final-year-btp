package com.dtu.mobileagent

import android.app.Application
import com.dtu.mobileagent.data.local.AppDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob

class MobileAgentApplication : Application() {
    private val applicationScope = CoroutineScope(SupervisorJob())

    val database by lazy {
        AppDatabase.getDatabase(this, applicationScope)
    }

    override fun onCreate() {
        super.onCreate()
        // Application-level singletons initialized
    }
}
