# Keep JNI bridge methods and native library symbols
-keepclassmembers class com.dtu.mobileagent.engine.LlamaBridge {
    native <methods>;
}

# Keep Room entity and DAO definitions
-keep class com.dtu.mobileagent.data.local.entities.** { *; }
-keep interface com.dtu.mobileagent.data.local.dao.** { *; }

# Keep GSON data structures for function calling
-keepclassmembers class com.dtu.mobileagent.engine.ToolCall { *; }
-keepclassmembers class com.dtu.mobileagent.engine.ToolResult { *; }
