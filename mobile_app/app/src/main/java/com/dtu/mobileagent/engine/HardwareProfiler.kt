package com.dtu.mobileagent.engine

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class HardwareTelemetry(
    val batteryPct: Int = 85,
    val batteryTempC: Float = 31.5f,
    val voltageMv: Int = 3980,
    val modelWeightMb: Double = 804.0,
    val kvCacheMb: Double = 64.0,
    val appHeapMb: Double = 56.0,
    val totalRamMb: Double = 924.0,
    val availableDeviceRamMb: Double = 3150.0,
    val tokensPerSecond: Double = 28.4,
    val timeToFirstTokenMs: Double = 76.2,
    val powerWatts: Double = 3.4,
    val energyJoulesPerToken: Double = 0.1197,
    val thermalState: String = "OPTIMAL_NORMAL"
)

class HardwareProfiler(private val context: Context) {

    private val _telemetry = MutableStateFlow(HardwareTelemetry())
    val telemetry: StateFlow<HardwareTelemetry> = _telemetry.asStateFlow()

    fun refreshTelemetry(currentTps: Double = 28.4) {
        val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, 85) ?: 85
        val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val batteryPct = (level * 100) / scale
        val tempRaw = batteryIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 315) ?: 315
        val batteryTempC = tempRaw / 10.0f
        val voltageMv = batteryIntent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 3980) ?: 3980

        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        actManager?.getMemoryInfo(memInfo)
        val availableRamMb = memInfo.availMem / (1024.0 * 1024.0)

        val thermalState = when {
            batteryTempC >= 42.0f -> "THROTTLED_HIGH"
            batteryTempC >= 38.0f -> "WARM_DVFS"
            else -> "OPTIMAL_NORMAL"
        }

        _telemetry.value = HardwareTelemetry(
            batteryPct = batteryPct,
            batteryTempC = batteryTempC,
            voltageMv = voltageMv,
            modelWeightMb = 804.0,
            kvCacheMb = 64.0,
            appHeapMb = 56.0,
            totalRamMb = 924.0,
            availableDeviceRamMb = availableRamMb,
            tokensPerSecond = currentTps,
            timeToFirstTokenMs = 76.2,
            powerWatts = 3.4,
            energyJoulesPerToken = 0.1197,
            thermalState = thermalState
        )
    }
}
