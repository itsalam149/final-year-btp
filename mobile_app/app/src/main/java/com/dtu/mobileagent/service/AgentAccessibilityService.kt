package com.dtu.mobileagent.service

import android.accessibilityservice.AccessibilityService
import android.os.Bundle
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

class AgentAccessibilityService : AccessibilityService() {

    companion object {
        private const val TAG = "AgentAccessibility"
        var instance: AgentAccessibilityService? = null
            private set

        val isServiceRunning: Boolean
            get() = instance != null
    }

    private var activePackageName: String = ""

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "Level 3 Accessibility Automation Service connected successfully.")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event?.packageName?.let {
            activePackageName = it.toString()
        }
    }

    override fun onInterrupt() {
        Log.w(TAG, "Accessibility service interrupted.")
    }

    override fun onDestroy() {
        super.onDestroy()
        if (instance == this) {
            instance = null
        }
        Log.i(TAG, "Accessibility service destroyed.")
    }

    /**
     * Recursively traverses root window and finds an element containing target text,
     * then executes ACTION_CLICK on the node or its clickable parent.
     */
    fun clickElementByText(targetText: String): Boolean {
        val root = rootInActiveWindow ?: return false
        val cleanTarget = targetText.trim().lowercase()

        val matchingNodes = root.findAccessibilityNodeInfosByText(cleanTarget)
        if (matchingNodes.isNullOrEmpty()) {
            return false
        }

        for (node in matchingNodes) {
            var currentNode: AccessibilityNodeInfo? = node
            while (currentNode != null) {
                if (currentNode.isClickable) {
                    val clicked = currentNode.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                    if (clicked) {
                        Log.i(TAG, "Clicked node matching '$targetText' in $activePackageName")
                        return true
                    }
                }
                currentNode = currentNode.parent
            }
        }

        // If no clickable ancestor found, attempt direct click
        val directNode = matchingNodes.firstOrNull()
        return directNode?.performAction(AccessibilityNodeInfo.ACTION_CLICK) ?: false
    }

    /**
     * Finds an editable field matching hint or text, focuses it, and enters text.
     */
    fun enterTextInField(targetLabel: String, textToEnter: String): Boolean {
        val root = rootInActiveWindow ?: return false
        val cleanTarget = targetLabel.trim().lowercase()

        val nodes = root.findAccessibilityNodeInfosByText(cleanTarget)
        for (node in nodes) {
            if (node.isEditable) {
                val args = Bundle().apply {
                    putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, textToEnter)
                }
                val success = node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
                if (success) {
                    Log.i(TAG, "Entered text into field '$targetLabel' in $activePackageName")
                    return true
                }
            }
        }

        // Fallback: search for any focused editable field
        val focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
        if (focused != null && focused.isEditable) {
            val args = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, textToEnter)
            }
            return focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
        }

        return false
    }

    /**
     * Returns a list of visible texts and button labels currently rendered on screen.
     */
    fun dumpVisibleScreenText(): List<String> {
        val root = rootInActiveWindow ?: return emptyList()
        val elements = mutableListOf<String>()
        collectNodeText(root, elements)
        return elements
    }

    private fun collectNodeText(node: AccessibilityNodeInfo?, list: MutableList<String>) {
        if (node == null || !node.isVisibleToUser) return

        val text = node.text?.toString()?.trim()
        val desc = node.contentDescription?.toString()?.trim()

        if (!text.isNullOrBlank()) {
            list.add(text)
        } else if (!desc.isNullOrBlank()) {
            list.add(desc)
        }

        for (i in 0 until node.childCount) {
            collectNodeText(node.getChild(i), list)
        }
    }

    /**
     * Executes global navigation actions: back, home, recents.
     */
    fun performNavigation(action: String): Boolean {
        return when (action.lowercase()) {
            "back" -> performGlobalAction(GLOBAL_ACTION_BACK)
            "home" -> performGlobalAction(GLOBAL_ACTION_HOME)
            "recents" -> performGlobalAction(GLOBAL_ACTION_RECENTS)
            "notifications" -> performGlobalAction(GLOBAL_ACTION_NOTIFICATIONS)
            else -> false
        }
    }
}
