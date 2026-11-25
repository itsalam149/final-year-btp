package com.dtu.mobileagent

import com.dtu.mobileagent.engine.ModelGalleryManager
import com.dtu.mobileagent.engine.ModelSpec
import org.junit.Assert.*
import org.junit.Test

class ModelCatalogTest {

    @Test
    fun testCatalogHasAllRequiredModels() {
        val catalog = ModelGalleryManager.CATALOG
        assertEquals(5, catalog.size)

        val ids = catalog.map { it.id }
        assertTrue("Catalog must contain gemma-2-2b", ids.contains("gemma-2-2b"))
        assertTrue("Catalog must contain llama-3.2-3b", ids.contains("llama-3.2-3b"))
        assertTrue("Catalog must contain llama-3.2-1b", ids.contains("llama-3.2-1b"))
        assertTrue("Catalog must contain qwen2.5-0.5b", ids.contains("qwen2.5-0.5b"))
        assertTrue("Catalog must contain tinyllama-1.1b", ids.contains("tinyllama-1.1b"))
    }

    @Test
    fun testDefaultModelIsGemma2B() {
        val defaultModel = ModelGalleryManager.CATALOG.first()
        assertEquals("gemma-2-2b", defaultModel.id)
        assertEquals("Gemma-2-2B-Instruct", defaultModel.displayName)
        assertEquals(1629, defaultModel.sizeMb)
        assertEquals(2200, defaultModel.ramMb)
    }

    @Test
    fun testModelSpecsIntegrity() {
        for (spec in ModelGalleryManager.CATALOG) {
            assertTrue("Filename must end with .gguf: ${spec.filename}", spec.filename.endsWith(".gguf"))
            assertTrue("Download URL must be HTTPS: ${spec.downloadUrl}", spec.downloadUrl.startsWith("https://"))
            assertTrue("Size MB must be positive: ${spec.sizeMb}", spec.sizeMb > 0)
            assertTrue("RAM MB must be greater than size MB: ${spec.ramMb} vs ${spec.sizeMb}", spec.ramMb >= spec.sizeMb)
            assertTrue("Precision must describe quantization: ${spec.precision}", spec.precision.contains("Q4_K_M"))
            assertFalse("Description must not be empty", spec.description.isBlank())
        }
    }
}
