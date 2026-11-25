#ifndef LLAMA_BRIDGE_H
#define LLAMA_BRIDGE_H

#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <android/log.h>

#define TAG "OfflineMobileAgent-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct InferenceMetrics {
    double promptEvalMs = 0.0;
    double evalMs = 0.0;
    int evalTokens = 0;
    double tokensPerSecond = 0.0;
    size_t memoryAllocatedMb = 0;
};

class MobileLlamaEngine {
public:
    MobileLlamaEngine();
    ~MobileLlamaEngine();

    bool initialize(const std::string& modelPath, int nThreads, int nCtx);
    std::string generateWithGrammar(const std::string& prompt, 
                                   const std::string& grammarRules, 
                                   int maxTokens, 
                                   float temperature);
    InferenceMetrics getMetrics() const;
    void release();
    bool isLoaded() const;

private:
    bool m_isLoaded = false;
    std::string m_modelPath;
    int m_nThreads = 4;
    int m_nCtx = 2048;
    InferenceMetrics m_lastMetrics;
};

#endif // LLAMA_BRIDGE_H
