package com.aihub.local;

import android.app.Activity;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.webkit.WebSettings;
import android.widget.Toast;

import androidx.webkit.WebViewAssetLoader;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class MainActivity extends Activity {
    private WebView webView;
    private CoreOnnxRuntime coreRuntime;
    private WebViewAssetLoader assetLoader;
    private final Pattern tokenPattern = Pattern.compile("\\w+|[^\\w\\s]", Pattern.UNICODE_CHARACTER_CLASS);
    private final Map<String, Integer> tokenIds = new HashMap<>();
    private final List<String> vocabulary = new ArrayList<>();
    private final Map<Integer, Map<Integer, Integer>> transitions = new HashMap<>();
    private final Map<Integer, Integer> unigram = new HashMap<>();
    private String startupWarning;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            loadPrototypeModel();
        } catch (Exception error) {
            startupWarning = "Demo model unavailable: " + safeMessage(error);
            android.util.Log.e("AiHub", startupWarning, error);
            installMinimalFallback();
        }
        try {
            coreRuntime = CoreOnnxRuntime.tryLoad(getAssets());
        } catch (Throwable error) {
            startupWarning = "Core runtime unavailable: " + safeMessage(error);
            android.util.Log.e("AiHub", startupWarning, error);
            coreRuntime = null;
        }
        assetLoader = new WebViewAssetLoader.Builder()
                .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                .build();
        createWebView();
    }

    private void createWebView() {
        webView = new WebView(this);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setSafeBrowsingEnabled(true);
        webView.addJavascriptInterface(new LocalModelBridge(), "AiHub");
        webView.setWebViewClient(new WebViewClient() {
            @Override public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                return assetLoader.shouldInterceptRequest(request.getUrl());
            }
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                return !"appassets.androidplatform.net".equals(request.getUrl().getHost());
            }
        });
        setContentView(webView);
        webView.loadUrl("https://appassets.androidplatform.net/assets/index.html");
    }

    private void loadPrototypeModel() throws Exception {
        try (InputStream input = getAssets().open("core-prototype.json");
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int count;
            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
            JSONObject root = new JSONObject(output.toString(StandardCharsets.UTF_8.name()));
            JSONArray words = root.getJSONObject("tokenizer").getJSONArray("vocabulary");
            for (int i = 0; i < words.length(); i++) {
                String word = words.getString(i);
                vocabulary.add(word);
                tokenIds.put(word, i);
            }
            JSONObject model = root.getJSONObject("model");
            JSONObject counts = model.getJSONObject("counts");
            java.util.Iterator<String> previous = counts.keys();
            while (previous.hasNext()) {
                String key = previous.next();
                JSONObject next = counts.getJSONObject(key);
                Map<Integer, Integer> row = new HashMap<>();
                java.util.Iterator<String> ids = next.keys();
                while (ids.hasNext()) {
                    String id = ids.next();
                    int tokenId = Integer.parseInt(id);
                    int frequency = next.getInt(id);
                    if (tokenId < 0 || tokenId >= words.length() || frequency < 0)
                        throw new IllegalArgumentException("Invalid prototype transition");
                    row.put(tokenId, frequency);
                }
                transitions.put(Integer.parseInt(key), row);
            }
            JSONObject singles = model.getJSONObject("unigram");
            java.util.Iterator<String> ids = singles.keys();
            while (ids.hasNext()) {
                String id = ids.next();
                int tokenId = Integer.parseInt(id);
                int frequency = singles.getInt(id);
                if (tokenId < 0 || tokenId >= words.length() || frequency < 0)
                    throw new IllegalArgumentException("Invalid prototype unigram");
                unigram.put(tokenId, frequency);
            }
            if (vocabulary.isEmpty() || unigram.isEmpty())
                throw new IllegalArgumentException("Prototype model has no vocabulary or token counts");
        }
    }

    private void installMinimalFallback() {
        vocabulary.clear(); tokenIds.clear(); transitions.clear(); unigram.clear();
        String[] words = {"Ai'Hub", " is", " starting", ".", " Please", " try", " again", "."};
        for (int i = 0; i < words.length; i++) { vocabulary.add(words[i]); tokenIds.put(words[i], i); }
        unigram.put(0, 1); unigram.put(1, 1); unigram.put(2, 1); unigram.put(3, 1);
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty() ? error.getClass().getSimpleName() : message;
    }

    private List<Integer> encode(String text) {
        List<Integer> result = new ArrayList<>();
        Matcher matcher = tokenPattern.matcher(text);
        while (matcher.find()) result.add(tokenIds.getOrDefault(matcher.group(), 0));
        return result;
    }

    private String decode(List<Integer> ids) {
        StringBuilder result = new StringBuilder();
        for (int id : ids) {
            String token = id >= 0 && id < vocabulary.size() ? vocabulary.get(id) : "<unk>";
            if (".,!?;:)]}".contains(token) || token.equals("'")) result.append(token);
            else if (token.equals("(") || token.equals("[") || token.equals("{")) result.append(token);
            else {
                if (result.length() > 0 && result.charAt(result.length() - 1) != ' '
                        && result.charAt(result.length() - 1) != '('
                        && result.charAt(result.length() - 1) != '['
                        && result.charAt(result.length() - 1) != '{') result.append(' ');
                result.append(token);
            }
        }
        return result.toString();
    }

    private int sample(Map<Integer, Integer> distribution, Random random) {
        if (distribution == null || distribution.isEmpty()) distribution = unigram;
        long total = 0;
        for (int count : distribution.values()) total += count;
        if (total <= 0) return 0;
        long choice = (long)(random.nextDouble() * total);
        for (Map.Entry<Integer, Integer> entry : distribution.entrySet()) {
            choice -= entry.getValue();
            if (choice < 0) return entry.getKey();
        }
        return 0;
    }

    private final class LocalModelBridge {
        @JavascriptInterface public String modelName() {
            return coreRuntime != null ? coreRuntime.modelName() : "Ai'Hub Core bigram demo";
        }
        @JavascriptInterface public String startupStatus() {
            return startupWarning == null ? "" : startupWarning;
        }
        @JavascriptInterface public String generate(String prompt, int maxTokens) {
            try {
                if (prompt != null && prompt.length() > 12000)
                    throw new IllegalArgumentException("Message is too long. Keep it under 12,000 characters.");
                if (coreRuntime != null) {
                    JSONObject response = new JSONObject();
                    response.put("text", coreRuntime.generate(prompt, maxTokens));
                    response.put("prototype", false);
                    return response.toString();
                }
                if (maxTokens < 1 || maxTokens > 128)
                    throw new IllegalArgumentException("Choose 1–128 new tokens.");
                List<Integer> output = encode(prompt == null ? "" : prompt);
                Random random = new Random();
                for (int i = 0; i < maxTokens; i++) {
                    Integer previous = output.isEmpty() ? null : output.get(output.size() - 1);
                    output.add(sample(previous == null ? null : transitions.get(previous), random));
                }
                JSONObject response = new JSONObject();
                response.put("text", decode(output));
                response.put("prototype", true);
                return response.toString();
            } catch (Throwable error) {
                JSONObject response = new JSONObject();
                try { response.put("error", safeMessage(error)); } catch (Exception ignored) {}
                return response.toString();
            }
        }
    }

    @Override public void onBackPressed() {
        if (webView != null && webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }

    @Override protected void onDestroy() {
        if (webView != null) { webView.stopLoading(); webView.removeJavascriptInterface("AiHub"); webView.destroy(); webView = null; }
        if (coreRuntime != null) { coreRuntime.close(); coreRuntime = null; }
        super.onDestroy();
    }
}
