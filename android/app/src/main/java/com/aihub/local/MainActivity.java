package com.aihub.local;

import android.app.Activity;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceRequest;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

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
    private LocalAssetServer assetServer;
    private CoreOnnxRuntime coreRuntime;
    private final Pattern tokenPattern = Pattern.compile("\\w+|[^\\w\\s]", Pattern.UNICODE_CHARACTER_CLASS);
    private final Map<String, Integer> tokenIds = new HashMap<>();
    private final List<String> vocabulary = new ArrayList<>();
    private final Map<Integer, Map<Integer, Integer>> transitions = new HashMap<>();
    private final Map<Integer, Integer> unigram = new HashMap<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            loadPrototypeModel();
            coreRuntime = CoreOnnxRuntime.tryLoad(getAssets());
            assetServer = new LocalAssetServer(getAssets());
            assetServer.start();
        } catch (Exception error) {
            Toast.makeText(this, "Could not start Ai'Hub: " + error.getMessage(),
                    Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        webView = new WebView(this);
        webView.getSettings().setJavaScriptEnabled(true);
        webView.getSettings().setDomStorageEnabled(true);
        webView.getSettings().setAllowFileAccess(false);
        webView.getSettings().setAllowContentAccess(false);
        webView.getSettings().setMixedContentMode(android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        webView.addJavascriptInterface(new LocalModelBridge(), "AiHub");
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                String host = request.getUrl().getHost();
                return !("127.0.0.1".equals(host) || "localhost".equals(host));
            }
        });
        setContentView(webView);
        webView.loadUrl("http://127.0.0.1:8080/");
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
            JSONObject counts = root.getJSONObject("model").getJSONObject("counts");
            java.util.Iterator<String> previous = counts.keys();
            while (previous.hasNext()) {
                String key = previous.next();
                JSONObject next = counts.getJSONObject(key);
                Map<Integer, Integer> row = new HashMap<>();
                java.util.Iterator<String> ids = next.keys();
                while (ids.hasNext()) {
                    String id = ids.next();
                    row.put(Integer.parseInt(id), next.getInt(id));
                }
                transitions.put(Integer.parseInt(key), row);
            }
            JSONObject singles = root.getJSONObject("model").getJSONObject("unigram");
            java.util.Iterator<String> ids = singles.keys();
            while (ids.hasNext()) {
                String id = ids.next();
                unigram.put(Integer.parseInt(id), singles.getInt(id));
            }
        }
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
            if (".,!?;:)]}".contains(token) || token.equals("'")) {
                result.append(token);
            } else if (token.equals("(") || token.equals("[") || token.equals("{")) {
                result.append(token);
            } else {
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
        int total = 0;
        for (int count : distribution.values()) total += count;
        if (total <= 0) return 0;
        int choice = random.nextInt(total);
        for (Map.Entry<Integer, Integer> entry : distribution.entrySet()) {
            choice -= entry.getValue();
            if (choice < 0) return entry.getKey();
        }
        return 0;
    }

    private final class LocalModelBridge {
        @JavascriptInterface
        public String modelName() {
            return coreRuntime != null ? coreRuntime.modelName() : "Ai'Hub Core bigram demo";
        }

        @JavascriptInterface
        public String generate(String prompt, int maxTokens) {
            try {
                if (coreRuntime != null) {
                    JSONObject response = new JSONObject();
                    response.put("text", coreRuntime.generate(prompt, maxTokens));
                    response.put("prototype", false);
                    return response.toString();
                }
                if (maxTokens < 1 || maxTokens > 128) throw new IllegalArgumentException("Choose 1–128 new tokens.");
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
            } catch (Exception error) {
                JSONObject response = new JSONObject();
                try { response.put("error", error.getMessage()); } catch (Exception ignored) {}
                return response.toString();
            }
        }
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (webView != null) webView.destroy();
        if (assetServer != null) assetServer.close();
        if (coreRuntime != null) coreRuntime.close();
        super.onDestroy();
    }
}
