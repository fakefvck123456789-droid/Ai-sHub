package com.aihub.local;

import android.app.Activity;
import android.os.Bundle;
import android.webkit.WebResourceRequest;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

public final class MainActivity extends Activity {
    private WebView webView;
    private LocalAssetServer assetServer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        assetServer = new LocalAssetServer(getAssets());
        try {
            assetServer.start();
        } catch (Exception error) {
            Toast.makeText(this, "Could not start the Ai'Hub app interface: " + error.getMessage(),
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

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (webView != null) webView.destroy();
        if (assetServer != null) assetServer.close();
        super.onDestroy();
    }
}
