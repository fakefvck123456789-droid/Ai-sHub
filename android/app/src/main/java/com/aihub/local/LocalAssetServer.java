package com.aihub.local;

import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

final class LocalAssetServer implements AutoCloseable {
    private final AssetManager assets;
    private volatile boolean running;
    private ServerSocket serverSocket;
    private Thread worker;

    LocalAssetServer(AssetManager assets) { this.assets = assets; }

    void start() throws IOException {
        serverSocket = new ServerSocket(8080, 16, java.net.InetAddress.getByName("127.0.0.1"));
        running = true;
        worker = new Thread(this::serve, "aihub-assets-http");
        worker.setDaemon(true);
        worker.start();
    }

    private void serve() {
        while (running) {
            try {
                Socket socket = serverSocket.accept();
                socket.setSoTimeout(5000);
                Thread client = new Thread(() -> handle(socket), "aihub-assets-client");
                client.setDaemon(true);
                client.start();
            } catch (IOException error) {
                if (running) android.util.Log.w("AiHub", "Asset server accept failed", error);
            }
        }
    }

    private void handle(Socket socket) {
        try (Socket client = socket;
             BufferedReader reader = new BufferedReader(new InputStreamReader(
                     client.getInputStream(), StandardCharsets.US_ASCII));
             OutputStream output = client.getOutputStream()) {
            String requestLine = reader.readLine();
            if (requestLine == null) return;
            String[] parts = requestLine.split(" ");
            if (parts.length != 3 || !("GET".equals(parts[0]) || "HEAD".equals(parts[0]))) {
                respond(output, 405, "text/plain", "Method not allowed".getBytes(StandardCharsets.UTF_8),
                        "HEAD".equals(parts[0]));
                return;
            }
            String path = parts[1].split("\\?", 2)[0];
            if ("/".equals(path)) path = "/index.html";
            if (path.contains("..") || path.contains("\\\\")) {
                respond(output, 400, "text/plain", new byte[0], "HEAD".equals(parts[0]));
                return;
            }
            String assetPath = path.startsWith("/") ? path.substring(1) : path;
            byte[] body;
            try (InputStream input = assets.open(assetPath);
                 ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {
                byte[] chunk = new byte[8192];
                int count;
                while ((count = input.read(chunk)) != -1) buffer.write(chunk, 0, count);
                body = buffer.toByteArray();
            } catch (IOException missing) {
                respond(output, 404, "text/plain", "Not found".getBytes(StandardCharsets.UTF_8),
                        "HEAD".equals(parts[0]));
                return;
            }
            respond(output, 200, mime(assetPath), body, "HEAD".equals(parts[0]));
        } catch (IOException ignored) {
            // Client disconnected.
        }
    }

    private static String mime(String path) {
        if (path.endsWith(".html")) return "text/html; charset=utf-8";
        if (path.endsWith(".js")) return "text/javascript; charset=utf-8";
        if (path.endsWith(".css")) return "text/css; charset=utf-8";
        if (path.endsWith(".webmanifest") || path.endsWith(".json")) return "application/manifest+json";
        if (path.endsWith(".svg")) return "image/svg+xml";
        return "application/octet-stream";
    }

    private static void respond(OutputStream output, int status, String type, byte[] body,
                                boolean headOnly) throws IOException {
        String label = status == 200 ? "OK" : status == 404 ? "Not Found" :
                status == 405 ? "Method Not Allowed" : "Bad Request";
        String headers = "HTTP/1.1 " + status + " " + label + "\r\n" +
                "Content-Type: " + type + "\r\n" +
                "Content-Length: " + body.length + "\r\n" +
                "Cache-Control: no-store\r\n" +
                "X-Content-Type-Options: nosniff\r\n" +
                "Connection: close\r\n\r\n";
        output.write(headers.getBytes(StandardCharsets.US_ASCII));
        if (!headOnly) output.write(body);
        output.flush();
    }

    @Override
    public void close() {
        running = false;
        if (serverSocket != null) {
            try { serverSocket.close(); } catch (IOException ignored) {}
        }
    }
}
