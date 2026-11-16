package com.aihub.local;

import android.content.res.AssetManager;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

final class CoreOnnxRuntime implements AutoCloseable {
    private final OrtEnvironment environment;
    private final OrtSession session;
    private final ByteBpe tokenizer;
    private final String inputName;
    private final int contextLength;
    private final int vocabSize;
    private final Random random = new Random();

    static CoreOnnxRuntime tryLoad(AssetManager assets) {
        try {
            return new CoreOnnxRuntime(assets);
        } catch (Exception missingOrInvalidModel) {
            android.util.Log.i("AiHub", "Core ONNX model not bundled; using offline demo", missingOrInvalidModel);
            return null;
        }
    }

    private CoreOnnxRuntime(AssetManager assets) throws Exception {
        JSONObject metadata = new JSONObject(readAsset(assets, "core-0.1-model.json"));
        JSONObject config = metadata.getJSONObject("config");
        contextLength = config.getInt("context_length");
        vocabSize = config.getInt("vocab_size");
        tokenizer = new ByteBpe(readAsset(assets, "core-0.1-tokenizer.json"));
        environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
        session = environment.createSession(readAssetBytes(assets, "core-0.1.onnx"), options);
        inputName = session.getInputNames().iterator().next();
    }

    String modelName() { return "Ai'Hub Core 0.1 · ONNX"; }

    String generate(String prompt, int maxTokens) throws Exception {
        if (maxTokens < 1 || maxTokens > 128) throw new IllegalArgumentException("Choose 1–128 new tokens.");
        List<Integer> tokens = tokenizer.encode(prompt == null ? "" : prompt);
        if (tokens.isEmpty()) tokens.add(32);
        for (int step = 0; step < maxTokens; step++) {
            int start = Math.max(0, tokens.size() - contextLength);
            long[] ids = new long[tokens.size() - start];
            for (int i = start; i < tokens.size(); i++) ids[i - start] = tokens.get(i);
            long[] shape = new long[]{1, ids.length};
            try (OnnxTensor input = OnnxTensor.createTensor(environment,
                    LongBuffer.wrap(ids), shape);
                 OrtSession.Result result = session.run(Collections.singletonMap(inputName, input))) {
                float[][][] logits = (float[][][]) result.get(0).getValue();
                float[] last = logits[0][logits[0].length - 1];
                int next = sampleTopK(last, 40);
                if (next < 0 || next >= vocabSize) throw new IllegalStateException("Model emitted an invalid token ID.");
                tokens.add(next);
            }
        }
        return tokenizer.decode(tokens);
    }

    private int sampleTopK(float[] logits, int topK) {
        int count = Math.min(topK, logits.length);
        int[] indices = new int[count];
        float[] values = new float[count];
        Arrays.fill(values, Float.NEGATIVE_INFINITY);
        for (int i = 0; i < logits.length; i++) {
            float value = logits[i];
            for (int slot = 0; slot < count; slot++) {
                if (value > values[slot]) {
                    for (int move = count - 1; move > slot; move--) {
                        values[move] = values[move - 1];
                        indices[move] = indices[move - 1];
                    }
                    values[slot] = value;
                    indices[slot] = i;
                    break;
                }
            }
        }
        double max = values[0], total = 0.0;
        double[] weights = new double[count];
        for (int i = 0; i < count; i++) {
            weights[i] = Math.exp(values[i] - max);
            total += weights[i];
        }
        double pick = random.nextDouble() * total;
        for (int i = 0; i < count; i++) {
            pick -= weights[i];
            if (pick <= 0) return indices[i];
        }
        return indices[0];
    }

    private static byte[] readAssetBytes(AssetManager assets, String name) throws Exception {
        try (InputStream input = assets.open(name); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16384];
            int count;
            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
            return output.toByteArray();
        }
    }

    private static String readAsset(AssetManager assets, String name) throws Exception {
        return new String(readAssetBytes(assets, name), StandardCharsets.UTF_8);
    }

    @Override public void close() {
        try { session.close(); } catch (Exception ignored) {}
        try { environment.close(); } catch (Exception ignored) {}
    }

    private static final class ByteBpe {
        private final List<int[]> merges = new ArrayList<>();
        private final List<byte[]> expansions = new ArrayList<>();

        ByteBpe(String json) throws Exception {
            JSONObject root = new JSONObject(json);
            if (!"byte_bpe".equals(root.getString("type"))) throw new IllegalArgumentException("Unsupported tokenizer");
            for (int i = 0; i < 256; i++) expansions.add(new byte[]{(byte)i});
            JSONArray pairs = root.getJSONArray("merges");
            for (int i = 0; i < pairs.length(); i++) {
                JSONArray pair = pairs.getJSONArray(i);
                int left = pair.getInt(0), right = pair.getInt(1);
                if (left < 0 || right < 0 || left >= expansions.size() || right >= expansions.size())
                    throw new IllegalArgumentException("Invalid BPE merge");
                merges.add(new int[]{left, right});
                byte[] a = expansions.get(left), b = expansions.get(right);
                byte[] combined = Arrays.copyOf(a, a.length + b.length);
                System.arraycopy(b, 0, combined, a.length, b.length);
                expansions.add(combined);
            }
        }

        List<Integer> encode(String text) {
            byte[] raw = text.getBytes(StandardCharsets.UTF_8);
            List<Integer> ids = new ArrayList<>(raw.length);
            for (byte value : raw) ids.add(value & 0xff);
            while (ids.size() > 1) {
                int bestRank = Integer.MAX_VALUE, bestIndex = -1;
                for (int i = 0; i + 1 < ids.size(); i++) {
                    int left = ids.get(i), right = ids.get(i + 1);
                    for (int rank = 0; rank < merges.size(); rank++) {
                        int[] pair = merges.get(rank);
                        if (pair[0] == left && pair[1] == right) {
                            if (rank < bestRank) { bestRank = rank; bestIndex = i; }
                            break;
                        }
                    }
                }
                if (bestIndex < 0) break;
                ids.set(bestIndex, 256 + bestRank);
                ids.remove(bestIndex + 1);
            }
            return ids;
        }

        String decode(List<Integer> ids) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            for (int id : ids) {
                if (id < 0 || id >= expansions.size()) throw new IllegalArgumentException("Invalid token ID");
                output.write(expansions.get(id), 0, expansions.get(id).length);
            }
            return new String(output.toByteArray(), StandardCharsets.UTF_8);
        }
    }
}
