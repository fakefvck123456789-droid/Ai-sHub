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
import java.util.Map;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Comparator;
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
        if (!"Ai'Hub Core 0.1".equals(metadata.getString("project")))
            throw new IllegalArgumentException("Unexpected Core model project");
        JSONObject config = metadata.getJSONObject("config");
        contextLength = config.getInt("context_length");
        vocabSize = config.getInt("vocab_size");
        if (contextLength < 1 || vocabSize < 256)
            throw new IllegalArgumentException("Invalid Core model dimensions");
        tokenizer = new ByteBpe(readAsset(assets, "core-0.1-tokenizer.json"));
        if (tokenizer.vocabSize() != vocabSize)
            throw new IllegalArgumentException("Tokenizer and model vocabulary sizes differ");
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
            try (OnnxTensor input = OnnxTensor.createTensor(environment, LongBuffer.wrap(ids), shape);
                 OrtSession.Result result = session.run(Collections.singletonMap(inputName, input))) {
                float[][][] logits = (float[][][]) result.get(0).getValue();
                float[] last = logits[0][logits[0].length - 1];
                int next = sampleTopK(last, 40);
                if (next < 0 || next >= vocabSize)
                    throw new IllegalStateException("Model emitted an invalid token ID.");
                tokens.add(next);
            }
        }
        return tokenizer.decode(tokens.subList(tokens.size() - maxTokens, tokens.size()));
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
        private final Map<Long, Integer> mergeRanks = new HashMap<>();

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
                mergeRanks.put(pairKey(left, right), i);
                byte[] a = expansions.get(left), b = expansions.get(right);
                byte[] combined = Arrays.copyOf(a, a.length + b.length);
                System.arraycopy(b, 0, combined, a.length, b.length);
                expansions.add(combined);
            }
        }

        int vocabSize() { return expansions.size(); }

        private static long pairKey(int left, int right) {
            return (((long) left) << 32) | (right & 0xffffffffL);
        }

        private static final class Candidate {
            final int rank, index, leftToken, rightToken;
            Candidate(int rank, int index, int leftToken, int rightToken) {
                this.rank = rank; this.index = index;
                this.leftToken = leftToken; this.rightToken = rightToken;
            }
        }

        List<Integer> encode(String text) {
            byte[] raw = text.getBytes(StandardCharsets.UTF_8);
            int count = raw.length;
            List<Integer> output = new ArrayList<>(count);
            if (count == 0) return output;
            int[] values = new int[count], previous = new int[count], next = new int[count];
            boolean[] alive = new boolean[count];
            for (int i = 0; i < count; i++) {
                values[i] = raw[i] & 0xff;
                previous[i] = i - 1;
                next[i] = i + 1 < count ? i + 1 : -1;
                alive[i] = true;
            }
            PriorityQueue<Candidate> queue = new PriorityQueue<>(
                    Comparator.comparingInt((Candidate item) -> item.rank)
                              .thenComparingInt(item -> item.index));
            for (int i = 0; i + 1 < count; i++) pushPair(queue, values, next, alive, i);
            while (!queue.isEmpty()) {
                Candidate candidate = queue.poll();
                int left = candidate.index;
                if (!alive[left]) continue;
                int right = next[left];
                if (right < 0 || !alive[right]) continue;
                if (values[left] != candidate.leftToken || values[right] != candidate.rightToken) continue;
                values[left] = 256 + candidate.rank;
                alive[right] = false;
                next[left] = next[right];
                if (next[left] >= 0) previous[next[left]] = left;
                pushPair(queue, values, next, alive, previous[left]);
                pushPair(queue, values, next, alive, left);
            }
            for (int index = 0; index >= 0; index = next[index])
                if (alive[index]) output.add(values[index]);
            return output;
        }

        private void pushPair(PriorityQueue<Candidate> queue, int[] values,
                              int[] next, boolean[] alive, int left) {
            if (left < 0 || !alive[left]) return;
            int right = next[left];
            if (right < 0 || !alive[right]) return;
            Integer rank = mergeRanks.get(pairKey(values[left], values[right]));
            if (rank != null)
                queue.add(new Candidate(rank, left, values[left], values[right]));
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
