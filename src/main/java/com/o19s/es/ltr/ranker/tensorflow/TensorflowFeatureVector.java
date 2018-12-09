package com.o19s.es.ltr.ranker.tensorflow;

import com.o19s.es.ltr.ranker.LtrRanker;
import org.tensorflow.Tensor;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.function.Function;

public class TensorflowFeatureVector implements LtrRanker.FeatureVector {
    private final FloatBuffer buffer;
    private final FloatBuffer outputBuffer = FloatBuffer.allocate(1);
    private final long[] shape;

    TensorflowFeatureVector(int size, long[] shape) {
        buffer = FloatBuffer.allocate(size);
        assert buffer.hasArray();
        this.shape = shape;
    }

    @Override
    public void setFeatureScore(int featureId, float score) {
        buffer.put(featureId, score);
    }

    @Override
    public float getFeatureScore(int featureId) {
        return buffer.get(featureId);
    }

    // Only visible for testing.
    public float[] vector() {
        return buffer.array();
    }

    public float run(Function<Tensor<Float>, Tensor<Float>> fn) {
        buffer.clear();
        outputBuffer.clear();
        try(Tensor<Float> tensor = Tensor.create(shape, buffer);
            Tensor<Float> result = fn.apply(tensor)) {
            result.writeTo(outputBuffer);
            return outputBuffer.get(0);
        }
    }

    void reset() {
        Arrays.fill(buffer.array(), 0F);
    }
}
