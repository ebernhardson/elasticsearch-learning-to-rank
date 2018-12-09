package com.o19s.es.ltr.ranker.tensorflow;

import com.o19s.es.ltr.ranker.LtrRanker;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;


public class TensorflowRanker implements LtrRanker {
    private final SavedModelBundle bundle;
    private final Session session;
    private final int size;
    private final String input;
    private final long[] inputShape;
    private final String output;

    public TensorflowRanker(SavedModelBundle bundle, String input, long[] inputShape, String output) {
        this.bundle = bundle;
        session = bundle.session();
        this.size = (int)inputShape[inputShape.length - 1];
        this.input = input;
        this.inputShape = inputShape;
        this.output = output;
    }

    @Override
    public TensorflowFeatureVector newFeatureVector(FeatureVector reuse) {
        if (reuse != null) {
            assert reuse instanceof TensorflowFeatureVector;
            TensorflowFeatureVector vector = (TensorflowFeatureVector)reuse;
            vector.reset();
            return vector;
        }
        return new TensorflowFeatureVector(size, inputShape);
    }

    @Override
    public String name() {
        return "tensorflow";
    }

    @Override
    public float score(FeatureVector vector) {
        assert vector instanceof TensorflowFeatureVector;
        return score((TensorflowFeatureVector) vector);
    }

    protected float score(TensorflowFeatureVector vector) {
        return vector.run((tensor) -> session.runner()
            .feed(input, tensor)
            .fetch(output)
            .run()
            .get(0)
            .expect(Float.class));
    }
}
