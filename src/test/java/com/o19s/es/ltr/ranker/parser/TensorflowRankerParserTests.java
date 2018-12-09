package com.o19s.es.ltr.ranker.parser;

import com.o19s.es.ltr.LtrTestUtils;
import com.o19s.es.ltr.feature.store.StoredFeature;
import com.o19s.es.ltr.feature.store.StoredFeatureSet;
import com.o19s.es.ltr.ranker.linear.LinearRankerTests;
import com.o19s.es.ltr.ranker.tensorflow.TensorflowFeatureVector;
import com.o19s.es.ltr.ranker.tensorflow.TensorflowRanker;
import org.apache.lucene.util.LuceneTestCase;
import org.elasticsearch.common.io.PathUtils;
import org.elasticsearch.common.io.Streams;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

public class TensorflowRankerParserTests extends LuceneTestCase {
    private final TensorflowRankerParser parser = new TensorflowRankerParser(() ->
            PathUtils.get(System.getProperty("java.io.tmpdir")));

    public void testHappyPath() throws IOException {
        String model = readModel("/models/tf.dense.zip");
        List<StoredFeature> features = new ArrayList<>();
        List<String> names = Arrays.asList("title",
                "opening_text",
                "text");
        for (String n : names) {
            features.add(LtrTestUtils.randomFeature(n));
        }

        StoredFeatureSet set = new StoredFeatureSet("set", features);
        TensorflowRanker ranker = parser.parse(set, model);

        TensorflowFeatureVector v = ranker.newFeatureVector(null);
        for (int i = random().nextInt(5000) + 1000; i > 0; i--) {
            LinearRankerTests.fillRandomWeights(v.vector());
            assertFalse(Float.isNaN(ranker.score(v)));
        }
    }

    private String readModel(String model) throws IOException {
        try (InputStream is = this.getClass().getResourceAsStream(model)) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            Streams.copy(is,  bos);
            byte[] encoded = Base64.getEncoder().encode(bos.toByteArray());
            return new String(encoded, StandardCharsets.UTF_8);
        }
    }
}
