package com.o19s.es.ltr.ranker.parser;

import com.google.protobuf.InvalidProtocolBufferException;
import com.o19s.es.ltr.feature.FeatureSet;
import com.o19s.es.ltr.ranker.tensorflow.TensorflowModelLoader;
import com.o19s.es.ltr.ranker.tensorflow.TensorflowRanker;
import org.elasticsearch.SpecialPermission;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.TensorFlow;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.framework.TensorShapeProto;

import java.io.IOException;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Arrays;
import java.util.Locale;
import java.util.function.Supplier;

public class TensorflowRankerParser implements LtrRankerParser {
    public static final String TYPE = "model/tensorflow";

    // From tf.saved_model.signature_constants in python
    public static final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default";
    public static final String PREDICT_METHOD_NAME = "tensorflow/serving/predict";
    public static final String TAG_SERVING = "serve";

    public static final String TENSORFLOW_VERSION;
    static {
        // The first time tensorflow jni is accessed it needs special access rights. Statically
        // initialize it when the plugin inits to ensure it's available. The version function
        // goes down to native, so why not grab it.
        SecurityManager sm = System.getSecurityManager();
        if (sm != null) {
            sm.checkPermission(new SpecialPermission());
        }
        TENSORFLOW_VERSION = AccessController.doPrivileged(new PrivilegedAction<String>() {
            @Override
            public String run() {
                return TensorFlow.version();
            }
        });
    }
    private final Supplier<Path> baseModelDir;

    public TensorflowRankerParser(Supplier<Path> baseModelDir) {
        this.baseModelDir = baseModelDir;
    }

    @Override
    public TensorflowRanker parse(FeatureSet set, String model) {
        // TODO: Model should be parsed with XContentParser to get various options
        // instead of assuming all the defaults here.
        return parse(set, model, TAG_SERVING, DEFAULT_SERVING_SIGNATURE_DEF_KEY);
    }

    private TensorflowRanker parse(FeatureSet set, String model, String tag, String signatureDefName) {
        try(TensorflowModelLoader loader = new TensorflowModelLoader(baseModelDir.get())) {
            SavedModelBundle bundle = loader.load(model, tag);
            SignatureDef signatureDef = getSignatureDefOrThrow(bundle, signatureDefName);
            return build(bundle, set, signatureDef);
        } catch (IOException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
    }

    private SignatureDef getSignatureDefOrThrow(SavedModelBundle bundle, String signatureDefName) {
        try {
            return MetaGraphDef.parseFrom(bundle.metaGraphDef())
                    .getSignatureDefOrThrow(signatureDefName);
        } catch (InvalidProtocolBufferException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
    }

    private TensorflowRanker build(SavedModelBundle bundle, FeatureSet set, SignatureDef signatureDef) {
        // TODO: Method Name should select an evaluation strategy
        String methodName = signatureDef.getMethodName();
        if (!PREDICT_METHOD_NAME.equals(methodName)) {
            throw new IllegalArgumentException();
        }
        if (signatureDef.getInputsCount() != 1) {
            throw new IllegalArgumentException("Model must have a single input");
        }
        TensorInfo input = signatureDef.getInputsMap().values().iterator().next();
        long[] inputShape = validateInput(input, set.size());

        if (signatureDef.getOutputsCount() != 1) {
            throw new IllegalArgumentException("Model must have a single output");
        }
        TensorInfo output = signatureDef.getOutputsMap().values().iterator().next();
        validateOutput(output);
        return new TensorflowRanker(bundle, input.getName(), inputShape, output.getName());
    }

    private boolean isCompatible(TensorShapeProto shape, long[] concrete) {
        for (int i = 0; i < concrete.length; i++) {
            long left = shape.getDim(i).getSize();
            long right = concrete[i];
            if (left != -1 && left != right) {
                return false;
            }
        }
        return true;
    }

    private long[] validateInput(TensorInfo info, int size) {
        if (info.getDtype() != DataType.DT_FLOAT) {
            throw new IllegalArgumentException();
        }

        TensorShapeProto shape = info.getTensorShape();
        long[] concrete;
        switch(shape.getDimCount()) {
            case 3:
                concrete = new long[]{1, 1, size};
                break;
            case 2:
                concrete = new long[]{1, size};
                break;
            default:
                throw new IllegalArgumentException("Only rank 2 or 3 inputs are currently supported.");
        }
        if (!isCompatible(shape, concrete)) {
            throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Input tensor shape is not compatible: [%s] [%s]",
                            shape.toString(), Arrays.toString(concrete)));
        }
        return concrete;
    }

    private void validateOutput(TensorInfo info) {
        if (info.getDtype() != DataType.DT_FLOAT) {
            throw new IllegalArgumentException();
        }
        TensorShapeProto shape = info.getTensorShape();
        long[] concrete;
        switch (shape.getDimCount()) {
            case 3:
                // should this be tied closer to input? do we care?
                concrete = new long[]{1, 1};
                break;
            case 2:
                concrete = new long[]{1};
                break;
            default:
                throw new IllegalArgumentException();
        }
        if (!isCompatible(shape, concrete)) {
            throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Output tensor shape is not compatible: [%s] [%s]",
                            shape.toString(), Arrays.toString(concrete)));
        }
    }
}
