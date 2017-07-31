package com.o19s.es.ltr.ranker.dectree;

import com.o19s.es.ltr.ranker.DenseFeatureVector;
import com.o19s.es.ltr.ranker.DenseLtrRanker;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Leaf;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Node;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Split;
import org.apache.lucene.util.Accountable;

import org.codehaus.janino.ScriptEvaluator;
import org.codehaus.janino.ClassBodyEvaluator;

import java.lang.reflect.Method;
import java.util.Locale;

public class CodeGenTree extends DenseLtrRanker implements Accountable {
    private final Method m;
    private final int nFeat;
    private static int nClass = 0;

    private CodeGenTree(Method m, int nFeat) {
        this.m = m;
        this.nFeat = nFeat;
    }

    @Override
    public String name() { return "codegen_additive_decision_tree"; }

    @Override
    protected int size() {
        return nFeat;
    }

    @Override
    public long ramBytesUsed() {
        return 99;
    }

    @Override
    public float score(DenseFeatureVector denseVect) {
        try {
            return (float) m.invoke(null, denseVect);
        } catch (Exception e) {
            throw new RuntimeException("...", e);
        }
    }

    static class BuildState {
        static int classNum = 0;

        private final Node[] trees;
        private final int nFeat;

        BuildState(Node[] trees, int nFeat) {
            this.trees = trees;
            this.nFeat = nFeat;
        }

        CodeGenTree build() {
            try {
                ClassBodyEvaluator cbe = new ClassBodyEvaluator();
                cbe.cook(genScript());
                Class<?> c = cbe.getClazz();
                Method m = c.getMethod("score", DenseFeatureVector.class);
                return new CodeGenTree(m, nFeat);
            } catch (Exception e) {
                throw new RuntimeException("Failed codegen", e);
            }
        }

        private String genScript() {
            StringBuilder sb = new StringBuilder();
            int i = 0;
            for (Node tree : trees) {
                sb.append(String.format(Locale.ENGLISH,"private static float evalTree%d(%s denseVect) {\n",
                        i, DenseFeatureVector.class.getName()));
                genScoreNode(sb, tree);
                sb.append("}\n");
                i++;
            }

            sb.append(String.format(Locale.ENGLISH,"public static float score(%s denseVect) {\n", DenseFeatureVector.class.getName()));
            sb.append("float score = 0F;\n");
            i = 0;
            for (Node tree : trees) {
                sb.append(String.format(Locale.ENGLISH,"score += evalTree%d(denseVect);\n", i));
                i++;
            }
            sb.append("return score;\n");
            sb.append("}");
            return sb.toString();
        }

        private String floatStr(float f) {
            return ((Float)f).toString();
        }

        private void genScoreNode(StringBuilder sb, Node n) {
            if (n.isLeaf()) {
                sb.append(String.format(Locale.ENGLISH,"return %sF;", floatStr(((Leaf) n).output())));
            } else {
                Split s = (Split) n;
                sb.append(String.format(Locale.ENGLISH,"if(%sF > denseVect.scores[%d]) {", floatStr(s.threshold()), s.feature()));
                genScoreNode(sb, s.left());
                sb.append("}else{");
                genScoreNode(sb, s.right());
                sb.append("}");
            }
        }
    }
}
