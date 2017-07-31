package com.o19s.es.ltr.ranker.dectree;

import com.o19s.es.ltr.ranker.DenseFeatureVector;
import com.o19s.es.ltr.ranker.DenseLtrRanker;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Leaf;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Node;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Split;
import javassist.CtClass;
import javassist.CtNewMethod;
import javassist.ClassPool;
import javassist.CtNewConstructor;
import org.apache.lucene.util.Accountable;

import java.util.Locale;

public abstract class JavassistCodeGenTree extends DenseLtrRanker implements Accountable {
    @Override
    public String name() { return "codegen_additive_decision_tree"; }

    @Override
    public long ramBytesUsed() {
        return 99;
    }

    @Override
    protected abstract int size();

    static class BuildState {
        static int classNum = 0;

        private final Node[] trees;
        private final int nFeat;

        BuildState(Node[] trees, int nFeat) {
            this.trees = trees;
            this.nFeat = nFeat;
        }

        JavassistCodeGenTree build() {
            try {
                ClassPool pool = ClassPool.getDefault();
                CtClass cc = pool.makeClass(String.format(Locale.ENGLISH,"GeneratedDecTree%d", classNum++));
                cc.setSuperclass(pool.get(JavassistCodeGenTree.class.getName()));
                cc.addConstructor(CtNewConstructor.defaultConstructor(cc));
                cc.addMethod(CtNewMethod.make(String.format(Locale.ENGLISH,"protected int size() { return %d; }", nFeat), cc));
                cc.addMethod(CtNewMethod.make(genScoreMethod(cc), cc));
                return (JavassistCodeGenTree) cc.toClass().newInstance();
            } catch (Exception e) {
                throw new RuntimeException("Failed codegen", e);
            }
        }

        private String genScoreMethod(CtClass cc) throws Exception {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format(Locale.ENGLISH,"public float score(%s denseVect) {", DenseFeatureVector.class.getName()));
            sb.append("float score = 0F;");
            int i = 0;
            for (Node tree : trees) {
                i++;
                cc.addMethod(CtNewMethod.make(genEvalMethod(i, tree), cc));
                sb.append(String.format(Locale.ENGLISH,"score += evalTree%d(denseVect);", i));
            }
            sb.append("return score;}");
            return sb.toString();
        }

        private String genEvalMethod(int i, Node n) {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format(Locale.ENGLISH,"private float evalTree%d(%s denseVect) {", i, DenseFeatureVector.class.getName()));
            genScoreNode(sb, n);
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
