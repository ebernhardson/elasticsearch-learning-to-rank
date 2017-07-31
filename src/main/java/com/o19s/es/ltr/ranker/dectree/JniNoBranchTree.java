package com.o19s.es.ltr.ranker.dectree;

import com.o19s.es.ltr.ranker.DenseFeatureVector;
import com.o19s.es.ltr.ranker.DenseLtrRanker;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Leaf;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Node;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Split;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.CountType;
import org.apache.lucene.util.Accountable;
import org.elasticsearch.SpecialPermission;

import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Queue;
import java.util.LinkedList;

public class JniNoBranchTree extends DenseLtrRanker implements Accountable {
    private final long handle;
    private final int nFeat;
    private boolean closed = false;

    static {
        final SecurityManager sm = System.getSecurityManager();
        if (sm != null) {
            sm.checkPermission(new SpecialPermission());
        }
        AccessController.doPrivileged(new PrivilegedAction<Object>() {
           @Override
            public Object run() {
               System.loadLibrary("bridge");
               return null;
           }
        });
    }

    JniNoBranchTree(long handle, int nFeat) {
        this.handle = handle;
        this.nFeat = nFeat;
    }

    protected void finalize() {
        if (!closed) {
            closed = true;
            destroyEnsemble(handle);
        }
    }

    private static native long createEnsemble(int numTrees, int numNodes);
    private static native void destroyEnsemble(long handle);
    private static native int addTree(long handle, byte depth, int[] fids, float[] threshOrScores);
    private static native void evalMultiple(long handle, DenseFeatureVector[] denseVects, float[] scores);
    private static native float eval(long handle, float[] features);

    @Override
    public String name() {
        return "nobrch_additive_decision_tree";
    }

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
        return eval(handle, denseVect.scores);
    }

    @Override
    public void score(DenseFeatureVector[] denseVects, float scores[]) {
        // While it's not fun to deal with the DenseFeatureVector object in jni, otherwise
        // we would have to copy all the feature values into a new array to pass them
        // as one.
        evalMultiple(handle, denseVects, scores);
    }

    static class BuildState {
        private final Node[] trees;
        private final int nFeat;

        BuildState(Node[] trees, int nFeat) {
            this.trees = trees;
            this.nFeat = nFeat;
        }

        public JniNoBranchTree build() {
            int total = 0;
            for (int i = 0; i < trees.length; i++) {
                trees[i] = balance(trees[i], depthOf(trees[i]));
                total += trees[i].count(CountType.All);
            }

            long handle = createEnsemble(trees.length, total);
            for (Node tree : trees) {
                flatten(handle, tree);
            }

            return new JniNoBranchTree(handle, nFeat);
        }

        private int depthOf(Node n) {
            if (n.isLeaf()) {
                return 0;
            } else {
                Split s = (Split) n;
                return 1 + Math.max(depthOf(s.left()), depthOf(s.right()));
            }
        }

        private Node balance(Node n, int depth) {
            if (depth == 0) {
                assert n.isLeaf();
                return n;
            } else if (n.isLeaf()) {
                Leaf l = (Leaf) n;
                return balance(new Split(l, l, 0, 0F), depth);
            } else {
                Split s = (Split) n;
                Node l = balance(s.left(), depth - 1);
                Node r = balance(s.right(), depth - 1);
                return new Split(l, r, s.feature(), s.threshold());
            }
        }

        private void flatten(long handle, Node tree) {
            int total = tree.count(CountType.All);
            int[] fids = new int[total];
            float[] threshOrScore = new float[total];

            Queue<Node> queue = new LinkedList<>();
            queue.add(tree);
            int i = 0;
            while (!queue.isEmpty()) {
                Node n = queue.remove();
                if (n.isLeaf()) {
                    threshOrScore[i] = ((Leaf) n).output();
                } else {
                    Split s = (Split) n;
                    threshOrScore[i] = s.threshold();
                    fids[i] = s.feature();
                    queue.add(s.left());
                    queue.add(s.right());
                }
                i++;
            }

            Integer res = addTree(handle, (byte) depthOf(tree), fids, threshOrScore);
            if (res != 1) {
                throw new RuntimeException("Error adding tree: " + res.toString());
            }
        }
    }
}
