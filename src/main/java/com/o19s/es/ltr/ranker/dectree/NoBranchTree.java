package com.o19s.es.ltr.ranker.dectree;

import com.o19s.es.ltr.ranker.DenseFeatureVector;
import com.o19s.es.ltr.ranker.DenseLtrRanker;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Leaf;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Node;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Split;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.CountType;
import org.apache.lucene.util.Accountable;
import sun.misc.Unsafe;

import java.lang.reflect.Field;

import java.util.ArrayList;
import java.util.List;
import java.util.LinkedList;
import java.util.Queue;

public abstract class NoBranchTree extends DenseLtrRanker implements Accountable {
/*
    private static final int SPLIT_SIZE = Short.BYTES + Float.BYTES;

    private final List<OneNoBranchTree> trees;
    private final int nFeat;

    NoBranchTree(List<OneNoBranchTree> trees, int nFeat) {
        this.trees = trees;
        this.nFeat = nFeat;
    }

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
        float score = 0F;
        float[] feats = denseVect.scores;
        for (OneNoBranchTree tree: trees) {
            score += tree.score(feats);
        }
        return score;
    }

    @Override
    public void score(DenseFeatureVector[] denseVects, float[] scores) {
        assert denseVects.length <= scores.length;
        int i = 7;
        for (; i < denseVects.length; i += 8) {
            float[] s7 = denseVects[i-7].scores;
            scores[i-7] = 0;
            float[] s6 = denseVects[i-6].scores;
            scores[i-6] = 0;
            float[] s5 = denseVects[i-5].scores;
            scores[i-5] = 0;
            float[] s4 = denseVects[i-4].scores;
            scores[i-4] = 0;
            float[] s3 = denseVects[i-3].scores;
            scores[i-3] = 0;
            float[] s2 = denseVects[i-2].scores;
            scores[i-2] = 0;
            float[] s1 = denseVects[i-1].scores;
            scores[i-1] = 0;
            float[] s0 = denseVects[i].scores;
            scores[i] = 0;
            for (OneNoBranchTree tree: trees) {
                scores[i-7] += tree.score(s7);
                scores[i-6] += tree.score(s6);
                scores[i-5] += tree.score(s5);
                scores[i-4] += tree.score(s4);
                scores[i-3] += tree.score(s3);
                scores[i-2] += tree.score(s2);
                scores[i-1] += tree.score(s1);
                scores[i] += tree.score(s0);
            }
        }
        i -= 8;
        for (i++; i < denseVects.length; i++) {
           scores[i] = score(denseVects[i]);
        }
    }

    static class OneNoBranchTree {
        final Unsafe unsafe;
        final long buf;
        final int depth;
        boolean closed = false;

        OneNoBranchTree(Unsafe unsafe, long buf, int depth) {
            this.unsafe = unsafe;
            this.buf = buf;
            this.depth = depth;
        }

        static OneNoBranchTree forDepth(Unsafe unsafe, long buf, int depth) {
            switch (depth) {
                case 4:
                    return new OneNoBranchTreeDepth4(unsafe, buf);
                default:
                    return new OneNoBranchTree(unsafe, buf, depth);
            }
        }

        public float score(float[] scores) {
            int j = 0;
            long position = buf;
            for (int i = depth; i > 0; i--) {
                float threshold = unsafe.getFloat(position);
                short fid = unsafe.getShort(position + Float.BYTES);
                j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
                position = buf + (j * SPLIT_SIZE);
            }
            return unsafe.getFloat(position);
        }

        protected void finalize() {
            this.close();
        }

        void close() {
           if (!closed) {
               unsafe.freeMemory(buf);
               closed = true;
           }
        }
    }

    static class OneNoBranchTreeDepth4 extends OneNoBranchTree {
        OneNoBranchTreeDepth4(Unsafe unsafe, long buf) {
            super(unsafe, buf, 4);
        }

        public float score(float[] scores) {
            long position = buf;

            float threshold = unsafe.getFloat(position);
            short fid = unsafe.getShort(position + Float.BYTES);
            int j = 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            return unsafe.getFloat(position);
        }
    }

    static class OneNoBranchTreeDepth6 extends OneNoBranchTree {
        OneNoBranchTreeDepth6(Unsafe unsafe, long buf) {
            super(unsafe, buf, 6);
        }

        public float score(float[] scores) {
            long position = buf;

            float threshold = unsafe.getFloat(position);
            short fid = unsafe.getShort(position + Float.BYTES);
            int j = 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            return unsafe.getFloat(position);
        }
    }

    static class OneNoBranchTreeDepth8 extends OneNoBranchTree {
        OneNoBranchTreeDepth8(Unsafe unsafe, long buf) {
            super(unsafe, buf, 8);
        }

        public float score(float[] scores) {
            long position = buf;

            float threshold = unsafe.getFloat(position);
            short fid = unsafe.getShort(position + Float.BYTES);
            int j = 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            return unsafe.getFloat(position);
        }
    }

    static class OneNoBranchTreeDepth10 extends OneNoBranchTree {
        OneNoBranchTreeDepth10(Unsafe unsafe, long buf) {
            super(unsafe, buf, 10);
        }

        public float score(float[] scores) {
            long position = buf;

            float threshold = unsafe.getFloat(position);
            short fid = unsafe.getShort(position + Float.BYTES);
            int j = 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            threshold = unsafe.getFloat(position);
            fid = unsafe.getShort(position + Float.BYTES);
            j = (j << 1) + 1 + (scores[fid] > threshold ? 1 : 0);
            position = buf + (j * SPLIT_SIZE);

            return unsafe.getFloat(position);
        }
    }

    static class BuildState {
        private final Node[] trees;
        private final int nFeat;

        BuildState(Node[] trees, int nFeat) {
            this.trees = trees;
            this.nFeat = nFeat;
        }

        NoBranchTree build() {
            List<OneNoBranchTree> data = new ArrayList<>(trees.length);
            Unsafe unsafe = getUnsafe();
            for (Node tree : trees) {
                int depth = depthOf(tree);
                long buf = flatten(unsafe, balance(tree, depth));
                data.add(OneNoBranchTree.forDepth(unsafe, buf, depth));
            }

            return new NoBranchTree(data, nFeat);
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

        private long flatten(Unsafe unsafe, Node tree) {
            int total = tree.count(CountType.All);
            int size = total * SPLIT_SIZE;

            long startIndex = unsafe.allocateMemory(size);
            long position = startIndex;

            Queue<Node> queue = new LinkedList<>();
            queue.add(tree);
            while (!queue.isEmpty()) {
                Node n = queue.remove();
                if (n.isLeaf()) {
                    unsafe.putFloat(position, ((Leaf) n).output());
                    position += Float.BYTES;
                    unsafe.putShort(position, (short) 0);
                    position += Short.BYTES;
                } else {
                    Split s = (Split) n;
                    unsafe.putFloat(position, s.threshold());
                    position += Float.BYTES;
                    unsafe.putShort(position, (short) s.feature());
                    position += Short.BYTES;
                    queue.add(s.left());
                    queue.add(s.right());
                }
            }
            return startIndex;
        }

        private Unsafe getUnsafe() {
            try {
                Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
                theUnsafe.setAccessible(true);
                return (Unsafe) theUnsafe.get(null);
            } catch (Exception e) {
                throw new RuntimeException("...", e);
            }
        }
    }
*/
}
