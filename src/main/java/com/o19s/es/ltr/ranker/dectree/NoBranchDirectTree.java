package com.o19s.es.ltr.ranker.dectree;

import com.o19s.es.ltr.ranker.DenseFeatureVector;
import com.o19s.es.ltr.ranker.DenseLtrRanker;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Leaf;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Node;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.Split;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree.CountType;
import org.apache.lucene.util.Accountable;

import java.nio.ByteBuffer;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class NoBranchDirectTree extends DenseLtrRanker implements Accountable {
    private static final int SPLIT_SIZE = Short.BYTES + Float.BYTES;

    private final List<OneNoBranchDirectTree> trees;
    private final int nFeat;

    NoBranchDirectTree(List<OneNoBranchDirectTree> trees, int nFeat) {
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
        for (OneNoBranchDirectTree tree: trees) {
            score += tree.score(denseVect);
        }
        return score;
    }

    static class OneNoBranchDirectTree {
        final ByteBuffer buf;
        final int depth;

        OneNoBranchDirectTree(ByteBuffer buf, int depth) {
            this.buf = buf;
            this.depth = depth;
        }

        static OneNoBranchDirectTree forDepth(ByteBuffer buf, int depth) {
            switch (depth) {
                case 4:
                    return new OneNoBranchDirectTreeDepth4(buf);
                default:
                    return new OneNoBranchDirectTree(buf, depth);
            }
        }

        public float score(DenseFeatureVector denseVect) {
            int j = 0;
            for (int i = depth; i > 0; i--) {
                buf.position(j * SPLIT_SIZE);
                float threshold = buf.getFloat();
                short fid = buf.getShort();
                j = (j << 1) + 1 + (threshold <= denseVect.scores[fid] ? 1 : 0);
            }
            return buf.getFloat(j * SPLIT_SIZE);
        }
    }

    static class OneNoBranchDirectTreeDepth4 extends OneNoBranchDirectTree {
        OneNoBranchDirectTreeDepth4(ByteBuffer buf) {
            super(buf, 4);
        }

        public float score(DenseFeatureVector denseVect) {
            buf.position(0);
            float threshold = buf.getFloat();
            short fid = buf.getShort();
            int j = 1 + (denseVect.scores[fid] > threshold ? 1 : 0);
            buf.position(j * SPLIT_SIZE);
            threshold = buf.getFloat();
            fid = buf.getShort();
            j = (j << 1) + 1 + (threshold > denseVect.scores[fid] ? 0 : 1);
            buf.position(j * SPLIT_SIZE);
            threshold = buf.getFloat();
            fid = buf.getShort();
            j = (j << 1) + 1 + (threshold > denseVect.scores[fid] ? 0 : 1);
            buf.position(j * SPLIT_SIZE);
            threshold = buf.getFloat();
            fid = buf.getShort();
            j = (j << 1) + 1 + (threshold > denseVect.scores[fid] ? 0 : 1);
            return buf.getFloat(j * SPLIT_SIZE);
        }
    }

    static class BuildState {
        private final Node[] trees;
        private final int nFeat;

        BuildState(Node[] trees, int nFeat) {
            this.trees = trees;
            this.nFeat = nFeat;
        }

        NoBranchDirectTree build() {
            List<OneNoBranchDirectTree> data = new ArrayList<>(trees.length);
            for (Node tree : trees) {
                int depth = depthOf(tree);
                ByteBuffer buf = flatten(balance(tree, depth));
                data.add(OneNoBranchDirectTree.forDepth(buf, depth));
            }

            return new NoBranchDirectTree(data, nFeat);
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

        private ByteBuffer flatten(Node tree) {
            int total = tree.count(CountType.All);
            int size = total * (SPLIT_SIZE);
            ByteBuffer buf = ByteBuffer.allocateDirect(size);

            Queue<Node> queue = new LinkedList<>();
            queue.add(tree);
            while (!queue.isEmpty()) {
                Node n = queue.remove();
                if (n.isLeaf()) {
                    buf.putFloat(((Leaf) n).output());
                    buf.putShort((short) 0);
                } else {
                    Split s = (Split) n;
                    buf.putFloat(s.threshold());
                    buf.putShort((short) s.feature());
                    queue.add(s.left());
                    queue.add(s.right());
                }
            }
            return buf;
        }
    }
}
