package com.o19s.es.ltr.ranker.dectree;

import com.o19s.es.ltr.LtrTestUtils;
import com.o19s.es.ltr.ranker.DenseFeatureVector;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTreeTests.SimpleCountRandomTreeGeneratorStatsCollector;
import org.apache.lucene.util.TestUtil;
import org.elasticsearch.test.ESTestCase;

public class NoBranchTreeTests extends ESTestCase {

    public void testScore() throws Exception {
        SimpleCountRandomTreeGeneratorStatsCollector counts = new SimpleCountRandomTreeGeneratorStatsCollector();
        // I've never seen this fail without the loop, but with 1k iterations in semi-reliably (hah) fails.
        // Need to figure out how to export that to get the reason...
        // The errors will generally be close, for example:
        //   expected:<97.083405> but was:<96.78854>
        //   expected:<-2.0881717> but was:<-0.48462558>
        // These might not be close enough to mean anything, not sure yet.
        for (int i = 0; i < 1000; i++) {
            NaiveAdditiveDecisionTree naive = NaiveAdditiveDecisionTreeTests.generateRandomDecTree(1, 1000,
                    1, 1000,
                    1, 8, counts);
            JniNoBranchTree nbTree = naive.toJniNoBranchTree();
            //NoBranchDirectTree nbTree = naive.toNoBranchTree();
            assertEquals(naive.size(), nbTree.size());
            //assertTrue(naive.ramBytesUsed() > binTree.ramBytesUsed());

            int nPass = TestUtil.nextInt(random(), 10, 8096);
            LtrTestUtils.assertRankersHaveSameScores(naive, nbTree, nPass);
        }
    }

    public void testEqualsSplit() throws Exception {
        NaiveAdditiveDecisionTree.Node trees[] = new NaiveAdditiveDecisionTree.Node[1];
        trees[0] = new NaiveAdditiveDecisionTree.Split(
            new NaiveAdditiveDecisionTree.Leaf(1F),
            new NaiveAdditiveDecisionTree.Leaf(0F),
            0, .1234F);

        NaiveAdditiveDecisionTree naive = new NaiveAdditiveDecisionTree(trees, 1);
        JniNoBranchTree cnbTree = naive.toJniNoBranchTree();
        NoBranchDirectTree nbdTree = naive.toNoBranchTree();

       DenseFeatureVector vec = naive.newFeatureVector(null);
       vec.scores[0] = .1234F;
       assertEquals(0F, naive.score(vec), Math.ulp(1F));
       assertEquals(0F, cnbTree.score(vec), Math.ulp(1F));
       assertEquals(0F, nbdTree.score(vec), Math.ulp(1F));
    }
}
