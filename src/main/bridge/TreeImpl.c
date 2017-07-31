/**
 * Implementation based on:
 *  Runtime Optimizations for Prediction with Tree-Based Models
 *  by Nima Asadi, Jimmy Lin1, Arjen P. de Vries
 * https://arxiv.org/pdf/1212.2287.pdf
 */
#include "jni.h"
#include "TreeImpl.h"

struct Node {
    jint fid;
    jfloat threshOrScore;
};
typedef struct Node Node;

struct Root {
    size_t offset;
    jbyte depth;
};
typedef struct Root Root;

struct Ensemble {
    // Length of roots array
    jsize totalTrees;
    // Number of initialized trees
    jsize numTrees;
    // position of root inside nodes
    Root *roots;

    // Length of nodes array
    jsize numNodes;
    // node data
    Node *nodes;

    // Largest feature id in enseble
    jint maxFid;
};
typedef struct Ensemble Ensemble;

EnsembleHandle createEnsemble(jint numTrees, jint numNodes) {
    Ensemble *e = (Ensemble *) malloc(sizeof(Ensemble));
    if (e == NULL) {
        return (EnsembleHandle) 0;
    }
    e->totalTrees = numTrees;
    e->numTrees = 0;
    e->roots = (Root *) malloc(sizeof(Root) * numTrees);
    if (e->roots == NULL) {
        free(e);
        return (EnsembleHandle) 0;
    }
    e->roots[0].offset = 0;

    e->numNodes = numNodes;
    e->nodes = (Node *) malloc(sizeof(Node) * numNodes);
    if (e->nodes == NULL) {
        free(e->roots);
        free(e);
        return (EnsembleHandle) 0;
    }

    e->maxFid = 0;

    return (EnsembleHandle) e;
}

void destroyEnsemble(EnsembleHandle handle) {
    Ensemble *e = (Ensemble *) handle;
    free(e->roots);
    free(e->nodes);
    free(e);
}

jint addTree(EnsembleHandle handle, jbyte depth, jsize len, jint fids[], jfloat threshOrScore[]) {
    Ensemble *e = (Ensemble *) handle;

    if (e->numTrees >= e->totalTrees) {
        // No more room for trees
        return -1;
    }

    jsize offset = e->roots[e->numTrees].offset;
    if (offset + len > e->numNodes) {
        // Not enough room for the nodes
        return -2;
    }

    if (depth < 1) {
        // Invalid depth
        return -3;
    }

    if (depth > 10) {
        // Not currently supported. Could certainly have some
        // loop implementation fallback
        return -4;
    }

    // Make sure depth and len agree
    if (len != ((1 << (depth+1)) - 1)) {
        // Tree is not fully balanced
        return -5;
    }

    e->roots[e->numTrees].depth = depth;

    // Copy data into ensemble
    for (jsize i = 0; i < len; i++) {
        if (fids[i] > e->maxFid) {
            e->maxFid = fids[i];
        }
        e->nodes[offset + i].fid = fids[i];
        e->nodes[offset + i].threshOrScore = threshOrScore[i];
    }

    e->numTrees++;
    if (e->numTrees < e->totalTrees) {
        // assign root position of next tree
        e->roots[e->numTrees].offset = offset + len;
    }

    return 1;
}

// Gcc will unroll some of these loops for us, making for mostly straight
// through code.  as N*M increases though it will eventually decide the N loop is
// better off as a loop.
// 
// This is perhaps less than optimal for a single feature vector, as there are
// additional indirections. It's probably minimal, but unmeasured.
#define EVAL_DN_FN(N, M) \
void eval_d ## N ## _f ## M (jfloat *f[], const Node *root, jfloat *res) { \
    size_t j[M]; \
    for (int i = 0; i < (M); i++) { \
        j[i] = 1 + (root[0].threshOrScore <= f[i][root[0].fid]); \
    } \
    for (int k = 0; k < (N) - 1; k++) { \
        for (int i = 0; i < (M); i++) { \
            j[i] = (j[i] << 1) + 1 + (root[j[i]].threshOrScore <= f[i][root[j[i]].fid]); \
        } \
    } \
    for (int i = 0; i < (M); i++) { \
        res[i] += root[j[i]].threshOrScore; \
    } \
}

#define EVAL_FUNC(N) \
    EVAL_DN_FN(10, N) \
    EVAL_DN_FN(9, N) \
    EVAL_DN_FN(8, N) \
    EVAL_DN_FN(7, N) \
    EVAL_DN_FN(6, N) \
    EVAL_DN_FN(5, N) \
    EVAL_DN_FN(4, N) \
    EVAL_DN_FN(3, N) \
    EVAL_DN_FN(2, N) \
    EVAL_DN_FN(1, N) \
    typedef void (*EvalFunc_f ## N)(jfloat *[], const Node *, jfloat *); \
    /* Build a jump table for supported depths */\
    EvalFunc_f ## N evalers_f ## N [10] = { \
        &eval_d1_f##N, &eval_d2_f##N, &eval_d3_f##N, &eval_d4_f##N, &eval_d5_f##N, \
        &eval_d6_f##N, &eval_d7_f##N, &eval_d8_f##N, &eval_d9_f##N, &eval_d10_f##N \
    }; \
    void eval_f ## N (const Ensemble *e, jfloat *f[], jfloat *res) { \
        for (int i = 0; i < (N); i++) { \
            res[i] = 0; \
        } \
        Root *root = e->roots; \
        for (int i = 0; i < e->numTrees; i++) { \
            Node *node = e->nodes + root->offset; \
            evalers_f ## N [root->depth - 1](f, node, res); \
            root++; \
        } \
    }


EVAL_FUNC(1)
EVAL_FUNC(8)
EVAL_FUNC(32)

// nfeat - the number of features in each feature vector
// nvec - the number of feature vectors
void eval(const EnsembleHandle handle, jfloat *f[], const jsize nfeat, const jsize nvec, jfloat* res) {
    Ensemble *e = (Ensemble *) handle;

    // Are the feature vectors long enough to be evaluated?
    if (nfeat <= e->maxFid) {
        for (int i = 0; i < nvec; i++) {
            res[i] = nanf("");
        }
    // See if we can use an unrolled implementation. If it was
    // really important we could generate many more implementations
    // and use a jump table like the functions themselves do .. but
    // it seems fine to specialize only a few sizes and let everything
    // else "slow" path.
    } else if (nvec == 32) {
        eval_f32(e, f, res);
    } else if (nvec == 8) {
        eval_f8(e, f, res);
    // Fallback to one at a time. This could probably 
    // be improved though to push the loop to per-tree,
    // rather than per-vector though. Possibly per-tree
    // would be more cache friendly (check with perf?).
    } else {
        for (int i = 0; i < nvec; i++) {
            eval_f1(e, &f[i], &res[i]);
        }
    }
}

