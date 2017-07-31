#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "jni.h"

typedef void *EnsembleHandle;

EnsembleHandle createEnsemble(jsize numTrees, jsize numNodes);

void destroyEnsemble(EnsembleHandle handle);

jint addTree(EnsembleHandle handle, jbyte depth, jsize numNodes, jint fids[], jfloat threshOrScore[]);

/**
 * Evaluates multiple feature vectors against the provided handle. res must be
 * at least as large as f.  The size of f is specified by numVectors. The size
 * of each vector is specified by numFeatures
 */
void eval(const EnsembleHandle handle, jfloat *f[], const jsize numFeatures, const jsize numVectors, jfloat *res);

