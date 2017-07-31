#include <jni.h>
#include "TreeImpl.h"

jfieldID denseVectScores;

jint JNI_OnLoad(JavaVM *jvm, void *reserved) {
    JNIEnv *jenv;
    jint err = (*jvm)->GetEnv(jvm, (void **)&jenv, JNI_VERSION_1_8);
    if (err != JNI_OK) {
        return -1;
    }

    // This is a "local" reference, and will be free'd by jni after exiting
    jclass clazz = (*jenv)->FindClass(jenv, "com/o19s/es/ltr/ranker/DenseFeatureVector");
    if (clazz == NULL) {
        // Clear exception?
        return -1;
    }
    // This is an opaque type, and not an object reference. It doesn't need to
    // be free'd.
    denseVectScores = (*jenv)->GetFieldID(jenv, clazz, "scores", "[F");
    if (denseVectScores == NULL) {
        // Clear exception?
        return -1;
    }

    // What jni version is appropriate?
    return JNI_VERSION_1_8;
}

void JNI_OnUnload(JavaVM *jvm, void *reserved) {
    denseVectScores = NULL;
}

/*
 * Class:     com_o19s_es_ltr_ranker_dectree_JniNoBranchTree
 * Method:    createEnsemble
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_o19s_es_ltr_ranker_dectree_JniNoBranchTree_createEnsemble
  (JNIEnv *jenv, jclass jcls, jint numTrees, jint numNodes) {
    EnsembleHandle handle = createEnsemble(numTrees, numNodes);
    return (jlong) handle;
}

/*
 * Class:     com_o19s_es_ltr_ranker_dectree_JniNoBranchTree
 * Method:    destroyEnsemble
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_o19s_es_ltr_ranker_dectree_JniNoBranchTree_destroyEnsemble
  (JNIEnv *jenv, jclass jcls, jlong jhandle) {
    EnsembleHandle handle = (EnsembleHandle) jhandle;
    destroyEnsemble(handle);
}

/*
 * Class:     com_o19s_es_ltr_ranker_dectree_JniNoBranchTree
 * Method:    addTree
 * Signature: (JB[I[F)I
 */
JNIEXPORT jint JNICALL Java_com_o19s_es_ltr_ranker_dectree_JniNoBranchTree_addTree
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jbyte depth, jintArray jfids, jfloatArray jthreshOrScores) {

    jsize lenFids = (*jenv)->GetArrayLength(jenv, jfids);
    jsize lenThresh = (*jenv)->GetArrayLength(jenv, jthreshOrScores);
    if (lenFids != lenThresh) {
        return -1;
    }

    jint *fids = (*jenv)->GetIntArrayElements(jenv, jfids, NULL);
    if (fids == NULL) {
        // OOM. Clear exception? Docs aren't clear...
        // TODO: document some consistent return codes to use throughout...
        return -99;
    }
    jfloat *threshOrScores = (*jenv)->GetFloatArrayElements(jenv, jthreshOrScores, NULL);
    if (threshOrScores == NULL) {
        // OOM. Clear exception? Docs aren't clear...
        return -99;
    }

    EnsembleHandle handle = (EnsembleHandle) jhandle;
    int res = addTree(handle, depth, lenFids, fids, threshOrScores);

    (*jenv)->ReleaseIntArrayElements(jenv, jfids, fids, JNI_ABORT);
    (*jenv)->ReleaseFloatArrayElements(jenv, jthreshOrScores, threshOrScores, JNI_ABORT);

    return res;
}


#define MAX_VECS 32

/*
 * Class:     com_o19s_es_ltr_ranker_dectree_JniNoBranchTree
 * Method:    evalMultiple
 * Signature: (J[Lcom/o19s/es/ltr/ranker/DenseFeatureVector;[F)V
 */
JNIEXPORT jint JNICALL Java_com_o19s_es_ltr_ranker_dectree_JniNoBranchTree_evalMultiple
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jobjectArray jdenseVects, jfloatArray jres) {

  jsize numFeatVecs = (*jenv)->GetArrayLength(jenv, jdenseVects);
  if (numFeatVecs < 1) {
      // Really? Seems better to be strict and bail early.
      return 0;
  } else if (numFeatVecs > MAX_VECS) {
      // We could support this, but it's easier not to.
      return -3;
  }

  // array of pointers to feature vector objects. Stack allocated to
  // avoid alloc/free memory in a tight loop.
  jfloatArray vecs[MAX_VECS];
  // Pointers to the actual feature vectors.
  jfloat *f[MAX_VECS];
  // return value. 0 is success. -1 for inconsistent vector lengths. -2 for OOM. -4 for more vectors than supported
  jint ret = 0;
  // evaluation results
  jfloat *res = NULL;
  // number of features per feature vector
  jsize numFeats = -1;

  // Get all the data out of the DenseFeatureVector objects and into
  // something we can work with.
  for (int i = 0; i < numFeatVecs; i++) {
    jobject vect = (*jenv)->GetObjectArrayElement(jenv, jdenseVects, i);
    vecs[i] = (jfloatArray) (*jenv)->GetObjectField(jenv, vect, denseVectScores);
    if (i == 0) {
        numFeats = (*jenv)->GetArrayLength(jenv, vecs[i]);
    } else {
        // assert same size? Seems safest.
        if (numFeats != (*jenv)->GetArrayLength(jenv, vecs[i])) {
            ret = -1;
            // We need to release the feature vectors we already declared a critical
            // sections on, along with freeing the **f array.
            numFeatVecs = i;
            goto evalMultiple_cleanup_vecs;
        }
    }
    // One problem with critical array sections is it basically disables GC
    // while we are running.  But it does try really hard to get us a pointer
    // to the jvm data instead of copying it off-heap. It seems possible
    // that servers constantly chugging on ltr in many threads will not have as
    // many gc opportunities ... needs testing.
    //
    // No clue if its still true...but a 2007 posting to hotspot-runtime-dev
    // list list said:
    //   Basically, the way things work now is that once Eden fills up, and a
    //   JNI critical section is in use, further entries into the critical
    //   section are delayed -- thus there is not a danger of starvation or
    //   lock out
    //
    // We should probably at least benchmark the difference between critical
    // sections and letting the jvm copy via GetFloatArrayElements.
    f[i] = (*jenv)->GetPrimitiveArrayCritical(jenv, jfeatures, NULL);
    if (f[i] == NULL) {
        // out of memory exception thrown
        ret = -2;
        numFeatVecs = i;
        goto evalMultiple_cleanup_vecs;
    }
  }

  // Get ahold of the array results will be stored into. 
  // The java side is responsible for zeroing this out before
  // passing it in.
  float *res = (*jenv)->GetPrimitiveArrayCritical(jenv, jres, NULL);
  if (res == NULL) {
    // OOM
    ret = -2;
    goto evalMultiple_cleanup_res;
  }

  // Finally we can evaluate the vectors and store the result.
  EnsembleHandle handle = (EnsembleHandle) jhandle;
  eval(handle, f, numFeats, numFeatVecs, res);

evalMultiple_cleanup_res:
  // 0 as last arg copys data back and frees res, if it was allocated and not a direct pointer.
  (*jenv)->ReleasePrimitiveArrayCritical(jenv, jres, res, 0);

evalMultiple_cleanup_vecs:
  for (int i = 0; i < numFeatVecs; i++) {
    (*jenv)->ReleasePrimitiveArrayCritical(jenv, vecs[i], f[i], JNI_ABORT);
  }

  return ret;
}


/*
 * Class:     com_o19s_es_ltr_ranker_dectree_JniNoBranchTree
 * Method:    eval
 * Signature: (J[F)F
 */
JNIEXPORT jfloat JNICALL Java_com_o19s_es_ltr_ranker_dectree_JniNoBranchTree_eval
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jfloatArray jfeatures) {

    jsize len = (*jenv)->GetArrayLength(jenv, jfeatures);
    jfloat *features = (*jenv)->GetPrimitiveArrayCritical(jenv, jfeatures, NULL);
    if (features == NULL) {
        // we have no good way to report an error .. but i think
        // just returning will propogate an uncleared exception?
        // needs testing.
        return 0;
    }

    EnsembleHandle handle = (EnsembleHandle) jhandle;
    jfloat *f[] = {features};
    jfloat res[1];
    eval(handle, f, len, 1, &res);

    (*jenv)->ReleasePrimitiveArrayCritical(jenv, jfeatures, features, JNI_ABORT);
    return res[0];
}

