#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include "TreeImpl.h"
#include "jni.h"

int randint(int n) {
    if ((n - 1) == RAND_MAX) {
        return rand();
    } else {
        long end = RAND_MAX / n;
        end *= n;

        int r;
        while ((r = rand()) >= end);

        return r % n;
    }
}

float randfloat(float min, float max) {
    float scale = rand() / (float) RAND_MAX;
    return min + scale * (min - max);
}

double timespecToSeconds(struct timespec *ts) {
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

int main(int argc, char *argv[]) {
    const int nfeat = 100;
    const int ndoc = 40000;
    const int depth=8;
    const int trees = 1000;
    const int len = (1<<(depth+1))-1;
    EnsembleHandle handle = createEnsemble(trees, len*trees);

    jint fids[len];
    jfloat threshOrScore[len];

    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);

    for (int i = 0; i < trees; i++) {
        for (int j = 0; j < len; j++) {
            fids[j] = randint(nfeat);
            threshOrScore[j] = randfloat(0, 100);
        }
        jint res = addTree(handle, depth, len, fids, threshOrScore);
        if (res != 1) {
            printf("Failed creating tree %d\n", i);
            return res;
        }
    }

    jfloat features[ndoc][nfeat];
    for (int i = 0; i < ndoc; i++) {
        for (int j = 0; j < nfeat; j++) {
            features[i][j] = randfloat(0, 100);
        }
    }

    // This has to be a define, rather than a var, so the *f[NVEC] array initializes correctly
    #define NVEC 32
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    jfloat res[NVEC];
    int i = (NVEC - 1);
    for (; i < ndoc; i += NVEC) {
        jfloat *f[NVEC] = {
            features[i-31], features[i-30], features[i-29], features[i-28], features[i-27], features[i-26], features[i-25], features[i-24],
            features[i-23], features[i-22], features[i-21], features[i-20], features[i-19], features[i-18], features[i-17], features[i-16],
            features[i-15], features[i-14], features[i-13], features[i-12], features[i-11], features[i-10], features[i-9], features[i-8],
            features[i-7], features[i-6], features[i-5], features[i-4], features[i-3], features[i-2], features[i-1], features[i]
        };
        eval(handle, f, nfeat, NVEC, res);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    printf("took: %f\n", timespecToSeconds(&end) - timespecToSeconds(&start));
    return 0;
}
