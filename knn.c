#include <time.h>

#define __KNN_IMPLEMENTATION__
#include "knn.h"
float data[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0
};

void verify(KNN n) {
    for(size_t i = 0; i < 2; i+= 1) {
        for(size_t j = 0; j < 2; j += 1) {
            MATRIX_AT(KNN_INPUT(n), 0, 0) = i; 
            MATRIX_AT(KNN_INPUT(n), 0, 1) = j;
            knn_forward(n);
            float y = MATRIX_AT(KNN_OUTPUT(n), 0, 0);
            
            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
}

int main(){
    srand(time(0));

    size_t arch[] = {2, 2, 1};
    KNN n = knn_alloc(arch, ARR_LEN(arch));
    KNN g = knn_alloc(arch, ARR_LEN(arch));
    knn_rand(n);
    
    size_t rows = sizeof(data)/sizeof(data[0])/3;
    Matrix ti = {
        .rows = rows,
        .cols = 2,
        .stride = 3,
        .p = data,
    };
    Matrix to = {
        .rows = rows,
        .cols = 1,
        .stride = 3,
        .p = data + 2,
    };
    
    // float eps = 1e-1;
    float rate = 1e-1;
    for(size_t i = 0; i < 100*1000; i += 1) {
        // knn_finite_diff(n, g, eps, ti, to);
        knn_backprop(n, g, ti, to);
        knn_learn(n, g, rate);
    }

    KNN_PRINT(n);
    verify(n);
    return 0;
}
