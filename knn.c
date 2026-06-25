#include <time.h>

#define __KNN_IMPLEMENTATION__
#include "knn.h"

typedef struct {
    Matrix a0;
    Matrix w1, b1, a1;
    Matrix w2, b2, a2;
} Xor;

float data[] = {
    0,0,1,
    0,1,0,
    1,0,1,
    1,1,0
};


float forward(Xor m, float x1, float x2) {
    MATRIX_AT(m.a0, 0, 0) = x1;
    MATRIX_AT(m.a0, 0, 1) = x2;

    mat_dot(m.a1, m.a0, m.w1);
    mat_add (m.a1, m.b1);
    mat_sig(m.a1);
    mat_dot(m.a2, m.a1, m.w2);
    mat_add(m.a2, m.b2);
    mat_sig(m.a2);
    return *m.a2.p;
}

float cost(Xor m, Matrix mi, Matrix mo) {
    assert(mi.rows == mo.rows);
    for(size_t i = 0; i < mi.rows; i++) {
        mat_copy(m.a0, mat_row(mi, i));
    }

}

int main(){
    srand(time(0));
    // float x1 = float_rand();
    // float x2 = float_rand();
    size_t rows = sizeof(data)/sizeof(data[0])/3;
    printf("rows = %f\n", rows);
    Matrix mi = {
      .rows = rows,
      .cols = 2,
      .stride = 3,
      .p = data,
    };
    Matrix mo = {
        .rows = rows,
      .cols = 1,
      .stride = 3,
      .p = data + 2,
    };
    mat_print(mi);
    mat_print(mo);
    // Xor m;
    // m.a0 = malloc_matrix(1,2);
    // m.w1 = malloc_matrix(2,2);
    // m.b1 = malloc_matrix(1,2);
    // m.a1 = malloc_matrix(1,2);
    // m.w2 = malloc_matrix(2,1);
    // m.b2 = malloc_matrix(1,1);
    // m.a2 = malloc_matrix(1,1);

    // rand_matrix(m.w1);
    // rand_matrix(m.w2);
    // rand_matrix(m.b1);
    // rand_matrix(m.b2);

    // float result = forward(m, x1, x2);

    // printf("%f", result);

    return 0;
}