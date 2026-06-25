#ifndef __KNN__
#define __KNN__
#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef _KNN_MALLOC_
#include <stdlib.h>
#define _KNN_MALLOC_ malloc
#endif

#ifndef _KNN_ASSERT_
#include <assert.h>
#define _KNN_ASSERT_ assert
#endif

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float* p;
} Matrix;

#define MATRIX_AT(m, i, j) (m).p[(i)*(m).stride + (j)]

Matrix malloc_matrix(size_t r, size_t c);
void rand_matrix(Matrix m);
void mat_dot(Matrix dst, Matrix a, Matrix b);
void mat_add(Matrix dst, Matrix a);
void mat_sig(Matrix m);
// Matrix mat_sub(Matrix m);
Matrix mat_row(Matrix m, size_t row);
void mat_copy(Matrix dst, Matrix src);
void mat_print(Matrix m);
float float_rand();
float sigmoidf(float x);



#endif

#ifdef __KNN_IMPLEMENTATION__
#include <time.h>
void mat_dot(Matrix dst, Matrix a, Matrix b) {
    _KNN_ASSERT_(a.cols == b.rows);
    _KNN_ASSERT_(dst.rows == a.rows);
    _KNN_ASSERT_(dst.cols == b.cols);
    for(size_t i = 0; i < dst.rows; i++) {
        for(size_t  j = 0; j < dst.cols; j++) {
            // a.cols == b.rows
            for(size_t k = 0; k < a.cols; k++) {
                MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j);
            }
        }
    }
    (void) dst;
    (void) a;
    (void) b;
}

void mat_add(Matrix dst, Matrix a) {
    _KNN_ASSERT_(dst.rows == a.rows);
    _KNN_ASSERT_(dst.cols == a.cols);
    for(size_t i = 0; i < dst.rows; i++) {
        for(size_t j = 0; j < dst.cols; j++) {
            MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, j);
        }
    }
}

void mat_print(Matrix m) {
    printf("[\n");
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            printf("    %f ", MATRIX_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}
Matrix malloc_matrix(size_t r, size_t c) {
    Matrix m;
    m.rows = r;
    m.cols = c;
    m.stride = c;
    m.p = _KNN_MALLOC_(sizeof(*m.p) * r * c);
    _KNN_ASSERT_(m.p != NULL);
    return m;
}

void rand_matrix(Matrix m) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MATRIX_AT(m, i, j) = float_rand();
        }
    }
}

float float_rand() {
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}
void mat_sig(Matrix m) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MATRIX_AT(m, i, j) = sigmoidf(MATRIX_AT(m, i, j));
        }
    }
}

Matrix mat_row(Matrix m, size_t row) {
    return (Matrix){
        .rows = 1,
        .cols = m.cols,
        .stride = m.cols,
        .p = &MATRIX_AT(m, row, 0),
    };
}

void mat_copy(Matrix dst, Matrix src) {
    _KNN_ASSERT_(dst.rows == src.rows);
    _KNN_ASSERT_(dst.cols == src.cols);
    for(size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
        }
    }
 }
#endif