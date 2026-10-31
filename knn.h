#ifndef __KNN__
#define __KNN__
#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef KNN_MALLOC
#include <stdlib.h>
#define KNN_MALLOC malloc
#endif

#ifndef KNN_ASSERT
#include <assert.h>
#define KNN_ASSERT assert
#endif

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float* p;
} Matrix;

#define MATRIX_AT(m, i, j) (m).p[(i)*(m).stride + (j)]

Matrix malloc_matrix(size_t r, size_t c);
void rand_matrix(Matrix m, int low, int high);
void mat_fill(Matrix m, float v);
void mat_dot(Matrix dst, Matrix a, Matrix b);
void mat_add(Matrix dst, Matrix a);
void mat_sig(Matrix m);
// Matrix mat_sub(Matrix m);
Matrix mat_row(Matrix m, size_t row);
void mat_copy(Matrix dst, Matrix src);
void mat_print(Matrix m);
float float_rand();
float sigmoidf(float x);

typedef struct {
    size_t count; // 层数
    Matrix* weights; // 权重
    Matrix* biases; // 偏置
    Matrix* activations; // 每一层的输入输出数据，上一层的输出是下一层的输入，
                    // 这个数组会额外保存初始输入，所以长度会大于weights和bias

} KNN;

KNN knn_alloc(size_t *arch, size_t total_count);
void knn_print(KNN n, char* name);
void knn_rand(KNN n, int low, int high);
void knn_forward(KNN n);
float knn_cost(KNN n, Matrix ti, Matrix to);
void knn_finite_diff(KNN n, KNN g, float eps, Matrix ti, Matrix to);
void knn_zero(KNN n);
void knn_backprop(KNN n, KNN g, Matrix ti, Matrix to);
void knn_learn(KNN n, KNN g, float rate);
#define KNN_PRINT(n) knn_print(n, #n);
#define ARR_LEN(n) sizeof((n)) / sizeof((n)[0])
#define KNN_INPUT(n) (n).activations[0]
#define KNN_OUTPUT(n) (n).activations[n.count]


#endif

#ifdef __KNN_IMPLEMENTATION__
#include <time.h>
void mat_dot(Matrix dst, Matrix a, Matrix b) {
    KNN_ASSERT(a.cols == b.rows);
    KNN_ASSERT(dst.rows == a.rows);
    KNN_ASSERT(dst.cols == b.cols);
    for(size_t i = 0; i < dst.rows; i++) {
        for(size_t  j = 0; j < dst.cols; j++) {
            MATRIX_AT(dst, i, j) = 0.f;
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
    KNN_ASSERT(dst.rows == a.rows);
    KNN_ASSERT(dst.cols == a.cols);
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
    m.p = KNN_MALLOC(sizeof(*m.p) * r * c);
    KNN_ASSERT(m.p != NULL);
    return m;
}

void rand_matrix(Matrix m, int low, int high) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            MATRIX_AT(m, i, j) = float_rand() * (high - low) + low;
        }
    }
}

void mat_fill(Matrix m, float v) {
    for(size_t i = 0; i < m.rows; i += 1) {
        for(size_t j = 0; j < m.cols; j += 1) {
            MATRIX_AT(m, i ,j) = v;
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
    KNN_ASSERT(dst.rows == src.rows);
    KNN_ASSERT(dst.cols == src.cols);
    for(size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
        }
    }
}
KNN knn_alloc(size_t *arch, size_t total_count) {
    KNN n;
    n.count = total_count - 1;

    n.weights = KNN_MALLOC(sizeof(*n.weights) * n.count);
    KNN_ASSERT(n.weights != NULL);
    n.biases = KNN_MALLOC(sizeof(*n.biases) * n.count);
    KNN_ASSERT(n.biases != NULL);
    n.activations = KNN_MALLOC(sizeof(*n.activations) * (n.count + 1));
    KNN_ASSERT(n.activations != NULL);

    KNN_INPUT(n) = malloc_matrix(1, arch[0]);
    for(size_t i = 1; i < total_count; i += 1) {
        n.weights[i - 1] = malloc_matrix(n.activations[i - 1].cols, arch[i]);
        n.biases[i - 1] = malloc_matrix(1, arch[i]);
        n.activations[i] = malloc_matrix(1, arch[i]);
    }

    return n;
}
void knn_print(KNN n, char* name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < n.count; i += 1) {
        mat_print(n.weights[i]);
        mat_print(n.biases[i]);
    }
    printf("]\n");
}

void knn_rand(KNN n, int low, int high) {
    for(size_t i = 0; i < n.count; i += 1) {
        rand_matrix(n.weights[i], low, high);
        rand_matrix(n.biases[i], low, high);
    }
}
void knn_forward(KNN n) {
    for (size_t i = 0; i < n.count; i += 1) {
        mat_dot(n.activations[i + 1], n.activations[i], n.weights[i]);
        mat_add(n.activations[i + 1], n.biases[i]);
        mat_sig(n.activations[i + 1]);
    }
}
float knn_cost(KNN n, Matrix ti, Matrix to) {
    KNN_ASSERT(ti.rows == to.rows);
    float cost = 0.f;
    for(size_t i = 0; i < ti.rows; i++) {
        Matrix x = mat_row(ti, i);
        Matrix y = mat_row(to, i);
        mat_copy(KNN_INPUT(n), x);
        knn_forward(n);
        for(size_t j = 0; j < to.cols; j++) {
            float d = MATRIX_AT(KNN_OUTPUT(n), 0, j) - MATRIX_AT(y, 0, j);
            cost += d * d;
        }

    }

    return cost / ti.rows;
}

void knn_finite_diff(KNN n, KNN g, float eps, Matrix ti, Matrix to) {
    float saved;
    // weights
    for (size_t i = 0; i < n.count; i += 1) {
        float c = knn_cost(n, ti, to);
        for (size_t j = 0; j < n.weights[i].rows; j += 1) {
            for(size_t k = 0; k < n.weights[i].cols; k += 1) {
                saved = MATRIX_AT(n.weights[i], j, k);
                MATRIX_AT(n.weights[i], j, k) += eps;
                MATRIX_AT(g.weights[i], j, k) = (knn_cost(n, ti, to) - c )/ eps;
                MATRIX_AT(n.weights[i], j, k) = saved;
            }
        }
        for (size_t j = 0; j < n.biases[i].rows; j += 1) {
            for(size_t k = 0; k < n.biases[i].cols; k += 1) {
                saved = MATRIX_AT(n.biases[i], j, k);
                MATRIX_AT(n.biases[i], j, k) += eps;
                MATRIX_AT(g.biases[i], j, k) = (knn_cost(n, ti, to) - c )/ eps;
                MATRIX_AT(n.biases[i], j, k) = saved;
            }
        }
    }
}
void knn_zero(KNN n) {
    for(size_t i = 0; i < n.count; i += 1) {
        mat_fill(n.weights[i], 0);
        mat_fill(n.biases[i], 0);
        mat_fill(n.activations[i], 0);
    }
    mat_fill(n.activations[n.count], 0);
}
void knn_backprop(KNN n, KNN g, Matrix ti, Matrix to) {
    KNN_ASSERT(ti.rows == to.rows);
    size_t samples_count = ti.rows;
    knn_zero(g);
    for(size_t i = 0; i < samples_count; i += 1) {
        mat_copy(KNN_INPUT(n), mat_row(ti, i));
        knn_forward(n);

        for(size_t l = 0; l <= n.count; l += 1) {
            mat_fill(g.activations[l], 0);
        }

        for (size_t j = 0; j < to.cols; j += 1) {
            // 向前推理后计算推理值和真实值的差，存到差值网络的output里
            // 系数2来自代价函数 d^2 的求导，只在输出层播种时出现一次
            MATRIX_AT(KNN_OUTPUT(g), 0, j) = (MATRIX_AT(KNN_OUTPUT(n), 0, j) - MATRIX_AT(to, i, j));
        }

        // backprop
        for(size_t l = n.count; l > 0; l -= 1) {
            for (size_t j = 0; j < n.activations[l].cols; j += 1) {
                float a = MATRIX_AT(n.activations[l], 0, j);
                float da = MATRIX_AT(g.activations[l], 0, j);
                MATRIX_AT(g.biases[l - 1], 0, j) += 2*da*a*(1 - a);
                for(size_t k = 0; k < n.activations[l - 1].cols; k += 1) {
                    float pa = MATRIX_AT(n.activations[l - 1], 0, k);
                    float pw = MATRIX_AT(n.weights[l - 1], k, j);
                    MATRIX_AT(g.weights[l - 1], k, j) += 2*da*a*(1 - a) * pa;
                    MATRIX_AT(g.activations[l - 1], 0, k) += 2*da*a*(1 - a) * pw;
                }
             }
        }
    }
    for(size_t i = 0; i < g.count; i += 1) {
        for(size_t j = 0; j < g.weights[i].rows; j += 1) {
            for(size_t k = 0; k < g.weights[i].cols; k += 1) {
                MATRIX_AT(g.weights[i], j, k) /= samples_count;
            }
        }

        for(size_t j = 0; j < g.biases[i].rows; j += 1) {
            for(size_t k = 0; k < g.biases[i].cols; k += 1) {
                MATRIX_AT(g.biases[i], j, k) /= samples_count;
            }
        }
    }
}

void knn_learn(KNN n, KNN g, float rate) {
    for(size_t i = 0; i < n.count; i += 1) {
        for (size_t j = 0; j < n.weights[i].rows; j += 1) {
            for(size_t k = 0; k < n.weights[i].cols; k += 1) {
                MATRIX_AT(n.weights[i], j, k) -= rate * MATRIX_AT(g.weights[i], j, k);
            }
        }

        for (size_t j = 0; j < n.biases[i].rows; j += 1) {
            for(size_t k = 0; k < n.biases[i].cols; k += 1) {
                MATRIX_AT(n.biases[i], j, k) -= rate * MATRIX_AT(g.biases[i], j, k);
            }
        }
    }
}
#endif