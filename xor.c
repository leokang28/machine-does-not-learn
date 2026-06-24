#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
} Xor;

typedef float xor_dataset[3];
xor_dataset dataset[] = {
    {0,0,1},
    {0,1,0},
    {1,0,1},
    {1,1,0}
};
#define TRAIN_SIZE 4

float sigmoid(float x) {
    return 1.f/(1.f + expf(-x));
}
float forward(Xor m, float x1, float x2) {
    float a = sigmoid(m.or_w1 * x1 + m.or_w2 * x2 + m.or_b);
    float b = sigmoid(m.nand_w1 * x1 + m.nand_w2 * x2 + m.nand_b);
    return sigmoid(m.and_w1 * a  + m.and_w2 * b + m.and_b);
}
float cost_function(Xor m) {
float result = 0.0;
    for(size_t i = 0; i < TRAIN_SIZE; i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float y = forward(m, x1, x2);
        float d = y - dataset[i][2];
        result += d*d;
    }
    return result /= TRAIN_SIZE;
}
float random_float(void) {
  return (float)rand() / (float) RAND_MAX;
}
void print_Xor(Xor m) {
    printf("or_w1:   %f\n", m.or_w1);
    printf("or_w2:   %f\n", m.or_w2);
    printf("or_b:    %f\n", m.or_b);
    printf("and_w1:  %f\n", m.and_w1);
    printf("and_w2:  %f\n", m.and_w2);
    printf("and_b:   %f\n", m.and_b);
    printf("nand_w1: %f\n", m.nand_w1);
    printf("nand_w2: %f\n", m.nand_w2);
    printf("nand_b:  %f\n", m.nand_b);
}

Xor random_xor() {
    Xor m;
    m.or_w1 = random_float();
    m.or_w2 = random_float();
    m.or_b = random_float();
    m.nand_w1 = random_float();
    m.nand_w2 = random_float();
    m.nand_b = random_float();
    m.and_w1 = random_float();
    m.and_w2 = random_float();
    m.and_b = random_float();
    return m;
}

Xor finite_diff(Xor m, float eps) {
    float c = cost_function(m);
    float saved;
    Xor diff;

    saved = m.or_w1;
    m.or_w1 += eps;
    diff.or_w1 = (cost_function(m) - c)/eps; 
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    diff.or_w2 = (cost_function(m) - c)/eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    diff.or_b = (cost_function(m) - c)/eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    diff.nand_w1 = (cost_function(m) - c)/eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    diff.nand_w2 = (cost_function(m) - c)/eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    diff.nand_b = (cost_function(m) - c)/eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    diff.and_w1 = (cost_function(m) - c)/eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    diff.and_w2 = (cost_function(m) - c)/eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    diff.and_b = (cost_function(m) - c)/eps;
    m.and_b = saved;

    return diff;

}

Xor train(Xor m, Xor d, float rate) {
    m.or_w1 -= d.or_w1 * rate;
    m.or_w2 -= d.or_w2 * rate;
    m.or_b -= d.or_b * rate;
    m.nand_w1 -= d.nand_w1 * rate;
    m.nand_w2 -= d.nand_w2 * rate;
    m.nand_b -= d.nand_b * rate;
    m.and_w1 -= d.and_w1 * rate;
    m.and_w2 -= d.and_w2 * rate;
    m.and_b -= d.and_b * rate;
    return m;
}

int main() {
    srand(time(0));
    Xor m = random_xor();
    float eps = 1e-1;
    float rate = 1e-2;

    for (size_t i = 0; i < 1000*500; i++) {
        Xor d = finite_diff(m, eps);
        m = train(m, d, rate);
    }
    
    // 训练结束：
    for (size_t i = 0; i < 2; i++) {
        for(size_t j = 0; j < 2; j++) {
            printf("%zu xor %zu = %f\n", i, j, forward(m, i, j));
        }
    }

    return 0;
}