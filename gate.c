#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef float train_set[3];

float sigmoidf(float x) {
    return 1.f/(1.f + expf(-x));
}

// or
train_set or_gate_train_set[] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,1}
};

// and
train_set and_gate_train_set[] = {
    {0,0,0},
    {0,1,0},
    {1,0,0},
    {1,1,1}
};

// nand
train_set nand_gate_train_set[] = {
    {0,0,1},
    {0,1,1},
    {1,0,1},
    {1,1,0}
};

train_set *dataset = or_gate_train_set;

#define TRAIN_SIZE 4

float cost_function(float w1, float w2, float b) {
    float result = 0.0;
    for(size_t i = 0; i < TRAIN_SIZE; i++) {
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float y = sigmoidf(x1 * w1 + x2 * w2  + b);
        float d = y - dataset[i][2];
        result += d*d;
    }
    return result /= TRAIN_SIZE;
}

float random_float(void) {
  return (float)rand() / (float) RAND_MAX;
}

// void train_or() {}

int main(void){
    srand(time(0));
    float w1 = random_float();
    float w2 = random_float();
    float b = random_float();

    float eps = 1e-1;
    float rate = 1e-1;
    for (size_t i = 0; i < 1000000; i++) {
        // cost_function()
        float cost = cost_function(w1, w2, b);
        float dw1 = (cost_function(w1 + eps, w2, b) - cost) / eps;
        float dw2 = (cost_function(w1, w2 + eps, b) - cost) / eps;
        float db = (cost_function(w1, w2, b + eps) - cost) / eps;
        
        w1 -= dw1 * rate;
        w2 -= dw2 * rate;
        b -= db * rate;
    }
    printf("c = %f, w1 = %f, w2 = %f, b = %f\n", cost_function(w1, w2, b), w1, w2, b);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("%zu | %zu = %f\n", i, j, sigmoidf(i * w1 + j * w2 + b));
        }
    }
}