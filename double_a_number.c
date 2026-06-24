#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train_set[][2] = {
  {0,0},
  {1,2},
  {2,4},
  {3,6},
  {4,8}
};

#define TRAIN_SIZE sizeof(train_set) / sizeof(train_set[0])

float random_float(void) {
  return (float)rand() / (float) RAND_MAX;
}

float cost_function(float w, float b) {
    float result = 0.0;
    for(size_t i = 0; i < TRAIN_SIZE; i++) {
        float x = train_set[i][0];
        float y = x * w + b;
        float d = y - train_set[i][1];
        result += d*d;
    }
    return result /= TRAIN_SIZE;
}

int main() {
    srand(0);
    float eps = 1e-3;
    float rate = 1e-3;
    // 拟合y=x*w , w就是只有一个参数的模型
    // update: 升级到y=x*w + b,新增b参数，所谓的偏移值
    float w = random_float() * 10.0;
    float b = random_float() * 5;
    for(size_t i = 0; i < 1000; i++) {
        float cost = cost_function(w, b);
        float dwcost = (cost_function(w + eps, b) - cost) / eps;
        float dbcost = (cost_function(w , b + eps) - cost) / eps;
        w -= rate * dwcost;
        b -= rate * dbcost;
        // w -= dcost;
        printf("w = %f, b = %f, %f\n",w, b, cost_function(w, b));
    }
    return 0;
}