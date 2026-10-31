#!/bin/sh

set -xe

# clang -Wall -Wextra -o .bin/double double_a_number.c -lm
# clang -Wall -Wextra -o .bin/gate gate.c -lm
# clang -Wall -Wextra -o .bin/xor xor.c -lm

# clang -Wall -Wextra -o .bin/knn knn.c -lm
# clang -Wall -Wextra -o .bin/dump_knn dump_knn.c -lm
clang -Wall -Wextra -o .bin/adder adder.c -lm