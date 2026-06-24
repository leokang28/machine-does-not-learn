#!/bin/sh

set -xe

clang -Wall -Wextra -o .bin/double double_a_number.c -lm
clang -Wall -Wextra -o .bin/gate gate.c -lm
clang -Wall -Wextra -o .bin/xor xor.c -lm