#!/bin/sh
set -xe
clang -Wall -Wextra -o .bin/knn knn.c -lm