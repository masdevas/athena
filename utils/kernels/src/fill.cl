#include "utils.h"

kernel void ffill(float pattern, global float* out) {
  out[get_global_id_linear()] = pattern;
}
