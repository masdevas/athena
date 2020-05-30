#pragma once

int get_global_id_linear() {
  int workDim = get_work_dim();
  if (workDim == 1) {
    return get_global_id(0);
  } else if (workDim == 2) {
    return get_global_id(1) + get_global_size(1) * get_global_id(0);
  } else if (workDim == 3) {
    return get_global_id(2) + get_global_size(2) * get_global_id(1) +
           get_global_size(2) * get_global_size(1) * get_global_id(0);
  }

  return 0;
}
