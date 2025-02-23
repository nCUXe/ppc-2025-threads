#include "seq/bessonov_e_radix_sort_simple_merging/include/ops_seq.hpp"

bool bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_.resize(output_size);

  return true;
}

bool bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential::RunImpl() {
  size_t n = input_.size();
  std::vector<uint64_t> bits(n);

  for (size_t i = 0; i < n; i++) {
    uint64_t b;
    std::memcpy(&b, &input_[i], sizeof(double));
    if (b & (1ULL << 63)) {
      b = ~b;
    } else {
      b ^= (1ULL << 63);
    }
    bits[i] = b;
  }

  const int RADIX = 256;
  const int PASSES = 8;
  std::vector<uint64_t> temp(n);

  for (int pass = 0; pass < PASSES; pass++) {
    int shift = pass * 8;
    std::vector<size_t> count(RADIX, 0);

    for (size_t i = 0; i < n; i++) {
      int digit = (bits[i] >> shift) & 0xFF;
      count[digit]++;
    }
    for (int i = 1; i < RADIX; i++) {
      count[i] += count[i - 1];
    }
    for (int i = n - 1; i >= 0; i--) {
      int digit = (bits[i] >> shift) & 0xFF;
      temp[--count[digit]] = bits[i];
    }
    bits.swap(temp);
  }

  for (size_t i = 0; i < n; i++) {
    uint64_t b = bits[i];
    if (b & (1ULL << 63)) {
      b ^= (1ULL << 63);
    } else {
      b = ~b;
    }
    double d;
    std::memcpy(&d, &b, sizeof(double));
    output_[i] = d;
  }

  return true;
}

bool bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}