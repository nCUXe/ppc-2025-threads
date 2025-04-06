#include "omp/bessonov_e_radix_sort_simple_merging/include/ops_omp.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace bessonov_e_radix_sort_simple_merging_omp {

bool TestTaskParallel::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  input_.resize(input_size);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(input_size); i++) {
    input_[i] = in_ptr[i];
  }

  unsigned int output_size = task_data->outputs_count[0];
  output_.resize(output_size);

  return true;
}

bool TestTaskParallel::ValidationImpl() {
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }

  if (task_data->inputs_count[0] > static_cast<size_t>(INT_MAX)) {
    return false;
  }

  return true;
}

static void ConvertDoubleToBits(std::vector<double>& input, std::vector<uint64_t>& bits) {
  int n = static_cast<int>(input.size());
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    uint64_t b = 0;
    std::memcpy(&b, &input[i], sizeof(double));
    if ((b & (1ULL << 63)) != 0ULL) {
      b = ~b;
    } else {
      b ^= (1ULL << 63);
    }
    bits[i] = b;
  }
}

static void CountDigits(const std::vector<uint64_t>& bits, int shift, std::vector<size_t>& count) {
  const int radix = 256;
#pragma omp parallel
  {
    std::vector<size_t> local_count(radix, 0);
#pragma omp for
    for (int i = 0; i < static_cast<int>(bits.size()); i++) {
      int digit = static_cast<int>((bits[i] >> shift) & 0xFF);
      local_count[digit]++;
    }
#pragma omp critical
    for (int i = 0; i < radix; i++) {
      count[i] += local_count[i];
    }
  }
}

static void ComputeOffsets(const std::vector<std::vector<size_t>>& thread_counts,
                           std::vector<std::vector<size_t>>& thread_offsets, std::vector<size_t>& count,
                           int num_threads, int radix) {
  for (int i = 1; i < radix; i++) {
    count[i] += count[i - 1];
  }

  for (int digit = 0; digit < radix; digit++) {
    size_t offset = (digit == 0) ? 0 : count[digit - 1];
    for (int t = 0; t < num_threads; t++) {
      thread_offsets[t][digit] = offset;
      offset += thread_counts[t][digit];
    }
  }
}

static void DistributeElements(std::vector<uint64_t>& bits, std::vector<uint64_t>& temp,
                               std::vector<std::vector<size_t>>& thread_offsets,
                               const std::vector<std::vector<uint64_t>>& thread_elements,
                               const std::vector<std::vector<int>>& thread_digits) {
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    std::vector<size_t> local_offsets = thread_offsets[thread_id];

    for (size_t i = 0; i < thread_elements[thread_id].size(); i++) {
      int digit = thread_digits[thread_id][i];
      size_t pos = local_offsets[digit];
      local_offsets[digit]++;
      temp[pos] = thread_elements[thread_id][i];
    }

    thread_offsets[thread_id] = local_offsets;
  }
  bits.swap(temp);
}

static void ConvertBitsToDouble(std::vector<uint64_t>& bits, std::vector<double>& output) {
  int n = static_cast<int>(bits.size());
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    uint64_t b = bits[i];
    if ((b & (1ULL << 63)) != 0ULL) {
      b ^= (1ULL << 63);
    } else {
      b = ~b;
    }
    double d = 0.0;
    std::memcpy(&d, &b, sizeof(double));
    output[i] = d;
  }
}

bool TestTaskParallel::RunImpl() {
  size_t n_size_t = input_.size();
  if (n_size_t > static_cast<size_t>(INT_MAX)) {
    return false;
  }

  int n = static_cast<int>(n_size_t);
  std::vector<uint64_t> bits(n);
  ConvertDoubleToBits(input_, bits);

  const int radix = 256;
  const int passes = 8;
  std::vector<uint64_t> temp(n);

  for (int pass = 0; pass < passes; pass++) {
    int shift = pass * 8;
    std::vector<size_t> count(radix, 0);
    CountDigits(bits, shift, count);

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<size_t>> thread_counts(num_threads, std::vector<size_t>(radix, 0));
    std::vector<std::vector<uint64_t>> thread_elements(num_threads);
    std::vector<std::vector<int>> thread_digits(num_threads);

#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      int chunk_size = n / num_threads;
      int start = thread_id * chunk_size;
      int end = (thread_id == num_threads - 1) ? n : start + chunk_size;

      thread_elements[thread_id].reserve(end - start);
      thread_digits[thread_id].reserve(end - start);

      for (int i = start; i < end; i++) {
        int digit = static_cast<int>((bits[i] >> shift) & 0xFF);
        thread_elements[thread_id].push_back(bits[i]);
        thread_digits[thread_id].push_back(digit);
        thread_counts[thread_id][digit]++;
      }
    }

    std::vector<std::vector<size_t>> thread_offsets(num_threads, std::vector<size_t>(radix, 0));
    ComputeOffsets(thread_counts, thread_offsets, count, num_threads, radix);
    DistributeElements(bits, temp, thread_offsets, thread_elements, thread_digits);
  }

  ConvertBitsToDouble(bits, output_);
  return true;
}

bool TestTaskParallel::PostProcessingImpl() {
  int n = static_cast<int>(output_.size());
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

}  // namespace bessonov_e_radix_sort_simple_merging_omp