#include "all/bessonov_e_radix_sort_simple_merging/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace bessonov_e_radix_sort_simple_merging_all {

void TestTaskALL::ConvertDoubleToBits(const std::vector<double>& input, std::vector<uint64_t>& bits, size_t start,
  size_t end) {
  for (size_t i = start; i < end; ++i) {
    uint64_t b = 0;
    std::memcpy(&b, &input[i], sizeof(double));
    b ^= (-static_cast<int64_t>(b >> 63) | (1ULL << 63));
    bits[i] = b;
  }
}

void TestTaskALL::ConvertBitsToDouble(const std::vector<uint64_t>& bits, std::vector<double>& output, size_t start,
  size_t end) {
  for (size_t i = start; i < end; ++i) {
    uint64_t b = bits[i];
    b ^= (((b >> 63) - 1) | (1ULL << 63));
    double d = NAN;
    std::memcpy(&d, &b, sizeof(double));
    output[i] = d;
  }
}

void TestTaskALL::RadixSortPass(std::vector<uint64_t>& bits, std::vector<uint64_t>& temp, int shift) {
  constexpr int kRadix = 256;
  const size_t n = bits.size();
  std::array<size_t, kRadix> count{};

  for (size_t i = 0; i < n; ++i) {
    count[(bits[i] >> shift) & 0xFF]++;
  }

  size_t total = 0;
  for (int i = 0; i < kRadix; ++i) {
    size_t old_count = count[i];
    count[i] = total;
    total += old_count;
  }

  for (size_t i = 0; i < n; ++i) {
    uint8_t digit = (bits[i] >> shift) & 0xFF;
    temp[count[digit]++] = bits[i];
  }

  bits.swap(temp);
}

bool TestTaskALL::PreProcessingImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    input_.resize(task_data->inputs_count[0]);
    auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    std::copy(in_ptr, in_ptr + task_data->inputs_count[0], input_.begin());
  }

  size_t total_size = input_.size();
  MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  size_t local_size = total_size / size;
  if (rank < total_size % size) {
    local_size++;
  }

  std::vector<double> local_data(local_size);
  std::vector<int> counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      counts[i] = total_size / size + (i < total_size % size ? 1 : 0);
      displs[i] = i == 0 ? 0 : displs[i - 1] + counts[i - 1];
    }
  }

  MPI_Scatterv(input_.data(), counts.data(), displs.data(), MPI_DOUBLE,
    local_data.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  input_ = std::move(local_data);
  output_.resize(input_.size());

  return true;
}

bool TestTaskALL::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (task_data->inputs.empty() || task_data->outputs.empty()) {
      return false;
    }

    if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
      return false;
    }

    if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
      return false;
    }

    if (task_data->inputs_count[0] == 0) {
      return false;
    }

    if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
      return false;
    }

    if (task_data->inputs_count[0] > static_cast<size_t>(std::numeric_limits<int>::max())) {
      return false;
    }
  }

  int valid = 1;
  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid != 0;
}

bool TestTaskALL::RunImpl() {
  const size_t n = input_.size();
  if (n == 0) {
    return true;
  }

  std::vector<uint64_t> bits(n);
  std::vector<uint64_t> temp(n);

  size_t num_threads = ppc::util::GetPPCNumThreads();
  num_threads = std::max<size_t>(1, num_threads);
  const size_t block_size = (n + num_threads - 1) / num_threads;

  {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
      size_t start = i * block_size;
      size_t end = std::min(start + block_size, n);
      if (start >= n) {
        break;
      }
      threads.emplace_back(ConvertDoubleToBits, std::cref(input_), std::ref(bits), start, end);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  constexpr int kPasses = sizeof(uint64_t);
  for (int pass = 0; pass < kPasses; ++pass) {
    RadixSortPass(bits, temp, pass * 8);
  }

  {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
      size_t start = i * block_size;
      size_t end = std::min(start + block_size, n);
      if (start >= n) {
        break;
      }
      threads.emplace_back(ConvertBitsToDouble, std::cref(bits), std::ref(output_), start, end);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  int local_count = static_cast<int>(output_.size());

  MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }

  std::vector<double> global_result;
  if (rank == 0) {
    global_result.resize(task_data->outputs_count[0]);
  }

  MPI_Gatherv(output_.data(), local_count, MPI_DOUBLE,
    global_result.data(), counts.data(), displs.data(), MPI_DOUBLE,
    0, MPI_COMM_WORLD);

  if (rank == 0) {
    output_ = std::move(global_result);
  }

  return true;
}

bool TestTaskALL::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(output_.begin(), output_.end(), out_ptr);
  }

  return true;
}
}  // namespace bessonov_e_radix_sort_simple_merging_all