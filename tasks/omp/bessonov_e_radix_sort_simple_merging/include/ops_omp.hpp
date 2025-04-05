#pragma once

#include <omp.h>

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_radix_sort_simple_merging_omp {

class TestTaskParallel : public ppc::core::Task {
 public:
  explicit TestTaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
};

}  // namespace bessonov_e_radix_sort_simple_merging_omp