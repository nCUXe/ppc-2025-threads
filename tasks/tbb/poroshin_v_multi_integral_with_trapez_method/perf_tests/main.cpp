#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/poroshin_v_multi_integral_with_trapez_method/include/ops_tbb.hpp"

namespace {
double F3advanced(std::vector<double> &arguments) {
  return sin(arguments.at(0)) * tan(arguments.at(1)) * log(arguments.at(2));
}
}  // namespace

TEST(poroshin_v_multi_integral_with_trapez_method_tbb, test_pipeline_run) {
  std::vector<int> n = {220, 220, 220};
  std::vector<double> a = {0.8, 1.9, 2.9};
  std::vector<double> b = {1.0, 2.0, 3.0};
  std::vector<double> out(1);
  double eps = 1e-6;
  std::shared_ptr<ppc::core::TaskData> task_omp = std::make_shared<ppc::core::TaskData>();
  task_omp->inputs_count.emplace_back(n.size());
  task_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_omp->outputs_count.emplace_back(out.size());
  auto test_task_omp =
      std::make_shared<poroshin_v_multi_integral_with_trapez_method_tbb::TestTaskTBB>(task_omp, F3advanced);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_NEAR(-0.00427191467841401, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_tbb, test_task_run) {
  std::vector<int> n = {220, 220, 220};
  std::vector<double> a = {0.8, 1.9, 2.9};
  std::vector<double> b = {1.0, 2.0, 3.0};
  std::vector<double> out(1);
  double eps = 1e-6;
  std::shared_ptr<ppc::core::TaskData> task_omp = std::make_shared<ppc::core::TaskData>();
  task_omp->inputs_count.emplace_back(n.size());
  task_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_omp->outputs_count.emplace_back(out.size());
  auto test_task_omp =
      std::make_shared<poroshin_v_multi_integral_with_trapez_method_tbb::TestTaskTBB>(task_omp, F3advanced);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_NEAR(-0.00427191467841401, out[0], eps);
}
