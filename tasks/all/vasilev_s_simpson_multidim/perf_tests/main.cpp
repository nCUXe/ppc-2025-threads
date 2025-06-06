#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "../include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
[[nodiscard]] std::tuple<std::size_t, vasilev_s_simpson_multidim::IntegrandFunction,
                         std::vector<vasilev_s_simpson_multidim::Bound>>
BuildTest() {
  return {400,
          [](const auto &coord) { return coord[0] + coord[1] + coord[2]; },
          {
              {.lo = 0., .hi = 1.},
              {.lo = 0., .hi = 1.},
              {.lo = 0., .hi = 1.},
          }};
}
}  // namespace

TEST(vasilev_s_simpson_multidim_all, test_pipeline_run) {
  boost::mpi::communicator world;

  auto [approxs, ifun, bounds] = BuildTest();
  double out{};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs = {reinterpret_cast<uint8_t *>(bounds.data()), reinterpret_cast<uint8_t *>(ifun),
                         reinterpret_cast<uint8_t *>(&approxs)};
    task_data->inputs_count.emplace_back(bounds.size());
    task_data->outputs = {reinterpret_cast<uint8_t *>(&out)};
    task_data->outputs_count.emplace_back(1);
  } else {
    task_data->inputs = {reinterpret_cast<uint8_t *>(ifun)};
  }

  auto task = std::make_shared<vasilev_s_simpson_multidim::SimpsonTaskAll>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(vasilev_s_simpson_multidim_all, test_task_run) {
  boost::mpi::communicator world;

  auto [approxs, ifun, bounds] = BuildTest();
  double out{};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs = {reinterpret_cast<uint8_t *>(bounds.data()), reinterpret_cast<uint8_t *>(ifun),
                         reinterpret_cast<uint8_t *>(&approxs)};
    task_data->inputs_count.emplace_back(bounds.size());
    task_data->outputs = {reinterpret_cast<uint8_t *>(&out)};
    task_data->outputs_count.emplace_back(1);
  } else {
    task_data->inputs = {reinterpret_cast<uint8_t *>(ifun)};
  }

  auto task = std::make_shared<vasilev_s_simpson_multidim::SimpsonTaskAll>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
