#include <gtest/gtest.h>

#include "core/task/include/task.hpp"
#include "seq/bessonov_e_radix_sort_simple_merging/include/ops_seq.hpp"


TEST(bessonov_e_radix_sort_simple_merging_seq, OrdinaryTest) {
  std::vector<double> input_vector = {3.4, 1.2, 0.5, 7.8, 2.3, 4.5, 6.7, 8.9, 1.0, 0.2, 5.6, 4.3, 9.1, 1.5, 3.0};
  std::vector<double> out(input_vector.size(), 0.0);
  std::vector<double> sorted_vector = {0.2, 0.5, 1.0, 1.2, 1.5, 2.3, 3.0, 3.4, 4.3, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, EmptyTest) {
  std::vector<double> input_vector;
  std::vector<double> out;
  std::vector<double> sorted_vector;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, SingleElementTest) {
  std::vector<double> input_vector = {42.0};
  std::vector<double> out(1, 0.0);
  std::vector<double> sorted_vector = {42.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, NegativeAndPositiveTest) {
  std::vector<double> input_vector = {-3.2, 1.1, -7.5, 0.0, 4.4, -2.2, 3.3};
  std::vector<double> out(input_vector.size(), 0.0);
  std::vector<double> sorted_vector = {-7.5, -3.2, -2.2, 0.0, 1.1, 3.3, 4.4};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}