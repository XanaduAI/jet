#pragma once

#include <complex>
#include <future>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <thread>

#include <taskflow/taskflow.hpp>

#include "PathInfo.hpp"
#include "TensorNetwork.hpp"

namespace Jet {

/**
 * @brief TaskBasedCpuContractor is a tensor network contractor that contracts
 *        tensors concurrently on the CPU using a task-based scheduler.
 *
 * @tparam Tensor Type of the tensors to be contracted. The only requirement
 *                for this type is that the following member functions exist:
 *                \code{.cpp}
 *     static Tensor AddTensors(const Tensor&, const Tensor&);
 *     static Tensor ContractTensors(const Tensor&, const Tensor&);
 *                \endcode
 */
template <typename Tensor> class TaskBasedCpuContractor {
  public:
    /// Type of the name-to-task map.
    using NameToTaskMap = std::unordered_map<std::string, tf::Task>;

    /// Type of the name-to-tensor map.
    using NameToTensorMap =
        std::unordered_map<std::string, std::unique_ptr<Tensor>>;

    /// Type of the name-to-parents map.
    using NameToParentsMap =
        std::unordered_map<std::string, std::unordered_set<std::string>>;

    /// Type of the task dependency graph.
    using TaskFlow = tf::Taskflow;

    /**
     * @brief Constructs a new `%TaskBasedCpuContractor` object.
     */
    TaskBasedCpuContractor()
        : executor_{}, memory_(0), flops_(0), reduced_(false)
    {
    }

    /**
     * @brief Returns the name-to-task map of this `%TaskBasedCpuContractor`.
     *
     * @return Map which associates names to tasks.
     */
    const NameToTaskMap &GetNameToTaskMap() const noexcept
    {
        return name_to_task_map_;
    }

    /**
     * @brief Returns the name-to-tensor map of this `%TaskBasedCpuContractor`.
     *
     * @return Map which associates names to tensors.
     */
    const NameToTensorMap &GetNameToTensorMap() const noexcept
    {
        return name_to_tensor_map_;
    }

    /**
     * @brief Returns the name-to-parents map of this `%TaskBasedCpuContractor`.
     *
     * @return Map which associates names to a vector of parent node IDs.
     */
    const NameToParentsMap &GetNameToParentsMap() const noexcept
    {
        return name_to_parents_map_;
    }

    /**
     * @brief Returns the result of the final tensor contraction after each call
     *        to AddContractionTasks().
     *
     * @note These results are only accessible after the future returned by
     *        Contract() becomes available.
     *
     * @see AddContractionTasks()
     * @see Contract()
     *
     * @return Vector of tensors.
     */
    const std::vector<Tensor> &GetResults() const noexcept { return results_; }

    /**
     * @brief Returns the reduction of the final tensor results.
     *
     * @note This result is only accessible after the future returned by
     *       Contract() becomes available.
     *
     * @see AddReductionTask()
     * @see Contract()
     *
     * @return Tensor at the end of the reduction task.
     */
    const Tensor &GetReductionResult() const noexcept
    {
        return reduction_result_;
    }

    /**
     * @brief Returns the taskflow of this `%TaskBasedCpuContractor`.
     *
     * @return Taskflow instance representing the task dependency graph.
     */
    const TaskFlow &GetTaskflow() const noexcept { return taskflow_; }

    /**
     * @brief Returns the number of floating-point operations needed to perform
     *        all the contraction tasks (assuming the tensor elements are
     *        floating-point numbers).
     *
     * @return Number of floating-point additions and multiplications.
     */
    double GetFlops() const noexcept { return flops_; }

    /**
     * @brief Returns the memory required (up to a constant sizeof() factor) to
     *        simultaneously hold all the intermediate and final results of the
     *        contraction tasks.
     *
     * @return Number of elements in the non-leaf tensors.
     */
    double GetMemory() const noexcept { return memory_; }

    /**
     * @brief Adds contraction tasks for a tensor network.
     *
     * @param tn Tensor network to be contracted.
     * @param path_info Contraction path through the tensor network.
     * @return Number of contraction tasks that are shared with previous calls
     *         to this function.
     */
    size_t AddContractionTasks(const TensorNetwork<Tensor> &tn,
                               const PathInfo &path_info) noexcept
    {
        const auto &path = path_info.GetPath();
        const auto &steps = path_info.GetSteps();

        if (path.empty()) {
            return 0;
        }

        const auto &nodes = tn.GetNodes();
        const size_t num_leaves = nodes.size();

        const size_t result_id = results_.size();
        results_.resize(results_.size() + 1);

        size_t shared_tasks = 0;

        for (size_t i = 0; i < path.size(); i++) {
            const auto [step_1_id, step_2_id] = path[i];

            const auto &step_1 = steps[step_1_id];
            const auto &step_2 = steps[step_2_id];
            const auto &step_3 = steps[num_leaves + i];

            const auto name_1 = DeriveTaskName_(step_1);
            const auto name_2 = DeriveTaskName_(step_2);
            auto name_3 = DeriveTaskName_(step_3);

            // Append the result ID to the final contraction task.
            const bool last_step = i == path.size() - 1;
            if (last_step) {
                name_3 += ":results[";
                name_3 += std::to_string(result_id);
                name_3 += ']';
            }

            // The name-to-parents map is used in AddDeletionTasks().
            name_to_parents_map_[name_1].emplace(name_3);
            name_to_parents_map_[name_2].emplace(name_3);

            // Ensure all the tensors have a place in the name-to-tensor map.
            if (step_1_id < num_leaves) {
                const auto &tensor = nodes[step_1_id].tensor;
                name_to_tensor_map_.try_emplace(
                    name_1, std::make_unique<Tensor>(tensor));
            }

            if (step_2_id < num_leaves) {
                const auto &tensor = nodes[step_2_id].tensor;
                name_to_tensor_map_.try_emplace(
                    name_2, std::make_unique<Tensor>(tensor));
            }

            name_to_tensor_map_.try_emplace(name_3, nullptr);

            // Do nothing if this contraction is already tracked.
            if (name_to_task_map_.count(name_3)) {
                shared_tasks++;
                continue;
            }

            flops_ += path_info.GetPathStepFlops(step_3.id);
            memory_ += path_info.GetPathStepMemory(step_3.id);

            AddContractionTask_(name_1, name_2, name_3);

            // Make sure the child tensors exist before the contraction happens.
            if (step_1_id >= num_leaves) {
                auto &task_1 = name_to_task_map_.at(name_1);
                auto &task_3 = name_to_task_map_.at(name_3);
                task_1.precede(task_3);
            }

            if (step_2_id >= num_leaves) {
                auto &task_2 = name_to_task_map_.at(name_2);
                auto &task_3 = name_to_task_map_.at(name_3);
                task_2.precede(task_3);
            }

            // Store the final tensor in the `results_` map.
            if (last_step) {
                AddStorageTask_(name_3, result_id);
            }
        }
        return shared_tasks;
    }

    /**
     * @brief Adds a reduction task to sum the result tensors.
     *
     * @warning Only one reduction task should be added per
     *          `%TaskBasedCpuContractor` instance.
     *
     * @return Number of created reduction tasks.
     */
    size_t AddReductionTask() noexcept
    {
        // Scheduling multiple reduction tasks introduces a race condition.
        if (reduced_) {
            return 0;
        }
        reduced_ = true;

        auto reduce_task = taskflow_
                               .reduce(results_.begin(), results_.end(),
                                       reduction_result_, Tensor::AddTensors)
                               .name("reduce");

        for (auto &result_task : result_tasks_) {
            result_task.precede(reduce_task);
        }

        return 1;
    }

    /**
     * @brief Adds deletion tasks for intermediate tensors.
     *
     * Each tensor that participates in a contraction will be paired with a
     * deletion task which deallocates the tensor once it is no longer needed.
     *
     * @return Number of created deletion tasks.
     */
    size_t AddDeletionTasks() noexcept
    {
        size_t delete_tasks = 0;
        for (const auto &[name, parents] : name_to_parents_map_) {
            if (parents.empty()) {
                continue;
            }

            const auto runner = [this, name = name]() {
                name_to_tensor_map_[name] = nullptr;
            };

            const std::string delete_task_name = name + ":delete";
            auto delete_task = taskflow_.emplace(runner).name(delete_task_name);
            ++delete_tasks;

            for (const auto &parent : parents) {
                const auto it = name_to_task_map_.find(parent);
                if (it != name_to_task_map_.end()) {
                    auto &parent_task = it->second;
                    parent_task.precede(delete_task);
                }
            }
        }
        return delete_tasks;
    }

    /**
     * @brief Executes the tasks in this `%TaskBasedCpuContractor`.
     *
     * @return Future that becomes available once all the tasks have finished.
     */
    std::future<void> Contract() { return executor_.run(taskflow_); }

  private:
    /// Taskflow executor to run tasks. Default-initialized to maximum number of
    /// system threads.
    tf::Executor executor_;

    /// Task graph to be executed during a contraction.
    TaskFlow taskflow_;

    /// Map that associates a task name with its corresponding task.
    NameToTaskMap name_to_task_map_;

    /// Map that associates a task name with its result tensor.
    NameToTensorMap name_to_tensor_map_;

    /// Map that associates a task name with a list of parent task names.
    /// Task `A` is a parent of task `B` if `A` immediately succeeds `B`.
    NameToParentsMap name_to_parents_map_;

    /// Tasks that store the results of a contraction in `results_`.
    std::vector<tf::Task> result_tasks_;

    /// Result of each call to AddContractionTasks().
    std::vector<Tensor> results_;

    /// Sum over the `results_` elements.
    Tensor reduction_result_;

    /// Memory required to store the intermediate tensors of a contraction.
    double memory_;

    /// Number of floating-point operations required to compute the intermediate
    /// tensors of a contraction.
    double flops_;

    /// Takes note of whether the reduction task has been added.
    bool reduced_;

    /**
     * @brief Derives the name of a task from a path step.
     *
     * @param step Path step to be used to derive the task name.
     * @return Name of the task.
     */
    std::string DeriveTaskName_(const PathStepInfo &step) const noexcept
    {
        return std::to_string(step.id) + ":" + step.name;
    }

    /**
     * @brief Adds a task which contracts two tensors to form a third tensor.
     *
     * The new task is added to `name_to_task_map_` and a pointer to the
     * resulting tensor is placed in the `name_to_tensor_map_`.
     *
     * @param name_1 Name of the first child tensor.
     * @param name_2 Name of the second child tensor.
     * @param name_3 Name of the resulting tensor.
     */
    void AddContractionTask_(const std::string &name_1,
                             const std::string &name_2,
                             const std::string &name_3) noexcept
    {
        const auto runner = [this, name_1, name_2, name_3]() {
            name_to_tensor_map_[name_3] = std::make_unique<Tensor>(
                Tensor::ContractTensors(*name_to_tensor_map_.at(name_1),
                                        *name_to_tensor_map_.at(name_2)));
        };

        const auto task_3 = taskflow_.emplace(runner).name(name_3);
        name_to_task_map_.emplace(name_3, task_3);
    }

    /**
     * @brief Stores the result of a tensor task in the `results_` member.
     *
     * The new task is added to `result_tasks_`.
     *
     * @note The tensor task is expected to place a scalar value at its key in
     *       the `name_to_tensor_map_`.
     *
     * @param name Name of the tensor to be stored.
     * @param result_id Index of `results_` where the tensor should be stored.
     */
    inline void AddStorageTask_(const std::string &name,
                                size_t result_id) noexcept
    {
        const auto runner = [this, result_id, name]() {
            auto &tensor = *name_to_tensor_map_.at(name);
            results_[result_id] = tensor;
        };

        std::string storage_task_name = name;
        storage_task_name += ":storage[";
        storage_task_name += std::to_string(result_id);
        storage_task_name += ']';

        auto storage_task = taskflow_.emplace(runner).name(storage_task_name);

        auto &preceeding_task = name_to_task_map_.at(name);
        preceeding_task.precede(storage_task);

        result_tasks_.emplace_back(storage_task);
    }
};

/**
 * @brief Streams a `TaskBasedCpuContractor` to an output stream.
 *
 * Currently, this function just dumps the task dependency graph of the given
 * `%TaskBasedCpuContractor` instance in a DOT format to the specified output
 * stream.
 *
 * @see See <a
 * href="https://taskflow.github.io/taskflow/classtf_1_1Taskflow.html#ac433018262e44b12c4cc9f0c4748d758">
 *      Taskflow::dump()</a>.
 *
 * @tparam Tensor Type of the tensors to be contracted.
 * @param out Output stream to be modified.
 * @param tbcc Task-based CPU contractor with the taskflow to be inserted.
 * @return Reference to the given output stream.
 */
template <class Tensor>
inline std::ostream &operator<<(std::ostream &out,
                                const TaskBasedCpuContractor<Tensor> &tbcc)
{
    const auto &taskflow = tbcc.GetTaskflow();
    taskflow.dump(out);
    return out;
}

}; // namespace Jet
