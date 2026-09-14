import unittest

from src.environment.evaluator import evaluate_trajectory
from src.environment.tools import DeterministicToolEnvironment
from src.tasks.generator import generate_task_set


class EnvironmentTests(unittest.TestCase):
    def test_same_initial_state_and_arguments_return_same_result(self):
        task = generate_task_set()[0]
        first = DeterministicToolEnvironment(task.hidden_state)
        second = DeterministicToolEnvironment(task.hidden_state)
        call = task.expected_tool_calls[0]
        self.assertEqual(
            first.execute(call["name"], call["arguments"]).result,
            second.execute(call["name"], call["arguments"]).result,
        )

    def test_runs_are_isolated_and_mutation_is_idempotent(self):
        task = next(value for value in generate_task_set() if value.pair_id == "precise-012")
        env = DeterministicToolEnvironment(task.hidden_state)
        call = task.expected_tool_calls[0]
        first = env.execute(call["name"], call["arguments"])
        second = env.execute(call["name"], call["arguments"])
        self.assertEqual(first.result, second.result)
        fresh = DeterministicToolEnvironment(task.hidden_state)
        self.assertNotEqual(env.snapshot(), fresh.snapshot())

    def test_schema_errors_are_structured(self):
        env = DeterministicToolEnvironment({})
        result = env.execute("read_records", {"dataset_id": 123, "extra": True})
        self.assertEqual(result.error_code, "schema_error")
        self.assertEqual(result.result["error"]["code"], "schema_error")
        self.assertGreaterEqual(len(result.result["error"]["details"]), 2)

    def test_expected_trajectory_and_final_state_succeed(self):
        task = next(value for value in generate_task_set() if value.pair_id == "long-011")
        env = DeterministicToolEnvironment(task.hidden_state)
        for call in task.expected_tool_calls:
            env.execute(call["name"], call["arguments"])
        evaluation = evaluate_trajectory(
            env.executions,
            env.snapshot(),
            expected_tool_calls=task.expected_tool_calls,
            allowed_equivalent_paths=task.allowed_equivalent_paths,
            expected_final_state=task.expected_final_state,
            turn_count=2,
        )
        self.assertTrue(evaluation.success)

    def test_all_generated_gold_trajectories_succeed(self):
        for task in generate_task_set():
            env = DeterministicToolEnvironment(task.hidden_state)
            for call in task.expected_tool_calls:
                env.execute(call["name"], call["arguments"])
            evaluation = evaluate_trajectory(
                env.executions,
                env.snapshot(),
                expected_tool_calls=task.expected_tool_calls,
                allowed_equivalent_paths=task.allowed_equivalent_paths,
                expected_final_state=task.expected_final_state,
                turn_count=len(task.expected_tool_calls),
            )
            self.assertTrue(evaluation.success, task.pair_id)

    def test_wrong_tool_and_repetition_are_classified(self):
        task = generate_task_set()[0]
        env = DeterministicToolEnvironment(task.hidden_state)
        env.execute("read_records", {"dataset_id": "wrong"})
        wrong = evaluate_trajectory(
            env.executions,
            env.snapshot(),
            expected_tool_calls=task.expected_tool_calls,
            allowed_equivalent_paths=(),
            expected_final_state={},
            turn_count=1,
        )
        self.assertEqual(wrong.failure_type, "wrong_tool")

        repeat_env = DeterministicToolEnvironment(task.hidden_state)
        call = task.expected_tool_calls[0]
        for _ in range(3):
            repeat_env.execute(call["name"], call["arguments"])
        repeated = evaluate_trajectory(
            repeat_env.executions,
            repeat_env.snapshot(),
            expected_tool_calls=task.expected_tool_calls,
            allowed_equivalent_paths=(),
            expected_final_state={},
            turn_count=3,
        )
        self.assertEqual(repeated.failure_type, "repeated_call")


if __name__ == "__main__":
    unittest.main()
