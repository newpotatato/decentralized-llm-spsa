import unittest

from src.assignment_buffer import (
    CandidateProposal,
    CandidateStatus,
    GlobalAssignmentBuffer,
    TaskStatus,
)


def proposal(agent_id, score, processing=2.0):
    return CandidateProposal(
        agent_id=agent_id,
        base_score=score,
        predicted_processing=processing,
    )


class TestGlobalAssignmentBuffer(unittest.TestCase):
    def test_one_task_is_assigned_to_only_one_agent(self):
        buffer = GlobalAssignmentBuffer()
        task = buffer.submit(
            task_id=1,
            controller_id=4,
            arrival_time=0.0,
            deadline=10.0,
            time_budget=10.0,
            proposals=[proposal(0, 0.8), proposal(1, 0.7)],
        )

        assignments = buffer.dispatch(0.0, {0: 0.0, 1: 0.0})

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].agent_id, 0)
        self.assertEqual(task.status, TaskStatus.ASSIGNED)
        self.assertEqual(task.proposals[0].status, CandidateStatus.ASSIGNED)
        self.assertEqual(task.proposals[1].status, CandidateStatus.INVALIDATED)

    def test_one_free_agent_receives_only_one_task(self):
        buffer = GlobalAssignmentBuffer()
        for task_id, score in [(1, 0.6), (2, 0.9)]:
            buffer.submit(
                task_id=task_id,
                controller_id=task_id,
                arrival_time=0.0,
                deadline=10.0,
                time_budget=10.0,
                proposals=[proposal(0, score)],
            )

        assignments = buffer.dispatch(0.0, {0: 0.0})

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].task_id, 2)
        self.assertEqual(buffer.tasks[1].status, TaskStatus.PENDING)

    def test_busy_agent_is_not_dispatched(self):
        buffer = GlobalAssignmentBuffer()
        buffer.submit(
            task_id=1,
            controller_id=0,
            arrival_time=0.0,
            deadline=10.0,
            time_budget=10.0,
            proposals=[proposal(0, 0.99), proposal(1, 0.50)],
        )

        assignments = buffer.dispatch(0.0, {0: 5.0, 1: 0.0})

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].agent_id, 1)

    def test_hot_task_outranks_cold_task_for_same_agent(self):
        buffer = GlobalAssignmentBuffer()
        buffer.submit(
            task_id=1,
            controller_id=0,
            arrival_time=0.0,
            deadline=2.1,
            time_budget=2.1,
            proposals=[proposal(0, 0.70, processing=2.0)],
        )
        buffer.submit(
            task_id=2,
            controller_id=1,
            arrival_time=0.0,
            deadline=20.0,
            time_budget=20.0,
            proposals=[proposal(0, 0.70, processing=2.0)],
        )

        assignments = buffer.dispatch(0.0, {0: 0.0})

        self.assertEqual(assignments[0].task_id, 1)
        self.assertLess(
            buffer.tasks[1].proposals[0].slack_coefficient,
            buffer.tasks[2].proposals[0].slack_coefficient,
        )

    def test_base_score_is_immutable_during_refresh(self):
        buffer = GlobalAssignmentBuffer()
        task = buffer.submit(
            task_id=1,
            controller_id=0,
            arrival_time=0.0,
            deadline=10.0,
            time_budget=10.0,
            proposals=[proposal(0, 0.75)],
        )
        original = task.proposals[0].proposal.base_score

        buffer.refresh(0.0, {0: 4.0})
        first_priority = task.proposals[0].priority
        buffer.refresh(3.0, {0: 4.0})

        self.assertEqual(task.proposals[0].proposal.base_score, original)
        self.assertNotEqual(task.proposals[0].priority, first_priority)

    def test_failed_execution_is_terminal(self):
        buffer = GlobalAssignmentBuffer()
        buffer.submit(
            task_id=1,
            controller_id=7,
            arrival_time=0.0,
            deadline=10.0,
            time_budget=10.0,
            proposals=[proposal(0, 0.8)],
        )
        buffer.dispatch(0.0, {0: 0.0})

        buffer.complete(1, completed_at=2.0, success=False)

        self.assertEqual(buffer.tasks[1].status, TaskStatus.FAILED)
        self.assertEqual(list(buffer.pending_tasks()), [])


if __name__ == "__main__":
    unittest.main()
