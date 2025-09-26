"""
MCTS Planning System for Exploring Solution Spaces and Finding Optimal Code Generation Strategies.

Integrates with server.make_llm_request_with_retry and storage patterns.
Safe-by-default: synchronous handler wrappers can call async plan_solution via run_until_complete.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class MCTSNode:
    """Node in the MCTS tree representing a partial solution state"""
    state: Dict[str, Any]
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: List[str] = field(default_factory=list)
    action: Optional[str] = None

    @property
    def uct_value(self) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.total_reward / max(1, self.visits)
        exploration = math.sqrt(2 * math.log(max(1, self.parent.visits if self.parent else 1)) / max(1, self.visits))
        return exploitation + exploration

    def best_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.uct_value)

    def add_child(self, action: str, state: Dict[str, Any]) -> 'MCTSNode':
        child = MCTSNode(
            state=state,
            parent=self,
            untried_actions=self._get_possible_actions(state),
            action=action,
        )
        self.children.append(child)
        try:
            self.untried_actions.remove(action)
        except ValueError:
            pass
        return child

    def _get_possible_actions(self, state: Dict[str, Any]) -> List[str]:
        actions: List[str] = []
        if state.get("needs_imports"):
            actions.append("add_imports")
        if state.get("needs_functions"):
            actions.extend(["add_function", "add_async_function"])  # async variant placeholder
        if state.get("needs_classes"):
            actions.append("add_class")
        if state.get("has_code"):
            actions.extend(["refactor", "optimize", "add_tests"])
        return actions


class MCTSCodePlanner:
    """Uses MCTS to explore different code generation strategies and find optimal solutions."""

    def __init__(self, server: Any, max_iterations: int = 1000) -> None:
        self.server = server
        self.max_iterations = max_iterations
        self.simulation_depth = 10

    async def plan_solution(self, task_description: str, constraints: Dict[str, Any], initial_code: str = "") -> Dict[str, Any]:
        initial_state: Dict[str, Any] = {
            "task": task_description,
            "constraints": constraints,
            "code": initial_code or "",
            "needs_imports": True,
            "needs_functions": True,
            "needs_classes": "class" in (task_description or "").lower(),
            "has_code": bool(initial_code),
            "quality_score": 0.0,
        }
        root = MCTSNode(state=initial_state, untried_actions=self._get_initial_actions(initial_state))

        iterations_used = 0
        for i in range(self.max_iterations):
            node = await self._tree_policy(root)
            reward = await self._default_policy(node)
            self._backup(node, reward)
            iterations_used = i + 1
            # Early stopping if high-quality child emerges
            if root.children:
                best = root.best_child()
                exp_q = best.total_reward / max(1, best.visits)
                if exp_q > 0.95:
                    break

        best_path = self._get_best_path(root)
        alternatives = self._get_alternative_paths(root, n=3)
        expected_quality = 0.0
        if root.children:
            bc = root.best_child()
            expected_quality = bc.total_reward / max(1, bc.visits)
        return {
            "action_sequence": best_path,
            "expected_quality": expected_quality,
            "alternative_paths": alternatives,
            "tree_size": self._count_nodes(root),
            "iterations_used": iterations_used,
        }

    def _get_initial_actions(self, state: Dict[str, Any]) -> List[str]:
        actions: List[str] = []
        task_lower = (state.get("task") or "").lower()
        if any(k in task_lower for k in ("implement", "create")):
            actions.extend(["scaffold_structure", "define_interfaces"])
        if "refactor" in task_lower:
            actions.extend(["analyze_current", "identify_patterns"])
        if "optimize" in task_lower:
            actions.extend(["profile_code", "identify_bottlenecks"])
        if "test" in task_lower:
            actions.extend(["generate_test_cases", "setup_fixtures"])
        if "debug" in task_lower:
            actions.extend(["add_logging", "trace_execution"])
        actions.extend(["analyze_requirements", "research_similar"])  # always available
        return actions

    async def _tree_policy(self, node: MCTSNode) -> MCTSNode:
        current = node
        while True:
            if current.untried_actions:
                action = random.choice(current.untried_actions)
                new_state = await self._apply_action(current.state, action)
                return current.add_child(action, new_state)
            if current.children:
                current = current.best_child()
                continue
            return current

    async def _apply_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        new_state = dict(state)
        if action == "add_imports":
            new_state["code"] = (new_state.get("code") or "") + "\n# Imports added"
            new_state["needs_imports"] = False
            new_state["quality_score"] = float(new_state.get("quality_score", 0.0)) + 0.1
        elif action == "add_function":
            prompt = f"Generate a clean, minimal function for: {state.get('task')}\nCurrent code (truncated): {state.get('code','')[:300]}"
            try:
                gen = await self.server.make_llm_request_with_retry(prompt, temperature=0.3)
            except Exception as e:
                gen = f"# generation error: {e}"
            new_state["code"] = (new_state.get("code") or "") + f"\n{(gen or '')[:200]}"
            new_state["needs_functions"] = False
            new_state["quality_score"] = float(new_state.get("quality_score", 0.0)) + 0.3
        elif action == "add_tests":
            new_state["code"] = (new_state.get("code") or "") + "\n# Tests added"
            new_state["has_tests"] = True
            new_state["quality_score"] = float(new_state.get("quality_score", 0.0)) + 0.2
        elif action == "optimize":
            new_state["code"] = (new_state.get("code") or "") + "\n# Optimizations applied"
            new_state["is_optimized"] = True
            new_state["quality_score"] = float(new_state.get("quality_score", 0.0)) + 0.15
        # State updates
        new_state["has_code"] = True
        return new_state

    async def _default_policy(self, node: MCTSNode) -> float:
        state = dict(node.state)
        depth = 0
        while depth < self.simulation_depth:
            poss = self._get_simulation_actions(state)
            if not poss:
                break
            action = random.choice(poss)
            state = await self._apply_action(state, action)
            depth += 1
        return await self._evaluate_state(state)

    def _get_simulation_actions(self, state: Dict[str, Any]) -> List[str]:
        # Merge possible actions and initial actions to diversify simulations
        actions = set(MCTSNode(state). _get_possible_actions(state))  # type: ignore
        for a in self._get_initial_actions(state):
            actions.add(a)
        return list(actions)

    async def _evaluate_state(self, state: Dict[str, Any]) -> float:
        score = float(state.get("quality_score", 0.0))
        if not state.get("needs_imports") and not state.get("needs_functions"):
            score += 0.2
        if state.get("has_tests"):
            score += 0.1
        if state.get("is_optimized"):
            score += 0.1
        if state.get("needs_functions"):
            score -= 0.2
        code_snippet = (state.get("code") or "")
        if code_snippet:
            prompt = f"Rate this code quality (0-1). Respond with a number only.\n{code_snippet[:500]}"
            try:
                llm_score = await self.server.make_llm_request_with_retry(prompt, temperature=0.1)
                try:
                    num = float(str(llm_score).strip().split()[0].replace('%',''))
                    if num > 1.0:
                        num = num / 100.0
                    score += max(0.0, min(1.0, num)) * 0.3
                except Exception:
                    pass
            except Exception:
                pass
        return max(0.0, min(1.0, score))

    def _backup(self, node: MCTSNode, reward: float) -> None:
        cur: Optional[MCTSNode] = node
        while cur is not None:
            cur.visits += 1
            cur.total_reward += reward
            cur = cur.parent

    def _get_best_path(self, root: MCTSNode) -> List[str]:
        path: List[str] = []
        cur = root
        while cur.children:
            bc = cur.best_child()
            if bc.action:
                path.append(bc.action)
            cur = bc
        return path

    def _get_alternative_paths(self, root: MCTSNode, n: int = 3) -> List[List[str]]:
        paths: List[List[str]] = []
        tmp: List[tuple[List[str], float]] = []
        def dfs(node: MCTSNode, acc: List[str]) -> None:
            if not node.children:
                if acc:
                    quality = node.total_reward / max(1, node.visits)
                    tmp.append((acc.copy(), quality))
                return
            for ch in node.children:
                if ch.action:
                    acc.append(ch.action)
                dfs(ch, acc)
                if ch.action and acc:
                    acc.pop()
        dfs(root, [])
        tmp.sort(key=lambda x: -x[1])
        for path, _ in tmp[1:n+1]:
            paths.append(path)
        return paths

    def _count_nodes(self, root: MCTSNode) -> int:
        count = 1
        for ch in root.children:
            count += self._count_nodes(ch)
        return count

