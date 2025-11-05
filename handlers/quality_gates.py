"""
Quality Gates for MCP Server - Phase 3.1 Implementation

Provides automated quality validation and enforcement for agent outputs,
ensuring consistent standards and iterative refinement until quality thresholds are met.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QualityRating(str, Enum):
    """Quality rating levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


class CriteriaType(str, Enum):
    """Types of quality criteria."""
    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"
    CLARITY = "clarity"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    TESTABILITY = "testability"


@dataclass
class QualityCriterion:
    """A single quality criterion with validation logic."""
    name: str
    type: CriteriaType
    description: str
    validator: Callable[[Any], float]  # Returns score 0.0-1.0
    weight: float = 1.0
    min_score: float = 0.7
    
    def evaluate(self, output: Any) -> float:
        """Evaluate this criterion against output."""
        try:
            score = self.validator(output)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Error evaluating criterion {self.name}: {e}")
            return 0.0


class QualityCriteria(BaseModel):
    """Collection of quality criteria for evaluation."""
    criteria: List[QualityCriterion] = Field(default_factory=list)
    min_overall_score: float = Field(default=0.7, ge=0.0, le=1.0)
    min_rating: QualityRating = Field(default=QualityRating.ACCEPTABLE)
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_criterion(self, criterion: QualityCriterion):
        """Add a quality criterion."""
        self.criteria.append(criterion)
    
    def get_rating_threshold(self, rating: QualityRating) -> float:
        """Get score threshold for a rating level."""
        thresholds = {
            QualityRating.POOR: 0.0,
            QualityRating.NEEDS_IMPROVEMENT: 0.5,
            QualityRating.ACCEPTABLE: 0.7,
            QualityRating.GOOD: 0.8,
            QualityRating.EXCELLENT: 0.9
        }
        return thresholds.get(rating, 0.7)


@dataclass
class CriterionResult:
    """Result of evaluating a single criterion."""
    criterion_name: str
    score: float
    passed: bool
    feedback: str


class QualityResult(BaseModel):
    """Result of quality evaluation."""
    overall_score: float = Field(ge=0.0, le=1.0)
    rating: QualityRating
    passed: bool
    criterion_results: List[Dict[str, Any]] = Field(default_factory=list)
    feedback: str
    timestamp: datetime = Field(default_factory=datetime.now)
    evaluation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": round(self.overall_score, 3),
            "rating": self.rating.value,
            "passed": self.passed,
            "criterion_results": self.criterion_results,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat(),
            "evaluation_time_ms": round(self.evaluation_time_ms, 2)
        }


class QualityGate:
    """
    Quality gate that evaluates outputs against criteria and enforces quality standards.
    
    Features:
    - Multiple quality criteria with weighted scoring
    - Iterative refinement until quality thresholds met
    - Detailed feedback for improvements
    - Performance tracking
    """
    
    def __init__(self, criteria: QualityCriteria, name: str = "default"):
        """
        Initialize quality gate.
        
        Args:
            criteria: Quality criteria to evaluate against
            name: Name of this quality gate
        """
        self.criteria = criteria
        self.name = name
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.pass_count = 0
        self.fail_count = 0
    
    def evaluate(self, output: Any, context: Optional[Dict[str, Any]] = None) -> QualityResult:
        """
        Evaluate output against quality criteria.
        
        Args:
            output: The output to evaluate
            context: Optional context for evaluation
            
        Returns:
            QualityResult with scores and feedback
        """
        start_time = time.time()
        self.evaluation_count += 1
        
        criterion_results = []
        total_weighted_score = 0.0
        total_weight = 0.0
        feedback_items = []
        
        # Evaluate each criterion
        for criterion in self.criteria.criteria:
            score = criterion.evaluate(output)
            passed = score >= criterion.min_score
            
            criterion_result = {
                "name": criterion.name,
                "type": criterion.type.value,
                "score": round(score, 3),
                "weight": criterion.weight,
                "passed": passed,
                "min_score": criterion.min_score
            }
            criterion_results.append(criterion_result)
            
            total_weighted_score += score * criterion.weight
            total_weight += criterion.weight
            
            if not passed:
                feedback_items.append(
                    f"- {criterion.name}: {score:.2f} (needs {criterion.min_score:.2f}) - {criterion.description}"
                )
        
        # Calculate overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine rating
        rating = self._get_rating(overall_score)
        
        # Check if passed
        passed = (
            overall_score >= self.criteria.min_overall_score and
            overall_score >= self.criteria.get_rating_threshold(self.criteria.min_rating)
        )
        
        # Generate feedback
        if passed:
            feedback = f"✅ Quality gate passed with {rating.value} rating (score: {overall_score:.2f})"
            self.pass_count += 1
        else:
            feedback = f"❌ Quality gate failed (score: {overall_score:.2f}, required: {self.criteria.min_overall_score:.2f})\n\n"
            feedback += "Improvements needed:\n" + "\n".join(feedback_items)
            self.fail_count += 1
        
        evaluation_time = (time.time() - start_time) * 1000
        self.total_evaluation_time += evaluation_time
        
        return QualityResult(
            overall_score=overall_score,
            rating=rating,
            passed=passed,
            criterion_results=criterion_results,
            feedback=feedback,
            evaluation_time_ms=evaluation_time
        )
    
    def enforce(
        self,
        output: Any,
        refiner: Callable[[Any, str], Any],
        max_retries: int = 3,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, QualityResult]:
        """
        Enforce quality gate with iterative refinement.
        
        Args:
            output: Initial output to evaluate
            refiner: Function to refine output given feedback
            max_retries: Maximum refinement attempts
            context: Optional context for evaluation and refinement
            
        Returns:
            Tuple of (refined_output, final_quality_result)
        """
        current_output = output
        history = []
        
        for attempt in range(max_retries + 1):
            result = self.evaluate(current_output, context)
            history.append({
                "attempt": attempt,
                "score": result.overall_score,
                "rating": result.rating.value,
                "passed": result.passed
            })
            
            if result.passed:
                logger.info(f"Quality gate '{self.name}' passed on attempt {attempt + 1}")
                return current_output, result
            
            if attempt < max_retries:
                logger.info(
                    f"Quality gate '{self.name}' failed attempt {attempt + 1}, "
                    f"refining (score: {result.overall_score:.2f})"
                )
                try:
                    current_output = refiner(current_output, result.feedback)
                except Exception as e:
                    logger.error(f"Error during refinement: {e}")
                    result.feedback += f"\n\n⚠️ Refinement error: {e}"
                    return current_output, result
        
        logger.warning(
            f"Quality gate '{self.name}' failed after {max_retries + 1} attempts "
            f"(final score: {result.overall_score:.2f})"
        )
        result.feedback += f"\n\n⚠️ Max refinement attempts ({max_retries}) reached. " \
                          f"Using best available output."
        return current_output, result
    
    def _get_rating(self, score: float) -> QualityRating:
        """Get quality rating for a score."""
        if score >= 0.9:
            return QualityRating.EXCELLENT
        elif score >= 0.8:
            return QualityRating.GOOD
        elif score >= 0.7:
            return QualityRating.ACCEPTABLE
        elif score >= 0.5:
            return QualityRating.NEEDS_IMPROVEMENT
        else:
            return QualityRating.POOR
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quality gate statistics."""
        avg_time = self.total_evaluation_time / self.evaluation_count if self.evaluation_count > 0 else 0.0
        pass_rate = self.pass_count / self.evaluation_count if self.evaluation_count > 0 else 0.0
        
        return {
            "name": self.name,
            "evaluation_count": self.evaluation_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "pass_rate": round(pass_rate, 3),
            "avg_evaluation_time_ms": round(avg_time, 2),
            "total_evaluation_time_ms": round(self.total_evaluation_time, 2)
        }


class QualityGateRegistry:
    """Registry for managing multiple quality gates."""
    
    def __init__(self):
        self.gates: Dict[str, QualityGate] = {}
    
    def register(self, gate: QualityGate) -> None:
        """Register a quality gate."""
        self.gates[gate.name] = gate
        logger.info(f"Registered quality gate: {gate.name}")
    
    def get(self, name: str) -> Optional[QualityGate]:
        """Get a quality gate by name."""
        return self.gates.get(name)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all quality gates."""
        return {
            name: gate.get_stats()
            for name, gate in self.gates.items()
        }


# Built-in quality criteria validators

def validate_completeness(output: Any) -> float:
    """Validate output completeness."""
    if output is None:
        return 0.0
    
    if isinstance(output, str):
        # Check string length and content
        if len(output) < 10:
            return 0.2
        if len(output) < 50:
            return 0.5
        # Check for placeholders or incomplete markers
        incomplete_markers = ['TODO', 'FIXME', '...', 'TBD', 'XXX']
        if any(marker.lower() in output.lower() for marker in incomplete_markers):
            return 0.6
        return 0.9
    
    if isinstance(output, dict):
        # Check if dict has required keys and non-empty values
        if not output:
            return 0.0
        empty_values = sum(1 for v in output.values() if v is None or v == "")
        completeness = 1.0 - (empty_values / len(output))
        return completeness
    
    return 0.8  # Default for other types


def validate_correctness(output: Any) -> float:
    """Validate output correctness (basic structural validation)."""
    if output is None:
        return 0.0
    
    # Basic type checking
    try:
        if isinstance(output, str):
            # Check for common error indicators
            error_indicators = ['error', 'exception', 'failed', 'invalid']
            if any(indicator in output.lower() for indicator in error_indicators):
                return 0.3
            return 0.8
        
        if isinstance(output, dict):
            # Check if it's valid JSON-serializable
            json.dumps(output)
            return 0.9
        
        return 0.7
    except Exception:
        return 0.3


def validate_clarity(output: Any) -> float:
    """Validate output clarity."""
    if output is None:
        return 0.0
    
    if isinstance(output, str):
        # Check readability factors
        if len(output) < 10:
            return 0.3
        
        # Check for structure (paragraphs, sentences)
        has_structure = '\n' in output or '. ' in output
        
        # Check for excessive length without breaks
        if len(output) > 1000 and not has_structure:
            return 0.5
        
        return 0.85 if has_structure else 0.7
    
    return 0.8


# Default quality gates

def create_default_quality_gate() -> QualityGate:
    """Create a default quality gate with standard criteria."""
    criteria = QualityCriteria(
        min_overall_score=0.7,
        min_rating=QualityRating.ACCEPTABLE
    )
    
    criteria.add_criterion(QualityCriterion(
        name="Completeness",
        type=CriteriaType.COMPLETENESS,
        description="Output is complete with all necessary information",
        validator=validate_completeness,
        weight=1.5,
        min_score=0.7
    ))
    
    criteria.add_criterion(QualityCriterion(
        name="Correctness",
        type=CriteriaType.CORRECTNESS,
        description="Output is structurally correct and error-free",
        validator=validate_correctness,
        weight=2.0,
        min_score=0.8
    ))
    
    criteria.add_criterion(QualityCriterion(
        name="Clarity",
        type=CriteriaType.CLARITY,
        description="Output is clear and well-structured",
        validator=validate_clarity,
        weight=1.0,
        min_score=0.7
    ))
    
    return QualityGate(criteria, name="default")


def create_strict_quality_gate() -> QualityGate:
    """Create a strict quality gate for production use."""
    criteria = QualityCriteria(
        min_overall_score=0.85,
        min_rating=QualityRating.GOOD
    )
    
    criteria.add_criterion(QualityCriterion(
        name="Completeness",
        type=CriteriaType.COMPLETENESS,
        description="Output is complete with all necessary information",
        validator=validate_completeness,
        weight=1.5,
        min_score=0.85
    ))
    
    criteria.add_criterion(QualityCriterion(
        name="Correctness",
        type=CriteriaType.CORRECTNESS,
        description="Output is structurally correct and error-free",
        validator=validate_correctness,
        weight=2.0,
        min_score=0.9
    ))
    
    criteria.add_criterion(QualityCriterion(
        name="Clarity",
        type=CriteriaType.CLARITY,
        description="Output is clear and well-structured",
        validator=validate_clarity,
        weight=1.0,
        min_score=0.8
    ))
    
    return QualityGate(criteria, name="strict")


# Global registry
_registry = QualityGateRegistry()

# Register default gates
_registry.register(create_default_quality_gate())
_registry.register(create_strict_quality_gate())


def get_registry() -> QualityGateRegistry:
    """Get the global quality gate registry."""
    return _registry
