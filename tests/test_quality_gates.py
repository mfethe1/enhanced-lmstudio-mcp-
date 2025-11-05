"""
Tests for Quality Gates - Phase 3.1

Comprehensive test suite for quality gate evaluation and enforcement.
"""

import pytest
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'handlers'))

from quality_gates import (
    QualityGate,
    QualityCriteria,
    QualityCriterion,
    QualityRating,
    CriteriaType,
    QualityGateRegistry,
    validate_completeness,
    validate_correctness,
    validate_clarity,
    create_default_quality_gate,
    create_strict_quality_gate,
    get_registry
)


class TestQualityValidators:
    """Test built-in quality validators."""
    
    def test_validate_completeness_empty(self):
        """Test completeness validation with empty input."""
        score = validate_completeness(None)
        assert score == 0.0
        
        score = validate_completeness("")
        assert score == 0.2
    
    def test_validate_completeness_short(self):
        """Test completeness validation with short input."""
        score = validate_completeness("Short")
        assert score == 0.2  # Less than 10 chars
    
    def test_validate_completeness_complete(self):
        """Test completeness validation with complete input."""
        score = validate_completeness("This is a complete and detailed response with sufficient information.")
        assert score >= 0.9
    
    def test_validate_completeness_with_todo(self):
        """Test completeness validation with TODO markers."""
        score = validate_completeness("This is incomplete. TODO: Add more details.")
        assert score < 0.9
    
    def test_validate_correctness_none(self):
        """Test correctness validation with None."""
        score = validate_correctness(None)
        assert score == 0.0
    
    def test_validate_correctness_error(self):
        """Test correctness validation with error markers."""
        score = validate_correctness("This is an error message")
        assert score < 0.5
    
    def test_validate_correctness_valid_dict(self):
        """Test correctness validation with valid dict."""
        score = validate_correctness({"key": "value", "status": "success"})
        assert score >= 0.9
    
    def test_validate_clarity_empty(self):
        """Test clarity validation with empty input."""
        score = validate_clarity(None)
        assert score == 0.0
        
        score = validate_clarity("Short")
        assert score < 0.7
    
    def test_validate_clarity_structured(self):
        """Test clarity validation with structured input."""
        score = validate_clarity("First point. Second point. Third point.")
        assert score >= 0.85


class TestQualityCriteria:
    """Test quality criteria configuration."""
    
    def test_create_criteria(self):
        """Test creating quality criteria."""
        criteria = QualityCriteria(
            min_overall_score=0.8,
            min_rating=QualityRating.GOOD
        )
        assert criteria.min_overall_score == 0.8
        assert criteria.min_rating == QualityRating.GOOD
    
    def test_add_criterion(self):
        """Test adding a criterion."""
        criteria = QualityCriteria()
        
        criterion = QualityCriterion(
            name="Test",
            type=CriteriaType.COMPLETENESS,
            description="Test criterion",
            validator=validate_completeness,
            weight=1.0,
            min_score=0.7
        )
        
        criteria.add_criterion(criterion)
        assert len(criteria.criteria) == 1
        assert criteria.criteria[0].name == "Test"
    
    def test_get_rating_threshold(self):
        """Test getting rating thresholds."""
        criteria = QualityCriteria()
        
        assert criteria.get_rating_threshold(QualityRating.POOR) == 0.0
        assert criteria.get_rating_threshold(QualityRating.ACCEPTABLE) == 0.7
        assert criteria.get_rating_threshold(QualityRating.GOOD) == 0.8
        assert criteria.get_rating_threshold(QualityRating.EXCELLENT) == 0.9


class TestQualityGate:
    """Test quality gate evaluation."""
    
    def test_create_quality_gate(self):
        """Test creating a quality gate."""
        criteria = QualityCriteria()
        gate = QualityGate(criteria, name="test")
        
        assert gate.name == "test"
        assert gate.evaluation_count == 0
    
    def test_evaluate_passing_output(self):
        """Test evaluating output that passes."""
        gate = create_default_quality_gate()
        
        output = "This is a complete, correct, and clear response that provides sufficient detail and structure."
        result = gate.evaluate(output)
        
        assert result.passed is True
        assert result.overall_score >= 0.7
        assert result.rating in [QualityRating.ACCEPTABLE, QualityRating.GOOD, QualityRating.EXCELLENT]
        assert gate.evaluation_count == 1
        assert gate.pass_count == 1
    
    def test_evaluate_failing_output(self):
        """Test evaluating output that fails."""
        gate = create_default_quality_gate()
        
        output = ""  # Empty output
        result = gate.evaluate(output)
        
        assert result.passed is False
        assert result.overall_score < 0.7
        assert result.rating in [QualityRating.POOR, QualityRating.NEEDS_IMPROVEMENT]
        assert gate.fail_count == 1
    
    def test_evaluate_partial_failure(self):
        """Test evaluating output with partial failures."""
        gate = create_strict_quality_gate()
        
        output = "Short"  # Too short, low completeness
        result = gate.evaluate(output)
        
        assert result.passed is False
        assert len(result.criterion_results) > 0
        
        # Check that completeness failed
        completeness_result = next((r for r in result.criterion_results if r['name'] == 'Completeness'), None)
        assert completeness_result is not None
        assert completeness_result['passed'] is False
    
    def test_enforce_immediate_pass(self):
        """Test enforcement when output immediately passes."""
        gate = create_default_quality_gate()
        
        output = "This is a complete and correct response with good structure."
        
        def dummy_refiner(current, feedback):
            return current + " (refined)"
        
        refined_output, result = gate.enforce(output, dummy_refiner, max_retries=3)
        
        assert result.passed is True
        assert refined_output == output  # Should not be refined
    
    def test_enforce_with_refinement(self):
        """Test enforcement with refinement attempts."""
        gate = create_strict_quality_gate()
        
        output = "Short"  # Will fail initially
        refinement_count = [0]
        
        def mock_refiner(current, feedback):
            refinement_count[0] += 1
            if refinement_count[0] == 1:
                return "This is a better response."
            return "This is a complete, correct, and clear response that provides sufficient detail and structure."
        
        refined_output, result = gate.enforce(output, mock_refiner, max_retries=3)
        
        assert refined_output != output
        assert refinement_count[0] > 0
    
    def test_get_stats(self):
        """Test getting gate statistics."""
        gate = create_default_quality_gate()
        gate.name = "test_stats"
        
        # Run some evaluations with passing outputs
        gate.evaluate("This is a complete and detailed response with sufficient information and structure.")
        gate.evaluate("Another complete response with good clarity and correctness markers.")
        gate.evaluate("")
        
        stats = gate.get_stats()
        
        assert stats['name'] == "test_stats"
        assert stats['evaluation_count'] == 3
        assert stats['pass_count'] > 0  # At least the first two should pass
        assert stats['pass_rate'] > 0


class TestQualityGateRegistry:
    """Test quality gate registry."""
    
    def test_create_registry(self):
        """Test creating a registry."""
        registry = QualityGateRegistry()
        assert len(registry.gates) == 0
    
    def test_register_gate(self):
        """Test registering a gate."""
        registry = QualityGateRegistry()
        gate = create_default_quality_gate()
        gate.name = "test_gate"
        
        registry.register(gate)
        assert "test_gate" in registry.gates
        assert registry.get("test_gate") == gate
    
    def test_get_all_stats(self):
        """Test getting all gate statistics."""
        registry = QualityGateRegistry()
        
        gate1 = create_default_quality_gate()
        gate1.name = "gate1"
        gate1.evaluate("Test output")
        
        gate2 = create_strict_quality_gate()
        gate2.name = "gate2"
        gate2.evaluate("Test output")
        
        registry.register(gate1)
        registry.register(gate2)
        
        all_stats = registry.get_all_stats()
        
        assert "gate1" in all_stats
        assert "gate2" in all_stats
        assert all_stats["gate1"]["evaluation_count"] == 1
        assert all_stats["gate2"]["evaluation_count"] == 1


class TestDefaultGates:
    """Test pre-configured quality gates."""
    
    def test_default_gate_exists(self):
        """Test that default gate is registered."""
        registry = get_registry()
        gate = registry.get("default")
        
        assert gate is not None
        assert gate.name == "default"
    
    def test_strict_gate_exists(self):
        """Test that strict gate is registered."""
        registry = get_registry()
        gate = registry.get("strict")
        
        assert gate is not None
        assert gate.name == "strict"
    
    def test_default_gate_criteria(self):
        """Test default gate has expected criteria."""
        gate = create_default_quality_gate()
        
        assert len(gate.criteria.criteria) == 3
        assert gate.criteria.min_overall_score == 0.7
        assert gate.criteria.min_rating == QualityRating.ACCEPTABLE
    
    def test_strict_gate_criteria(self):
        """Test strict gate has stricter criteria."""
        gate = create_strict_quality_gate()
        
        assert len(gate.criteria.criteria) == 3
        assert gate.criteria.min_overall_score == 0.85
        assert gate.criteria.min_rating == QualityRating.GOOD


class TestPerformance:
    """Test quality gate performance."""
    
    def test_evaluation_performance(self):
        """Test that evaluation completes quickly."""
        gate = create_default_quality_gate()
        
        output = "This is a test output for performance measurement."
        result = gate.evaluate(output)
        
        # Should complete in less than 10ms
        assert result.evaluation_time_ms < 10
    
    def test_multiple_evaluations(self):
        """Test performance with multiple evaluations."""
        gate = create_default_quality_gate()
        
        outputs = [
            "First test output",
            "Second test output",
            "Third test output",
            "Fourth test output",
            "Fifth test output"
        ]
        
        for output in outputs:
            result = gate.evaluate(output)
            assert result.evaluation_time_ms < 10
        
        stats = gate.get_stats()
        assert stats['avg_evaluation_time_ms'] < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
