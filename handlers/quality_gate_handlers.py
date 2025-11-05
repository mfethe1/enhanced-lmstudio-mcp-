"""
Quality Gates Handler Module

Provides MCP tool handlers for quality gate evaluation and enforcement.
"""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import quality gates module
try:
    import sys
    import os
    # Add parent directory to path if needed
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import from the handlers directory
    sys.path.insert(0, os.path.join(parent_dir, 'handlers'))
    from quality_gates import get_registry, QualityGate
    QUALITY_GATES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quality gates module not available: {e}")
    QUALITY_GATES_AVAILABLE = False
    get_registry = None


def handle_evaluate_quality(arguments: Dict[str, Any], server: Any) -> Dict[str, Any]:
    """
    Evaluate output against quality criteria.
    
    Args:
        arguments: {
            "output": str - The output to evaluate
            "gate_name": str - Quality gate to use (default: "default")
            "context": dict - Optional context for evaluation
        }
        server: MCP server instance
        
    Returns:
        Quality evaluation result with scores and feedback
    """
    if not QUALITY_GATES_AVAILABLE:
        return {
            "error": "Quality gates module not available",
            "passed": False
        }
    
    try:
        output = arguments.get("output", "")
        gate_name = arguments.get("gate_name", "default")
        context = arguments.get("context")
        
        if not output:
            return {
                "error": "Output is required",
                "passed": False
            }
        
        # Get quality gate from registry
        registry = get_registry()
        gate = registry.get(gate_name)
        
        if not gate:
            return {
                "error": f"Quality gate '{gate_name}' not found. Available gates: {list(registry.gates.keys())}",
                "passed": False
            }
        
        # Evaluate quality
        result = gate.evaluate(output, context)
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error evaluating quality: {e}", exc_info=True)
        return {
            "error": f"Quality evaluation failed: {str(e)}",
            "passed": False
        }


def handle_enforce_quality_gate(arguments: Dict[str, Any], server: Any) -> Dict[str, Any]:
    """
    Enforce quality gate with iterative refinement using LLM.
    
    Args:
        arguments: {
            "output": str - The output to enforce quality on
            "gate_name": str - Quality gate to use (default: "default")
            "max_retries": int - Maximum refinement attempts (default: 3)
            "refinement_instructions": str - Additional instructions for refinement
            "context": dict - Optional context for evaluation and refinement
        }
        server: MCP server instance
        
    Returns:
        Refined output and quality result
    """
    if not QUALITY_GATES_AVAILABLE:
        return {
            "error": "Quality gates module not available",
            "passed": False,
            "output": arguments.get("output", "")
        }
    
    try:
        output = arguments.get("output", "")
        gate_name = arguments.get("gate_name", "default")
        max_retries = arguments.get("max_retries", 3)
        refinement_instructions = arguments.get("refinement_instructions", "")
        context = arguments.get("context")
        
        if not output:
            return {
                "error": "Output is required",
                "passed": False,
                "output": ""
            }
        
        # Get quality gate from registry
        registry = get_registry()
        gate = registry.get(gate_name)
        
        if not gate:
            return {
                "error": f"Quality gate '{gate_name}' not found. Available gates: {list(registry.gates.keys())}",
                "passed": False,
                "output": output
            }
        
        # Define refiner function that uses LLM
        def refine_output(current_output: str, feedback: str) -> str:
            """Refine output based on feedback using LLM."""
            try:
                # Build refinement prompt
                prompt = f"""You are a quality improvement assistant. Your task is to refine the following output based on the feedback provided.

Original Output:
{current_output}

Quality Feedback:
{feedback}

{refinement_instructions if refinement_instructions else ''}

Please provide an improved version of the output that addresses all the feedback points. Output only the refined content, no explanations."""
                
                # Use server's LLM to refine
                from enhanced_mcp_tools import call_llm_with_routing
                
                refined = call_llm_with_routing(
                    server=server,
                    prompt=prompt,
                    complexity="medium",
                    role="refiner",
                    intent="improve_quality",
                    temperature=0.3
                )
                
                return refined.strip() if refined else current_output
                
            except Exception as e:
                logger.error(f"Error refining output: {e}")
                return current_output
        
        # Enforce quality gate with refinement
        refined_output, result = gate.enforce(
            output=output,
            refiner=refine_output,
            max_retries=max_retries,
            context=context
        )
        
        response = result.to_dict()
        response["output"] = refined_output
        response["original_output"] = output
        response["refinement_applied"] = refined_output != output
        
        return response
        
    except Exception as e:
        logger.error(f"Error enforcing quality gate: {e}", exc_info=True)
        return {
            "error": f"Quality enforcement failed: {str(e)}",
            "passed": False,
            "output": arguments.get("output", "")
        }


def handle_get_quality_stats(arguments: Dict[str, Any], server: Any) -> Dict[str, Any]:
    """
    Get statistics for all quality gates.
    
    Args:
        arguments: {}
        server: MCP server instance
        
    Returns:
        Statistics for all quality gates (pass rate, evaluation count, performance)
    """
    if not QUALITY_GATES_AVAILABLE:
        return {
            "error": "Quality gates module not available",
            "stats": {}
        }
    
    try:
        registry = get_registry()
        stats = registry.get_all_stats()
        
        # Format as markdown
        markdown = "# Quality Gates Statistics\n\n"
        
        if not stats:
            markdown += "No quality gates have been used yet.\n"
        else:
            markdown += f"**Total Gates**: {len(stats)}\n\n"
            
            for gate_name, gate_stats in stats.items():
                markdown += f"## {gate_name.capitalize()} Gate\n\n"
                markdown += f"- **Evaluations**: {gate_stats['evaluation_count']}\n"
                markdown += f"- **Pass Count**: {gate_stats['pass_count']}\n"
                markdown += f"- **Fail Count**: {gate_stats['fail_count']}\n"
                markdown += f"- **Pass Rate**: {gate_stats['pass_rate'] * 100:.1f}%\n"
                markdown += f"- **Avg Evaluation Time**: {gate_stats['avg_evaluation_time_ms']:.2f}ms\n"
                markdown += f"- **Total Evaluation Time**: {gate_stats['total_evaluation_time_ms']:.2f}ms\n\n"
        
        return {
            "stats": stats,
            "summary": markdown
        }
        
    except Exception as e:
        logger.error(f"Error getting quality stats: {e}", exc_info=True)
        return {
            "error": f"Failed to get quality stats: {str(e)}",
            "stats": {}
        }
