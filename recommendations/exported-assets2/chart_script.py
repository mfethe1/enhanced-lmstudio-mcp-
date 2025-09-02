import pandas as pd
import plotly.express as px

# Data from the provided JSON
data = [{"function": "handle_deep_research", "length": 193}, 
        {"function": "handle_tool_call", "length": 107}, 
        {"function": "get_all_tools", "length": 102}, 
        {"function": "handle_agent_team_plan_and_code", "length": 102}, 
        {"function": "handle_sequential_thinking", "length": 93}, 
        {"function": "handle_agent_team_refactor", "length": 68}, 
        {"function": "handle_llm_analysis_tool", "length": 58}, 
        {"function": "handle_debug_analysis", "length": 54}, 
        {"function": "handle_execution_trace", "length": 53}, 
        {"function": "_apply_proposed_changes", "length": 51}, 
        {"function": "handle_code_execution", "length": 50}, 
        {"function": "_build_sequential_thinking_prompt", "length": 48}, 
        {"function": "handle_health_check", "length": 48}, 
        {"function": "handle_message", "length": 47}, 
        {"function": "monitor_performance", "length": 39}]

df = pd.DataFrame(data)

# Truncate function names to meet 15 character limit while keeping recognizable
function_truncated = {
    "handle_deep_research": "handle_deep_res",
    "handle_tool_call": "handle_tool_ca",
    "get_all_tools": "get_all_tools",
    "handle_agent_team_plan_and_code": "agent_team_pln",
    "handle_sequential_thinking": "handle_seq_thi",
    "handle_agent_team_refactor": "agent_team_ref",
    "handle_llm_analysis_tool": "handle_llm_ana",
    "handle_debug_analysis": "handle_debug",
    "handle_execution_trace": "handle_exec_tr",
    "_apply_proposed_changes": "apply_prop_chg",
    "handle_code_execution": "handle_code_ex",
    "_build_sequential_thinking_prompt": "build_seq_prmp",
    "handle_health_check": "handle_health",
    "handle_message": "handle_message",
    "monitor_performance": "monitor_perf"
}

df['function_trunc'] = df['function'].map(function_truncated)

# Create horizontal bar chart
fig = px.bar(df, x='length', y='function_trunc', 
             orientation='h',
             title='Top 15 Longest Functions in paste.txt',
             labels={'length': 'Lines of Code', 'function_trunc': 'Function Name'},
             color_discrete_sequence=['#1FB8CD'])

# Add data labels on bars
fig.update_traces(
    cliponaxis=False,
    texttemplate='%{x}',
    textposition='outside'
)

fig.update_yaxes(categoryorder='total ascending')

fig.write_image('top_functions_chart.png')