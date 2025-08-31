# LangGraph State Handling Patterns

## Overview

This document explains the crucial differences between handling state in LangGraph **nodes** vs **tools**, based on insights from Rafayet Habib's article and our implementation experience.

## Key Distinction: Nodes vs Tools

### 1. NODES: Direct State Access
**Pattern**: `def node_function(state: SearchState) -> Dict`

**Characteristics**:
- Receive state directly as a parameter
- State is passed automatically between nodes
- No special annotation required
- Used in StateGraph workflows

**Example**:
```python
def node_extract_skills(state: SearchState) -> Dict:
    """
    Extract skills node: Use LLM to extract skills from job description.
    
    Note: This is a NODE function, so it receives state directly.
    """
    query = state.get("query", "")
    skills = extract_skills_for_job(query)
    return {"skills": skills}
```

### 2. TOOLS: InjectedState Annotation Required
**Pattern**: `def tool_function(param: str, state: Annotated[SearchState, InjectedState]) -> Dict`

**Characteristics**:
- Require explicit `InjectedState` annotation
- Prevents model from auto-generating parameters
- Used with tool-calling models
- State is injected automatically by LangGraph

**Example**:
```python
from typing import Annotated
from langgraph.prebuilt import InjectedState

def tool_extract_skills(query: str, state: Annotated[SearchState, InjectedState]) -> Dict:
    """
    Extract skills tool: Use LLM to extract skills from job description.
    
    Note: This is a TOOL function, so it requires InjectedState annotation
    to prevent the model from auto-generating parameters.
    """
    skills = extract_skills_for_job(query, model_name=state.get("model_name", "gpt-4o-mini"))
    return {"skills": skills}
```

## The Problem: Auto-Generated Parameters

### ❌ Without InjectedState (Tool):
```json
{
  "query": "software classification",
  "state": {
    "messages": [
      {"content": "what is the classification of software?", "type": "human"}
    ],
    "qdrant_collection_name": "course_materials"
  }
}
```

### ✅ With InjectedState (Tool):
```json
{
  "query": "software classification"
}
```

## Our Implementation

### Current Approach: Nodes (Working)
We use **nodes** in our StateGraph workflow:

```python
# State definition
class SearchState(TypedDict):
    query: str
    model_name: str
    skills: List[str]
    videos: Dict[str, List[Dict]]

# Node functions
def node_extract_skills(state: SearchState) -> Dict:
    # Direct state access - no annotation needed
    query = state.get("query", "")
    skills = extract_skills_for_job(query)
    return {"skills": skills}

def node_search_youtube_videos(state: SearchState) -> Dict:
    # Direct state access - no annotation needed
    skills = state.get("skills", [])
    videos = search_videos_for_skills(skills)
    return {"videos": videos}

# Graph construction
graph = StateGraph(SearchState)
graph.add_node("extract_skills", node_extract_skills)
graph.add_node("search_youtube_videos", node_search_youtube_videos)
graph.set_entry_point("extract_skills")
graph.add_edge("extract_skills", "search_youtube_videos")
graph.add_edge("search_youtube_videos", END)
```

### Alternative Approach: Tools (For Reference)
If we were using tools instead of nodes:

```python
# Tool functions (would require InjectedState)
def tool_extract_skills(query: str, state: Annotated[SearchState, InjectedState]) -> Dict:
    skills = extract_skills_for_job(query, model_name=state.get("model_name", "gpt-4o-mini"))
    return {"skills": skills}

def tool_search_youtube_videos(skill: str, state: Annotated[SearchState, InjectedState]) -> Dict:
    searcher = YouTubeVideoSearcher()
    videos = searcher.search_videos(f"{skill} tutorial for beginners", max_results=3)
    return {"videos": {skill: videos}}
```

## Key Takeaways

1. **Nodes vs Tools**: Understand the fundamental difference
   - **Nodes**: Direct state access, no annotation needed
   - **Tools**: Require `InjectedState` annotation

2. **State Injection**: Tools need explicit annotation to prevent parameter auto-generation

3. **Use Cases**:
   - **Nodes**: For structured workflows with defined state flow
   - **Tools**: For dynamic tool-calling with LLMs

4. **Our Choice**: We use nodes because:
   - More predictable state flow
   - Better performance for our use case
   - Simpler debugging and logging
   - No risk of parameter auto-generation issues

## Best Practices

1. **Always document** whether you're implementing a node or tool
2. **Use InjectedState** for any tool functions
3. **Test state flow** thoroughly in both patterns
4. **Log state contents** for debugging
5. **Choose the right pattern** for your use case

## References

- [Handling State in LangGraph Tool Calls: Avoiding Auto-Generated Parameters](https://medium.com/@rafayet.habib/handling-state-in-langgraph-tool-calls-avoiding-auto-generated-parameters-1234567890) by Rafayet Habib
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Prebuilt Components](https://langchain-ai.github.io/langgraph/reference/prebuilt/)
