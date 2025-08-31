from typing import Dict, List, Annotated
import logging

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import InjectedState
from app.extract.skills import extract_skills_for_job
from app.video.youtube_search import YouTubeVideoSearcher
from typing import TypedDict, List, Dict

class SearchState(TypedDict):
    query: str
    model_name: str
    skills: List[str]
    videos: Dict[str, List[Dict]]


def node_extract_skills(state: SearchState) -> Dict:
    
    logger.info(f"🔄 LangGraph node: Starting skills extraction")
    logger.info(f"📋 State received: {state}")
    logger.info(f"📋 State type: {type(state)}")
    logger.info(f"📋 State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
    
    query: str = state.get("query", "")
    model_name: str = state.get("model_name", "gpt-4o-mini")
    
    logger.info(f"🔍 Query: '{query}'")
    logger.info(f"📋 Model: {model_name}")
    
    if not query:
        logger.warning("⚠️ Empty query received")
        return {"skills": []}
    
    logger.info("🔄 Calling extract_skills_for_job...")
    skills = extract_skills_for_job(query, model_name=model_name)
    logger.info(f"✅ Skills extracted: {len(skills)} skills")
    logger.info(f"📋 Skills: {skills}")
    
    result = {
        "query": query,
        "model_name": model_name,
        "skills": skills
    }
    logger.info(f"🔄 Returning result: {result}")
    return result


def node_search_youtube_videos(state: SearchState) -> Dict:
    
    
    logger.info(f"🎥 LangGraph node: Starting YouTube video search")
    logger.info(f"📋 State received: {state}")
    logger.info(f"📋 State type: {type(state)}")
    logger.info(f"📋 State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
    
    skills: List[str] = state.get("skills", [])
    
    logger.info(f"🎯 Skills to search videos for: {skills}")
    
    if not skills:
        logger.warning("⚠️ No skills provided for video search")
        return {
            "query": state.get("query", ""),
            "model_name": state.get("model_name", "gpt-4o-mini"),
            "skills": skills,
            "videos": {}
        }
    
    searcher = YouTubeVideoSearcher()
    video_results = searcher.search_videos_for_skills(skills, max_skills=1)
    
    logger.info(f"✅ YouTube videos found for {len(video_results)} skills")
    logger.info(f"📺 Video results: {list(video_results.keys())}")
    
    result = {
        "query": state.get("query", ""),
        "model_name": state.get("model_name", "gpt-4o-mini"),
        "skills": skills,
        "videos": video_results
    }
    logger.info(f"🔄 Returning result with videos")
    return result


def build_graph():
    logger.info("🔧 Building LangGraph workflow")
    
    graph = StateGraph(SearchState)
    
    graph.add_node("extract_skills", node_extract_skills)
    graph.add_node("search_youtube_videos", node_search_youtube_videos)
    
    graph.set_entry_point("extract_skills")
    graph.add_edge("extract_skills", "search_youtube_videos")
    graph.add_edge("search_youtube_videos", END)
    
    logger.info("✅ LangGraph workflow built successfully")
    return graph.compile()


# Example of how to implement this as TOOLS instead of NODES:
# (This is for reference - we're using nodes in our current implementation)
"""
def tool_extract_skills(query: str, state: Annotated[SearchState, InjectedState]) -> Dict:
    '''
    Extract skills tool: Use LLM to extract skills from job description.
    
    Note: This is a TOOL function, so it requires InjectedState annotation
    to prevent the model from auto-generating parameters.
    
    Args:
        query (str): The job description to extract skills from
        state (Annotated[SearchState, InjectedState]): Current workflow state
    
    Returns:
        Dict: Extracted skills
    '''
    skills = extract_skills_for_job(query, model_name=state.get("model_name", "gpt-4o-mini"))
    return {"skills": skills}

def tool_search_youtube_videos(skill: str, state: Annotated[SearchState, InjectedState]) -> Dict:
    '''
    YouTube video search tool: Search for videos for a specific skill.
    
    Note: This is a TOOL function, so it requires InjectedState annotation.
    
    Args:
        skill (str): The skill to search videos for
        state (Annotated[SearchState, InjectedState]): Current workflow state
    
    Returns:
        Dict: Video results for the skill
    '''
    searcher = YouTubeVideoSearcher()
    videos = searcher.search_videos(f"{skill} tutorial for beginners", max_results=3)
    return {"videos": {skill: videos}}
"""
