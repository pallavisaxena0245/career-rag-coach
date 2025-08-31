"""
Career Learning Agent
A LangGraph-powered agent that learns from job market data and adapts its recommendations.
"""

import logging
from typing import Dict, List, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import InjectedState, ToolNode
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import Tool
import json
import time

logger = logging.getLogger(__name__)

class CareerAgentState(TypedDict):
    """State for the career learning agent."""
    job_query: str
    user_background: Optional[str]
    extracted_skills: List[str]
    market_analysis: Dict
    personalized_recommendations: List[Dict]
    learning_path: List[Dict]
    agent_memory: Dict  # Agent's learned knowledge
    confidence_scores: Dict
    feedback_history: List[Dict]

class CareerLearningAgent:
    """
    A LangGraph agent that learns from job market data and user interactions
    to provide increasingly personalized career guidance.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.llm = None
        self.agent_graph = None
        self.memory_store = {}  # Persistent memory across sessions
        self._initialize_llm()
        self._build_agent_graph()
        
    def _initialize_llm(self):
        """Initialize the LLM for the agent."""
        try:
            from app.config import get_openai_api_key
            api_key = get_openai_api_key()
            
            if not api_key:
                logger.error("❌ OpenAI API key not found")
                return
                
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.2,  # Slightly creative but consistent
                api_key=api_key
            )
            logger.info(f"✅ Career agent LLM initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent LLM: {str(e)}")
            self.llm = None
    
    def _build_agent_graph(self):
        """Build the LangGraph agent workflow."""
        logger.info("🤖 Building career learning agent graph")
        
        # Create tools for the agent
        tools = [
            self._create_skill_analysis_tool(),
            self._create_market_research_tool(),
            self._create_learning_path_tool(),
            self._create_feedback_processor_tool()
        ]
        
        # Build the graph
        graph = StateGraph(CareerAgentState)
        
        # Add nodes
        graph.add_node("analyze_query", self._analyze_query_node)
        graph.add_node("research_market", self._research_market_node)
        graph.add_node("generate_recommendations", self._generate_recommendations_node)
        graph.add_node("create_learning_path", self._create_learning_path_node)
        graph.add_node("update_memory", self._update_memory_node)
        graph.add_node("tools", ToolNode(tools))
        
        # Define the flow
        graph.set_entry_point("analyze_query")
        graph.add_edge("analyze_query", "research_market")
        graph.add_edge("research_market", "generate_recommendations")
        graph.add_edge("generate_recommendations", "create_learning_path")
        graph.add_edge("create_learning_path", "update_memory")
        graph.add_edge("update_memory", END)
        
        # Add conditional edges for tool usage
        graph.add_conditional_edges(
            "generate_recommendations",
            self._should_use_tools,
            {
                "tools": "tools",
                "continue": "create_learning_path"
            }
        )
        graph.add_edge("tools", "create_learning_path")
        
        self.agent_graph = graph.compile()
        logger.info("✅ Career learning agent graph built")
    
    def _analyze_query_node(self, state: CareerAgentState) -> Dict:
        """Analyze the user's job query and extract intent."""
        logger.info("🔍 Agent: Analyzing user query")
        
        job_query = state.get("job_query", "")
        user_background = state.get("user_background", "")
        
        if not self.llm:
            logger.warning("⚠️ LLM not available for query analysis")
            return {
                "extracted_skills": [],
                "confidence_scores": {"query_analysis": 0.1}
            }
        
        # Analyze the query using LLM
        system_prompt = """You are a career analysis expert. Analyze job queries and user backgrounds to understand:
1. Specific role requirements and responsibilities
2. Technical skills needed
3. Experience level implied
4. Industry context
5. Career progression opportunities

Be specific and detailed in your analysis."""

        human_prompt = f"""
Job Query: {job_query}
User Background: {user_background or "Not provided"}

Please analyze this query and provide:
1. Core technical skills required
2. Soft skills needed
3. Experience level (entry/mid/senior)
4. Industry focus
5. Growth trajectory

Return a JSON object with these insights.
"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            analysis = self._parse_llm_response(response.content)
            
            return {
                "extracted_skills": analysis.get("technical_skills", []),
                "confidence_scores": {"query_analysis": 0.9},
                "agent_memory": {
                    "last_query_analysis": analysis,
                    "timestamp": time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in query analysis: {str(e)}")
            return {
                "extracted_skills": [],
                "confidence_scores": {"query_analysis": 0.1}
            }
    
    def _research_market_node(self, state: CareerAgentState) -> Dict:
        """Research the job market for the analyzed role."""
        logger.info("📊 Agent: Researching job market")
        
        job_query = state.get("job_query", "")
        extracted_skills = state.get("extracted_skills", [])
        
        # Use our intelligent LinkedIn analyzer
        try:
            from app.linkedin.job_analyzer import LinkedInJobAnalyzer
            analyzer = LinkedInJobAnalyzer()
            
            # Get intelligent analysis for this role
            market_data = analyzer.get_intelligent_job_analysis(
                job_title=job_query,
                max_results=20  # Smaller sample for agent processing
            )
            
            # Process the market data through the agent's lens
            market_insights = self._process_market_data(market_data, extracted_skills)
            
            return {
                "market_analysis": market_insights,
                "confidence_scores": {
                    **state.get("confidence_scores", {}),
                    "market_research": 0.8 if market_data.get("total_jobs", 0) > 0 else 0.3
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in market research: {str(e)}")
            return {
                "market_analysis": {},
                "confidence_scores": {
                    **state.get("confidence_scores", {}),
                    "market_research": 0.1
                }
            }
    
    def _generate_recommendations_node(self, state: CareerAgentState) -> Dict:
        """Generate personalized recommendations based on analysis."""
        logger.info("💡 Agent: Generating personalized recommendations")
        
        if not self.llm:
            return {"personalized_recommendations": []}
        
        job_query = state.get("job_query", "")
        extracted_skills = state.get("extracted_skills", [])
        market_analysis = state.get("market_analysis", {})
        user_background = state.get("user_background", "")
        agent_memory = state.get("agent_memory", {})
        
        # Generate recommendations using agent's accumulated knowledge
        system_prompt = """You are an AI career coach that learns and adapts. You have access to:
1. Real job market data
2. User's background and goals  
3. Your accumulated knowledge from previous interactions

Generate highly personalized, actionable recommendations that:
- Are specific to the user's situation
- Backed by real market data
- Include concrete next steps
- Consider career progression
- Adapt based on your learned knowledge

Format as a JSON array of recommendation objects."""

        context = f"""
Job Query: {job_query}
User Background: {user_background}
Extracted Skills: {extracted_skills}
Market Analysis: {json.dumps(market_analysis, indent=2)}
Agent Memory: {json.dumps(agent_memory, indent=2)}

Based on this comprehensive information, generate 5-7 personalized recommendations.
Each recommendation should have:
- category: (skill_development, networking, portfolio, certification, experience)
- title: Brief title
- description: Detailed actionable advice
- priority: (high, medium, low)
- timeline: Expected time to complete
- confidence: Your confidence in this recommendation (0-1)
"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = self.llm.invoke(messages)
            recommendations = self._parse_llm_response(response.content)
            
            if isinstance(recommendations, list):
                return {"personalized_recommendations": recommendations}
            else:
                return {"personalized_recommendations": [recommendations] if recommendations else []}
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {str(e)}")
            return {"personalized_recommendations": []}
    
    def _create_learning_path_node(self, state: CareerAgentState) -> Dict:
        """Create a structured learning path."""
        logger.info("🛤️ Agent: Creating learning path")
        
        recommendations = state.get("personalized_recommendations", [])
        extracted_skills = state.get("extracted_skills", [])
        market_analysis = state.get("market_analysis", {})
        
        # Create a structured learning path
        learning_path = []
        
        # Group recommendations by priority and timeline
        high_priority = [r for r in recommendations if r.get("priority") == "high"]
        medium_priority = [r for r in recommendations if r.get("priority") == "medium"]
        
        # Create phases
        phases = [
            {
                "phase": 1,
                "title": "Foundation Building",
                "duration": "1-2 months",
                "recommendations": high_priority[:3],
                "focus": "Core skills and immediate gaps"
            },
            {
                "phase": 2,  
                "title": "Skill Enhancement",
                "duration": "2-4 months",
                "recommendations": high_priority[3:] + medium_priority[:2],
                "focus": "Advanced skills and specialization"
            },
            {
                "phase": 3,
                "title": "Market Positioning",
                "duration": "1-2 months", 
                "recommendations": medium_priority[2:],
                "focus": "Portfolio and networking"
            }
        ]
        
        return {"learning_path": phases}
    
    def _update_memory_node(self, state: CareerAgentState) -> Dict:
        """Update the agent's memory with new learnings."""
        logger.info("🧠 Agent: Updating memory")
        
        # Extract key insights to remember
        memory_update = {
            "job_query": state.get("job_query"),
            "successful_analysis": True,
            "market_trends": self._extract_trends(state.get("market_analysis", {})),
            "recommendation_patterns": self._analyze_recommendation_patterns(state.get("personalized_recommendations", [])),
            "timestamp": time.time()
        }
        
        # Update persistent memory
        self.memory_store[f"session_{int(time.time())}"] = memory_update
        
        return {
            "agent_memory": {
                **state.get("agent_memory", {}),
                "memory_update": memory_update,
                "total_sessions": len(self.memory_store)
            }
        }
    
    def _should_use_tools(self, state: CareerAgentState) -> str:
        """Decide whether to use tools or continue."""
        confidence = state.get("confidence_scores", {}).get("market_research", 0)
        return "tools" if confidence < 0.5 else "continue"
    
    def _process_market_data(self, market_data: Dict, extracted_skills: List[str]) -> Dict:
        """Process raw market data through agent intelligence."""
        if not market_data:
            return {}
        
        # Extract key insights the agent should focus on
        intelligent_analysis = market_data.get("intelligent_skills_analysis", {})
        
        processed = {
            "total_jobs_analyzed": market_data.get("total_jobs", 0),
            "analysis_method": market_data.get("analysis_method"),
            "top_demanded_skills": intelligent_analysis.get("top_skills_overall", {}),
            "skill_categories": intelligent_analysis.get("skills_by_category", {}),
            "ai_trends": intelligent_analysis.get("ai_ml_insights", {}),
            "emerging_skills": intelligent_analysis.get("trending_skills", []),
            "skill_gaps": self._identify_skill_gaps(extracted_skills, intelligent_analysis),
            "market_strength": "strong" if market_data.get("total_jobs", 0) > 50 else "moderate"
        }
        
        return processed
    
    def _identify_skill_gaps(self, user_skills: List[str], market_analysis: Dict) -> List[str]:
        """Identify gaps between user skills and market demand."""
        market_skills = set()
        
        # Get all skills from market analysis
        for category_skills in market_analysis.get("skills_by_category", {}).values():
            market_skills.update(category_skills.keys())
        
        # Compare with user skills (case insensitive)
        user_skills_lower = [s.lower() for s in user_skills]
        gaps = []
        
        for market_skill in market_skills:
            if market_skill.lower() not in user_skills_lower:
                gaps.append(market_skill)
        
        return gaps[:10]  # Top 10 gaps
    
    def _extract_trends(self, market_analysis: Dict) -> List[str]:
        """Extract trending patterns from market analysis."""
        trends = []
        
        if market_analysis.get("emerging_skills"):
            trends.append(f"Emerging skills: {', '.join(market_analysis['emerging_skills'][:3])}")
        
        if market_analysis.get("ai_trends"):
            ai_insights = market_analysis["ai_trends"]
            if ai_insights.get("rag_adoption", 0) > 0:
                trends.append(f"RAG adoption growing: {ai_insights['rag_adoption']} mentions")
        
        return trends
    
    def _analyze_recommendation_patterns(self, recommendations: List[Dict]) -> Dict:
        """Analyze patterns in recommendations for learning."""
        if not recommendations:
            return {}
        
        patterns = {
            "most_common_category": self._get_most_common_category(recommendations),
            "avg_confidence": self._calculate_avg_confidence(recommendations),
            "priority_distribution": self._get_priority_distribution(recommendations)
        }
        
        return patterns
    
    def _get_most_common_category(self, recommendations: List[Dict]) -> str:
        """Get the most recommended category."""
        categories = [r.get("category", "unknown") for r in recommendations]
        return max(set(categories), key=categories.count) if categories else "unknown"
    
    def _calculate_avg_confidence(self, recommendations: List[Dict]) -> float:
        """Calculate average confidence in recommendations."""
        confidences = [r.get("confidence", 0.5) for r in recommendations if "confidence" in r]
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _get_priority_distribution(self, recommendations: List[Dict]) -> Dict:
        """Get distribution of recommendation priorities."""
        priorities = [r.get("priority", "medium") for r in recommendations]
        return {
            "high": priorities.count("high"),
            "medium": priorities.count("medium"), 
            "low": priorities.count("low")
        }
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response as JSON."""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Try to parse the entire response
                return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"❌ Failed to parse LLM response as JSON: {response[:200]}")
            return {}
    
    def _create_skill_analysis_tool(self):
        """Create a tool for deep skill analysis."""
        def analyze_skill_demand(skill_name: str) -> str:
            """Analyze demand for a specific skill in the job market."""
            # This would connect to our market analysis
            return f"Analyzing demand for {skill_name}..."
        
        return Tool(
            name="analyze_skill_demand",
            description="Analyze job market demand for a specific skill",
            func=analyze_skill_demand
        )
    
    def _create_market_research_tool(self):
        """Create a tool for market research."""
        def research_job_market(query: str) -> str:
            """Research the job market for specific roles or skills."""
            return f"Researching job market for: {query}..."
        
        return Tool(
            name="research_job_market", 
            description="Research job market trends and opportunities",
            func=research_job_market
        )
    
    def _create_learning_path_tool(self):
        """Create a tool for learning path generation."""
        def generate_learning_path(skills: str) -> str:
            """Generate a structured learning path for specified skills."""
            return f"Generating learning path for: {skills}..."
        
        return Tool(
            name="generate_learning_path",
            description="Generate structured learning paths for skill development", 
            func=generate_learning_path
        )
    
    def _create_feedback_processor_tool(self):
        """Create a tool for processing user feedback."""
        def process_feedback(feedback: str) -> str:
            """Process user feedback to improve recommendations."""
            return f"Processing feedback: {feedback[:50]}..."
        
        return Tool(
            name="process_feedback",
            description="Process user feedback to improve future recommendations",
            func=process_feedback
        )
    
    def analyze_career_path(self, job_query: str, user_background: str = "") -> Dict:
        """
        Main method to analyze career path using the agent.
        
        Args:
            job_query: The job role or description to analyze
            user_background: Optional user background information
            
        Returns:
            Comprehensive career analysis and recommendations
        """
        logger.info(f"🚀 Starting career analysis for: {job_query}")
        
        if not self.agent_graph:
            logger.error("❌ Agent graph not available")
            return {"error": "Agent not properly initialized"}
        
        # Initial state
        initial_state = {
            "job_query": job_query,
            "user_background": user_background,
            "extracted_skills": [],
            "market_analysis": {},
            "personalized_recommendations": [],
            "learning_path": [],
            "agent_memory": self.memory_store.copy(),  # Include previous learnings
            "confidence_scores": {},
            "feedback_history": []
        }
        
        try:
            # Run the agent workflow
            result = self.agent_graph.invoke(initial_state)
            
            logger.info("✅ Career analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in career analysis: {str(e)}")
            return {"error": str(e), "partial_result": initial_state}
    
    def provide_feedback(self, session_id: str, feedback: Dict) -> Dict:
        """Process user feedback to improve future recommendations."""
        logger.info(f"📝 Processing feedback for session: {session_id}")
        
        # Store feedback in agent memory
        feedback_entry = {
            "session_id": session_id,
            "feedback": feedback,
            "timestamp": time.time()
        }
        
        if "feedback_history" not in self.memory_store:
            self.memory_store["feedback_history"] = []
        
        self.memory_store["feedback_history"].append(feedback_entry)
        
        # Analyze feedback patterns for learning
        feedback_analysis = self._analyze_feedback_patterns()
        
        return {
            "feedback_processed": True,
            "feedback_analysis": feedback_analysis,
            "memory_updated": True
        }
    
    def _analyze_feedback_patterns(self) -> Dict:
        """Analyze feedback patterns to improve agent performance."""
        feedback_history = self.memory_store.get("feedback_history", [])
        
        if not feedback_history:
            return {}
        
        # Simple pattern analysis
        ratings = [f.get("feedback", {}).get("rating", 0) for f in feedback_history]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            "total_feedback": len(feedback_history),
            "average_rating": avg_rating,
            "improvement_trend": "positive" if avg_rating > 3.5 else "needs_improvement"
        }
