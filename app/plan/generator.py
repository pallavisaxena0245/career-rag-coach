from typing import List, Dict, Optional
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
logger.info("✅ LangChain OpenAI imported successfully for plan generation")

from app.config import get_openai_api_key


class StudyPlan:
    
    def __init__(self, skills: List[str], days_until_interview: int, hours_per_day: float):
        self.skills = skills
        self.days_until_interview = days_until_interview
        self.hours_per_day = hours_per_day
        self.total_hours = days_until_interview * hours_per_day
        self.plan = []
        self.video_recommendations = {}
    
    def generate_plan(self) -> Dict:
        logger.info(f"📚 Starting plan generation for {len(self.skills)} skills")
        logger.info(f"📅 Days until interview: {self.days_until_interview}")
        logger.info(f"⏰ Hours per day: {self.hours_per_day}")
        logger.info(f"📊 Total hours: {self.total_hours}")
        
        api_key = get_openai_api_key()
        if not api_key:
            logger.warning("⚠️ No API key available, using fallback plan")
            return self._generate_fallback_plan()
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        logger.info("✅ LLM initialized for plan generation")
        
        prompt = f"""
        Create a detailed study plan for interview preparation with the following constraints:
        
        Skills to prepare for: {', '.join(self.skills)}
        Days until interview: {self.days_until_interview}
        Hours available per day: {self.hours_per_day}
        Total hours available: {self.total_hours}
        
        Please create a structured plan that includes:
        1. Daily breakdown of topics to study
        2. Time allocation for each skill/topic
        3. Priority order (most important skills first)
        4. Practice exercises and mock interview suggestions
        5. YouTube video search keywords for each skill
        
        Return the response as a JSON object with the following structure:
        {{
            "daily_plan": [
                {{
                    "day": 1,
                    "date": "YYYY-MM-DD",
                    "topics": [
                        {{
                            "skill": "skill_name",
                            "time_allocation": "X hours",
                            "description": "what to study",
                            "youtube_keywords": ["keyword1", "keyword2"],
                            "practice_exercises": ["exercise1", "exercise2"]
                        }}
                    ],
                    "total_hours": X
                }}
            ],
            "priority_skills": ["skill1", "skill2", "skill3"],
            "mock_interview_schedule": [
                {{
                    "day": X,
                    "focus_areas": ["area1", "area2"],
                    "duration": "X hours"
                }}
            ]
        }}
        """
        
        logger.info(f"📝 Sending plan generation prompt to LLM")
        logger.info(f"📝 Prompt length: {len(prompt)} characters")
        
        logger.info("🔄 Invoking LLM for plan generation...")
        result = llm.invoke(prompt)
        content = getattr(result, "content", "")
        logger.info(f"✅ LLM response received, length: {len(content)} characters")
        logger.info(f"📄 Raw LLM response: {repr(content[:200])}...")
        
        logger.info("🔄 Attempting to parse JSON response...")
        
        cleaned_content = content.strip()
        if cleaned_content.startswith('```json'):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.startswith('```'):
            cleaned_content = cleaned_content[3:]
        if cleaned_content.endswith('```'):
            cleaned_content = cleaned_content[:-3]
        
        cleaned_content = cleaned_content.strip()
        logger.info(f"🧹 Cleaned content starts with: {repr(cleaned_content[:50])}")
        
        plan_data = json.loads(cleaned_content)
        logger.info("✅ JSON parsing successful")
        logger.info(f"📊 Plan data keys: {list(plan_data.keys())}")
        return plan_data
    
    def _generate_fallback_plan(self) -> Dict:
        logger.info("🔄 Generating fallback plan")
        start_date = datetime.now()
        
        daily_plan = []
        for day in range(1, self.days_until_interview + 1):
            current_date = start_date + timedelta(days=day-1)
            
            skills_per_day = max(1, len(self.skills) // self.days_until_interview)
            start_idx = (day - 1) * skills_per_day
            end_idx = min(start_idx + skills_per_day, len(self.skills))
            day_skills = self.skills[start_idx:end_idx]
            
            logger.info(f"📅 Day {day}: {len(day_skills)} skills - {day_skills}")
            
            topics = []
            for skill in day_skills:
                topics.append({
                    "skill": skill,
                    "time_allocation": f"{self.hours_per_day / len(day_skills):.1f} hours",
                    "description": f"Study and practice {skill}",
                    "youtube_keywords": [f"{skill} interview preparation", f"{skill} tutorial"],
                    "practice_exercises": [f"Practice {skill} problems", f"Mock interview questions on {skill}"]
                })
            
            daily_plan.append({
                "day": day,
                "date": current_date.strftime("%Y-%m-%d"),
                "topics": topics,
                "total_hours": self.hours_per_day
            })
        
        fallback_plan = {
            "daily_plan": daily_plan,
            "priority_skills": self.skills[:5],
            "mock_interview_schedule": [
                {
                    "day": max(1, self.days_until_interview // 2),
                    "focus_areas": self.skills[:3],
                    "duration": f"{self.hours_per_day} hours"
                }
            ]
        }
        
        logger.info(f"✅ Fallback plan generated successfully")
        logger.info(f"📊 Plan has {len(daily_plan)} days")
        logger.info(f"🎯 Priority skills: {fallback_plan['priority_skills']}")
        
        return fallback_plan


def generate_youtube_search_queries(skills: List[str]) -> List[str]:
    queries = []
    for skill in skills:
        queries.extend([
            f"{skill} interview preparation",
            f"{skill} tutorial for beginners",
            f"{skill} interview questions and answers",
            f"{skill} crash course",
            f"best {skill} videos"
        ])
    return queries


def format_plan_for_display(plan_data: Dict) -> str:
    if not plan_data:
        return "No plan generated."
    
    output = []
    
    daily_plan = plan_data.get("daily_plan", [])
    if daily_plan:
        total_days = len(daily_plan)
        total_hours = sum(day.get("total_hours", 0) for day in daily_plan)
        output.append(f"## 📅 Study Plan Summary")
        output.append(f"- **Total Days**: {total_days}")
        output.append(f"- **Total Hours**: {total_hours}")
        output.append("")
    
    priority_skills = plan_data.get("priority_skills", [])
    if priority_skills:
        output.append("## 🎯 Priority Skills")
        for i, skill in enumerate(priority_skills, 1):
            output.append(f"{i}. {skill}")
        output.append("")
    
    if daily_plan:
        output.append("## 📚 Daily Study Schedule")
        for day_data in daily_plan:
            day = day_data.get("day", 0)
            date = day_data.get("date", "")
            topics = day_data.get("topics", [])
            total_hours = day_data.get("total_hours", 0)
            
            output.append(f"### Day {day} - {date} ({total_hours} hours)")
            
            for topic in topics:
                skill = topic.get("skill", "")
                time_allocation = topic.get("time_allocation", "")
                description = topic.get("description", "")
                youtube_keywords = topic.get("youtube_keywords", [])
                
                output.append(f"**{skill}** ({time_allocation})")
                output.append(f"- {description}")
                if youtube_keywords:
                    output.append(f"- YouTube keywords: {', '.join(youtube_keywords)}")
                output.append("")
    
    mock_interviews = plan_data.get("mock_interview_schedule", [])
    if mock_interviews:
        output.append("## 🎤 Mock Interview Schedule")
        for interview in mock_interviews:
            day = interview.get("day", 0)
            focus_areas = interview.get("focus_areas", [])
            duration = interview.get("duration", "")
            
            output.append(f"**Day {day}** ({duration})")
            output.append(f"Focus areas: {', '.join(focus_areas)}")
            output.append("")
    
    return "\n".join(output)
