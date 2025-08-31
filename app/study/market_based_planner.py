"""
Market-Based Study Plan Generator
Creates personalized study plans based on LinkedIn job market analysis.
"""

import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketBasedStudyPlanner:
    """Generates study plans based on job market demand analysis."""
    
    def __init__(self):
        self.skill_priority_weights = {
            'AI/ML Frameworks': 1.0,
            'RAG & Vector Tech': 0.95,
            'LLM Technologies': 0.9,
            'Programming Languages': 0.85,
            'MLOps & Tools': 0.8,
            'Cloud AI Services': 0.75,
            'Data Processing': 0.7,
            'Computer Vision': 0.65,
            'Web Frameworks': 0.6,
            'DevOps & Infrastructure': 0.55,
            'Databases': 0.5,
            'AI Specializations': 0.45,
            'Development Tools': 0.4,
            'Methodologies': 0.35
        }
    
    def generate_study_plan(self, 
                          market_analysis: Dict, 
                          user_skills: List[str] = None,
                          study_hours_per_week: int = 10,
                          target_weeks: int = 12) -> Dict:
        """
        Generate a personalized study plan based on market analysis.
        
        Args:
            market_analysis: Comprehensive market analysis from LinkedIn data
            user_skills: List of skills user already has
            study_hours_per_week: Hours available for study per week
            target_weeks: Number of weeks for the study plan
            
        Returns:
            Structured study plan with priorities and resources
        """
        logger.info(f"📚 Generating market-based study plan for {target_weeks} weeks")
        
        if not market_analysis or 'skills_analysis' not in market_analysis:
            return {'error': 'Insufficient market analysis data'}
        
        user_skills = user_skills or []
        skills_analysis = market_analysis['skills_analysis']
        
        # Get skill priorities based on market demand
        skill_priorities = self._calculate_skill_priorities(skills_analysis, user_skills)
        
        # Generate learning path
        learning_path = self._create_learning_path(skill_priorities, target_weeks)
        
        # Create weekly schedule
        weekly_schedule = self._create_weekly_schedule(learning_path, study_hours_per_week, target_weeks)
        
        # Generate resource recommendations
        resources = self._recommend_resources(skill_priorities)
        
        return {
            'study_plan': {
                'duration_weeks': target_weeks,
                'hours_per_week': study_hours_per_week,
                'total_hours': study_hours_per_week * target_weeks,
                'generated_date': datetime.now().isoformat(),
                'market_based': True
            },
            'skill_priorities': skill_priorities,
            'learning_path': learning_path,
            'weekly_schedule': weekly_schedule,
            'resource_recommendations': resources,
            'market_insights': self._generate_study_insights(market_analysis, skill_priorities)
        }
    
    def _calculate_skill_priorities(self, skills_analysis: Dict, user_skills: List[str]) -> Dict:
        """Calculate skill learning priorities based on market demand and user gaps."""
        top_skills = skills_analysis.get('top_skills', {})
        skill_categories = skills_analysis.get('skill_categories', {})
        technology_breakdown = skills_analysis.get('technology_breakdown', {})
        
        priorities = {}
        
        # Process skills from technology breakdown for more detailed analysis
        for category, data in technology_breakdown.items():
            category_weight = self.skill_priority_weights.get(category, 0.3)
            
            for skill, count in data.get('skills', {}).items():
                # Skip if user already has this skill
                if skill.lower() in [s.lower() for s in user_skills]:
                    continue
                
                # Calculate priority score
                market_demand = count / max(top_skills.values()) if top_skills else 0
                priority_score = market_demand * category_weight
                
                priorities[skill] = {
                    'priority_score': round(priority_score, 3),
                    'market_demand': count,
                    'category': category,
                    'difficulty': self._estimate_difficulty(skill, category),
                    'learning_time_weeks': self._estimate_learning_time(skill, category)
                }
        
        # Sort by priority score
        return dict(sorted(priorities.items(), key=lambda x: x[1]['priority_score'], reverse=True))
    
    def _estimate_difficulty(self, skill: str, category: str) -> str:
        """Estimate learning difficulty for a skill."""
        difficulty_map = {
            'AI/ML Frameworks': 'Advanced',
            'RAG & Vector Tech': 'Advanced', 
            'LLM Technologies': 'Advanced',
            'Programming Languages': 'Intermediate',
            'MLOps & Tools': 'Intermediate',
            'Cloud AI Services': 'Intermediate',
            'Data Processing': 'Intermediate',
            'Computer Vision': 'Advanced',
            'Web Frameworks': 'Beginner',
            'DevOps & Infrastructure': 'Intermediate',
            'Databases': 'Beginner',
            'AI Specializations': 'Advanced',
            'Development Tools': 'Beginner',
            'Methodologies': 'Beginner'
        }
        
        return difficulty_map.get(category, 'Intermediate')
    
    def _estimate_learning_time(self, skill: str, category: str) -> int:
        """Estimate learning time in weeks for a skill."""
        time_map = {
            'AI/ML Frameworks': 6,
            'RAG & Vector Tech': 8,
            'LLM Technologies': 10,
            'Programming Languages': 12,
            'MLOps & Tools': 4,
            'Cloud AI Services': 3,
            'Data Processing': 4,
            'Computer Vision': 8,
            'Web Frameworks': 3,
            'DevOps & Infrastructure': 4,
            'Databases': 2,
            'AI Specializations': 10,
            'Development Tools': 1,
            'Methodologies': 1
        }
        
        base_time = time_map.get(category, 4)
        
        # Adjust for specific skills
        if any(term in skill.lower() for term in ['basic', 'intro', 'fundamentals']):
            return max(1, base_time - 2)
        elif any(term in skill.lower() for term in ['advanced', 'expert', 'deep']):
            return base_time + 3
        
        return base_time
    
    def _create_learning_path(self, skill_priorities: Dict, target_weeks: int) -> List[Dict]:
        """Create a structured learning path."""
        learning_path = []
        total_weeks = 0
        
        for skill, details in skill_priorities.items():
            if total_weeks >= target_weeks:
                break
            
            learning_time = min(details['learning_time_weeks'], target_weeks - total_weeks)
            
            path_item = {
                'skill': skill,
                'category': details['category'],
                'start_week': total_weeks + 1,
                'end_week': total_weeks + learning_time,
                'duration_weeks': learning_time,
                'difficulty': details['difficulty'],
                'priority_score': details['priority_score'],
                'market_demand': details['market_demand'],
                'learning_objectives': self._get_learning_objectives(skill, details['category'])
            }
            
            learning_path.append(path_item)
            total_weeks += learning_time
        
        return learning_path
    
    def _get_learning_objectives(self, skill: str, category: str) -> List[str]:
        """Get specific learning objectives for a skill."""
        objectives_map = {
            'LangChain': [
                'Understand LangChain architecture and components',
                'Build chains for different LLM tasks',
                'Implement document loaders and text splitters',
                'Create memory systems for conversational AI',
                'Deploy LangChain applications'
            ],
            'RAG': [
                'Understand Retrieval Augmented Generation concepts',
                'Implement document embedding and storage',
                'Build retrieval systems with vector databases',
                'Create RAG pipelines for Q&A systems',
                'Optimize retrieval performance and accuracy'
            ],
            'Vector Databases': [
                'Understand vector search and similarity matching',
                'Learn different vector database options',
                'Implement embedding storage and retrieval',
                'Optimize vector search performance',
                'Build production vector search systems'
            ],
            'PyTorch': [
                'Master PyTorch tensors and operations',
                'Build neural networks with nn.Module',
                'Implement training loops and optimization',
                'Work with datasets and data loaders',
                'Deploy PyTorch models'
            ]
        }
        
        return objectives_map.get(skill, [
            f'Learn fundamentals of {skill}',
            f'Practice hands-on {skill} projects',
            f'Build portfolio projects using {skill}',
            f'Apply {skill} to real-world problems'
        ])
    
    def _create_weekly_schedule(self, learning_path: List[Dict], hours_per_week: int, target_weeks: int) -> Dict:
        """Create a detailed weekly study schedule."""
        weekly_schedule = {}
        
        for week in range(1, target_weeks + 1):
            # Find skills being studied this week
            current_skills = [
                item for item in learning_path 
                if item['start_week'] <= week <= item['end_week']
            ]
            
            if current_skills:
                # Distribute hours among current skills
                hours_per_skill = hours_per_week // len(current_skills)
                remaining_hours = hours_per_week % len(current_skills)
                
                week_plan = {
                    'week': week,
                    'total_hours': hours_per_week,
                    'skills': []
                }
                
                for i, skill_item in enumerate(current_skills):
                    allocated_hours = hours_per_skill + (1 if i < remaining_hours else 0)
                    
                    week_plan['skills'].append({
                        'skill': skill_item['skill'],
                        'hours': allocated_hours,
                        'week_in_skill': week - skill_item['start_week'] + 1,
                        'progress_percentage': round(((week - skill_item['start_week'] + 1) / skill_item['duration_weeks']) * 100),
                        'activities': self._get_weekly_activities(skill_item['skill'], week - skill_item['start_week'] + 1)
                    })
                
                weekly_schedule[f'week_{week}'] = week_plan
        
        return weekly_schedule
    
    def _get_weekly_activities(self, skill: str, week_in_skill: int) -> List[str]:
        """Get specific activities for a week of skill learning."""
        if week_in_skill == 1:
            return [
                f'📚 Study {skill} fundamentals and concepts',
                f'🎥 Watch introductory tutorials',
                f'📖 Read documentation and guides'
            ]
        elif week_in_skill <= 3:
            return [
                f'💻 Practice basic {skill} exercises',
                f'🔨 Build simple {skill} projects',
                f'📝 Take notes on key concepts'
            ]
        else:
            return [
                f'🚀 Work on advanced {skill} projects',
                f'🔍 Explore {skill} best practices',
                f'📊 Apply {skill} to real-world scenarios'
            ]
    
    def _recommend_resources(self, skill_priorities: Dict) -> Dict:
        """Recommend learning resources for prioritized skills."""
        resources = {}
        
        for skill, details in list(skill_priorities.items())[:10]:  # Top 10 skills
            resources[skill] = {
                'priority': details['priority_score'],
                'youtube_keywords': self._get_youtube_keywords(skill),
                'documentation': self._get_documentation_links(skill),
                'practice_projects': self._get_practice_projects(skill),
                'certification_paths': self._get_certification_paths(skill)
            }
        
        return resources
    
    def _get_youtube_keywords(self, skill: str) -> List[str]:
        """Get YouTube search keywords for a skill."""
        base_keywords = [f"{skill} tutorial", f"{skill} course", f"learn {skill}"]
        
        specific_keywords = {
            'LangChain': ['langchain tutorial', 'langchain course', 'langchain rag', 'langchain agents'],
            'RAG': ['rag tutorial', 'retrieval augmented generation', 'rag pipeline', 'rag implementation'],
            'Vector Databases': ['vector database tutorial', 'pinecone tutorial', 'weaviate tutorial', 'chromadb tutorial'],
            'PyTorch': ['pytorch tutorial', 'pytorch course', 'deep learning pytorch', 'pytorch projects'],
            'Fine-tuning': ['fine tuning llm', 'model fine tuning', 'huggingface fine tuning', 'llm training']
        }
        
        return specific_keywords.get(skill, base_keywords)
    
    def _get_documentation_links(self, skill: str) -> List[str]:
        """Get official documentation links for skills."""
        docs_map = {
            'LangChain': ['https://docs.langchain.com/', 'https://python.langchain.com/'],
            'PyTorch': ['https://pytorch.org/docs/', 'https://pytorch.org/tutorials/'],
            'Hugging Face': ['https://huggingface.co/docs/', 'https://huggingface.co/course/'],
            'FastAPI': ['https://fastapi.tiangolo.com/', 'https://fastapi.tiangolo.com/tutorial/'],
            'Pinecone': ['https://docs.pinecone.io/', 'https://docs.pinecone.io/docs/quickstart'],
            'Weaviate': ['https://weaviate.io/developers/weaviate/', 'https://weaviate.io/developers/weaviate/quickstart']
        }
        
        return docs_map.get(skill, [f"Search for {skill} official documentation"])
    
    def _get_practice_projects(self, skill: str) -> List[str]:
        """Get practice project ideas for skills."""
        projects_map = {
            'LangChain': [
                'Build a document Q&A chatbot',
                'Create a multi-step reasoning agent',
                'Implement a code analysis tool'
            ],
            'RAG': [
                'Build a knowledge base search system',
                'Create a document summarization app',
                'Implement a FAQ answering system'
            ],
            'Vector Databases': [
                'Build a semantic search engine',
                'Create a recommendation system',
                'Implement similarity matching for images'
            ],
            'PyTorch': [
                'Build an image classifier',
                'Create a text sentiment analyzer',
                'Implement a recommendation model'
            ]
        }
        
        return projects_map.get(skill, [f'Build a project using {skill}', f'Create a {skill} portfolio piece'])
    
    def _get_certification_paths(self, skill: str) -> List[str]:
        """Get certification recommendations for skills."""
        cert_map = {
            'AWS': ['AWS Certified Machine Learning', 'AWS Certified Solutions Architect'],
            'Google Cloud': ['Google Cloud Professional ML Engineer', 'Google Cloud Associate Cloud Engineer'],
            'Azure': ['Azure AI Engineer Associate', 'Azure Data Scientist Associate'],
            'PyTorch': ['PyTorch Certified Developer (unofficial)', 'Deep Learning Specialization (Coursera)'],
            'TensorFlow': ['TensorFlow Developer Certificate', 'TensorFlow: Advanced Techniques Specialization']
        }
        
        return cert_map.get(skill, [f'Search for {skill} certifications'])
    
    def _generate_study_insights(self, market_analysis: Dict, skill_priorities: Dict) -> List[str]:
        """Generate insights about the study plan based on market analysis."""
        insights = []
        
        if 'overview' in market_analysis:
            total_jobs = market_analysis['overview'].get('total_jobs', 0)
            insights.append(f"📊 This study plan is based on analysis of {total_jobs} job postings")
        
        if skill_priorities:
            top_skill = next(iter(skill_priorities))
            insights.append(f"🎯 Focus on {top_skill} first - highest market demand with {skill_priorities[top_skill]['market_demand']} mentions")
        
        # AI-specific insights
        if 'skills_analysis' in market_analysis and 'ai_specific_insights' in market_analysis['skills_analysis']:
            ai_insights = market_analysis['skills_analysis']['ai_specific_insights']
            
            if ai_insights.get('rag_adoption', 0) > 0:
                insights.append(f"🔥 RAG technology is hot - {ai_insights['rag_adoption']} companies are adopting it")
            
            if ai_insights.get('llm_frameworks'):
                frameworks = ', '.join(ai_insights['llm_frameworks'][:3])
                insights.append(f"🚀 Key LLM frameworks in demand: {frameworks}")
        
        # Add learning strategy insights
        high_priority_skills = [skill for skill, data in skill_priorities.items() if data['priority_score'] > 0.5]
        if high_priority_skills:
            insights.append(f"⚡ {len(high_priority_skills)} high-priority skills identified for immediate focus")
        
        return insights
