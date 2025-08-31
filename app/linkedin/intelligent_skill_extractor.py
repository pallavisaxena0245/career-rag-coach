"""
Intelligent Skill Extractor for LinkedIn Job Descriptions
Uses LLM to analyze job descriptions and extract skills dynamically.
"""

import logging
import time
from typing import Dict, List, Set
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import re

logger = logging.getLogger(__name__)

class IntelligentSkillExtractor:
    """LLM-powered skill extraction from job descriptions."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.llm = None
        self._initialize_llm()
        
        # Cache for processed job descriptions to avoid repeated LLM calls
        self.processed_cache = {}
        
    def _initialize_llm(self):
        """Initialize the LLM for skill extraction."""
        try:
            from app.config import get_openai_api_key
            api_key = get_openai_api_key()
            
            if not api_key:
                logger.error("❌ OpenAI API key not found")
                return
                
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.1,  # Low temperature for consistent extraction
                api_key=api_key
            )
            logger.info(f"✅ Initialized LLM for skill extraction: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {str(e)}")
            self.llm = None
    
    def extract_skills_from_job_description(self, job_description: str, job_title: str = "") -> Dict:
        """
        Extract skills from a single job description using LLM.
        
        Args:
            job_description: The job description text
            job_title: Optional job title for context
            
        Returns:
            Dictionary with extracted skills categorized by type
        """
        if not self.llm:
            logger.warning("⚠️ LLM not available, using fallback extraction")
            return self._fallback_extraction(job_description)
        
        # Check cache first
        cache_key = f"{hash(job_description)}_{job_title}"
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]
        
        try:
            system_prompt = """You are an expert technical recruiter and skill analyst. Your job is to extract ALL technical skills, tools, technologies, frameworks, programming languages, and methodologies mentioned in job descriptions.

IMPORTANT INSTRUCTIONS:
1. Extract ONLY skills that are explicitly mentioned in the job description
2. DO NOT infer or add skills that aren't directly stated
3. Categorize skills into these types:
   - programming_languages: Python, JavaScript, Java, etc.
   - frameworks_libraries: React, Django, TensorFlow, etc.
   - tools_platforms: Docker, Kubernetes, AWS, etc.
   - databases: MySQL, MongoDB, PostgreSQL, etc.
   - methodologies: Agile, DevOps, CI/CD, etc.
   - ai_ml_specific: RAG, LangChain, Vector Databases, etc.
   - soft_skills: Communication, Leadership, etc.

4. Be specific - if they mention "machine learning frameworks" and specifically name "PyTorch", extract "PyTorch", not just "machine learning"
5. Extract both acronyms and full names if mentioned (e.g., "AI" and "Artificial Intelligence")
6. Pay special attention to modern AI/ML technologies like RAG, LangChain, Vector Databases, LLMs, etc.

Return ONLY a valid JSON object with this exact structure:
{
    "programming_languages": ["skill1", "skill2"],
    "frameworks_libraries": ["skill1", "skill2"],
    "tools_platforms": ["skill1", "skill2"],
    "databases": ["skill1", "skill2"],
    "methodologies": ["skill1", "skill2"],
    "ai_ml_specific": ["skill1", "skill2"],
    "soft_skills": ["skill1", "skill2"]
}"""

            human_prompt = f"""Job Title: {job_title}

Job Description:
{job_description[:3000]}  # Limit to avoid token limits

Extract all skills mentioned in this job description and categorize them according to the instructions."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = self._parse_llm_response(response.content)
            
            # Cache the result
            self.processed_cache[cache_key] = result
            
            logger.info(f"✅ Extracted {sum(len(v) for v in result.values())} skills from job description")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error extracting skills with LLM: {str(e)}")
            return self._fallback_extraction(job_description)
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response and extract the JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM response as JSON: {str(e)}")
            logger.debug(f"Response was: {response[:500]}")
            return {
                "programming_languages": [],
                "frameworks_libraries": [],
                "tools_platforms": [],
                "databases": [],
                "methodologies": [],
                "ai_ml_specific": [],
                "soft_skills": []
            }
    
    def _fallback_extraction(self, job_description: str) -> Dict:
        """Fallback extraction using keyword matching when LLM is unavailable."""
        logger.info("🔄 Using fallback keyword-based extraction")
        
        # Common keywords for fallback
        keywords = {
            "programming_languages": [
                "python", "javascript", "java", "c++", "c#", "go", "rust", "typescript",
                "scala", "kotlin", "swift", "php", "ruby", "r programming"
            ],
            "frameworks_libraries": [
                "react", "angular", "vue", "django", "flask", "spring", "tensorflow",
                "pytorch", "keras", "pandas", "numpy", "scikit-learn", "langchain"
            ],
            "tools_platforms": [
                "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "git",
                "terraform", "ansible", "prometheus", "grafana"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                "cassandra", "dynamodb", "snowflake"
            ],
            "methodologies": [
                "agile", "scrum", "devops", "ci/cd", "tdd", "microservices",
                "rest api", "graphql"
            ],
            "ai_ml_specific": [
                "rag", "retrieval augmented generation", "vector database",
                "llm", "large language model", "machine learning", "deep learning",
                "natural language processing", "nlp", "computer vision"
            ],
            "soft_skills": [
                "communication", "leadership", "teamwork", "problem solving",
                "analytical thinking", "project management"
            ]
        }
        
        text_lower = job_description.lower()
        result = {}
        
        for category, skill_list in keywords.items():
            found_skills = []
            for skill in skill_list:
                if skill.lower() in text_lower:
                    found_skills.append(skill.title())
            result[category] = found_skills
        
        return result
    
    def analyze_multiple_jobs(self, jobs_data: List[Dict], batch_size: int = 5) -> Dict:
        """
        Analyze multiple job descriptions and aggregate skill insights.
        
        Args:
            jobs_data: List of job dictionaries with 'description' and 'title' fields
            batch_size: Number of jobs to process in each batch
            
        Returns:
            Aggregated skill analysis results
        """
        logger.info(f"🔍 Starting intelligent analysis of {len(jobs_data)} job descriptions")
        
        all_skills = {
            "programming_languages": {},
            "frameworks_libraries": {},
            "tools_platforms": {},
            "databases": {},
            "methodologies": {},
            "ai_ml_specific": {},
            "soft_skills": {}
        }
        
        processed_count = 0
        
        for i in range(0, len(jobs_data), batch_size):
            batch = jobs_data[i:i + batch_size]
            
            for job in batch:
                description = job.get('description', '')
                title = job.get('title', '')
                
                # If no description or very short description, generate one using LLM
                if not description.strip() or len(description.strip()) < 50:
                    logger.info(f"📝 Generating description for job: {title}")
                    description = self._generate_job_description_from_title(title, job)
                
                if not description.strip() or len(description.strip()) < 50:
                    logger.warning(f"⚠️ Skipping job with insufficient description: {title}")
                    continue
                
                # Extract skills from this job
                job_skills = self.extract_skills_from_job_description(description, title)
                
                # Aggregate skills with counts
                for category, skills in job_skills.items():
                    for skill in skills:
                        if skill not in all_skills[category]:
                            all_skills[category][skill] = 0
                        all_skills[category][skill] += 1
                
                processed_count += 1
                
                # Add small delay to avoid rate limiting
                if self.llm:
                    time.sleep(0.1)
            
            logger.info(f"📊 Processed {min(i + batch_size, len(jobs_data))}/{len(jobs_data)} jobs")
        
        # Generate insights
        insights = self._generate_skill_insights(all_skills, processed_count)
        
        logger.info(f"✅ Completed intelligent analysis of {processed_count} job descriptions")
        return insights
    
    def _generate_job_description_from_title(self, job_title: str, job_info: Dict) -> str:
        """
        Generate a realistic job description from job title and basic info using LLM.
        
        Args:
            job_title: The job title
            job_info: Additional job information (company, location, etc.)
            
        Returns:
            Generated job description
        """
        if not self.llm:
            logger.warning("⚠️ LLM not available for job description generation")
            return ""
        
        try:
            company = job_info.get('company', 'a leading company')
            location = job_info.get('location', 'remote/hybrid')
            
            system_prompt = """You are an expert technical recruiter. Generate realistic, detailed job descriptions based on job titles and basic information. Focus on:

1. Specific technical skills and technologies commonly required
2. Typical responsibilities and requirements  
3. Tools and frameworks used in the industry
4. Experience levels and qualifications

Make the description detailed and technical, mentioning specific technologies, frameworks, and skills that would realistically be required for this role."""

            human_prompt = f"""
Generate a detailed, realistic job description for:

Job Title: {job_title}
Company: {company}
Location: {location}

Include specific technical requirements, skills, tools, frameworks, and technologies that would typically be mentioned in a real job posting for this role. Make it comprehensive and industry-appropriate.

Format as a natural job description with responsibilities and requirements sections.
"""

            from langchain.schema import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            generated_description = response.content if hasattr(response, 'content') else str(response)
            
            if len(generated_description) > 100:
                logger.info(f"✅ Generated {len(generated_description)} character description for {job_title}")
                return generated_description
            else:
                logger.warning(f"⚠️ Generated description too short for {job_title}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error generating job description for {job_title}: {str(e)}")
            return ""
    
    def _generate_skill_insights(self, all_skills: Dict, total_jobs: int) -> Dict:
        """Generate insights from aggregated skill data."""
        
        # Get top skills in each category
        top_skills_by_category = {}
        for category, skills in all_skills.items():
            if skills:
                # Sort by count and get top 10
                sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
                top_skills_by_category[category] = dict(sorted_skills[:10])
        
        # Get overall top skills
        all_skills_flat = {}
        for category, skills in all_skills.items():
            for skill, count in skills.items():
                all_skills_flat[skill] = count
        
        top_skills_overall = dict(sorted(all_skills_flat.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # Generate technology insights
        ai_ml_insights = self._generate_ai_ml_insights(all_skills, total_jobs)
        
        # Generate market trends
        trending_skills = self._identify_trending_skills(all_skills, total_jobs)
        
        return {
            "overview": {
                "total_jobs_analyzed": total_jobs,
                "total_unique_skills": len(all_skills_flat),
                "analysis_method": "LLM-powered extraction"
            },
            "top_skills_overall": top_skills_overall,
            "skills_by_category": top_skills_by_category,
            "ai_ml_insights": ai_ml_insights,
            "trending_skills": trending_skills,
            "raw_skills_data": all_skills
        }
    
    def _generate_ai_ml_insights(self, all_skills: Dict, total_jobs: int) -> Dict:
        """Generate specific insights for AI/ML technologies."""
        ai_skills = all_skills.get("ai_ml_specific", {})
        frameworks = all_skills.get("frameworks_libraries", {})
        
        insights = {}
        
        # RAG adoption
        rag_mentions = ai_skills.get("RAG", 0) + ai_skills.get("Retrieval Augmented Generation", 0)
        insights["rag_adoption"] = rag_mentions
        
        # LLM frameworks
        llm_frameworks = []
        llm_keywords = ["langchain", "llamaindex", "hugging face", "openai", "anthropic"]
        for framework, count in frameworks.items():
            if any(keyword in framework.lower() for keyword in llm_keywords):
                llm_frameworks.append(framework)
        insights["llm_frameworks"] = llm_frameworks[:5]
        
        # Vector technologies
        vector_tech = []
        vector_keywords = ["vector", "pinecone", "weaviate", "chroma", "faiss", "qdrant"]
        for skill, count in ai_skills.items():
            if any(keyword in skill.lower() for keyword in vector_keywords):
                vector_tech.append(skill)
        insights["vector_tech_usage"] = vector_tech[:5]
        
        # Popular ML frameworks
        ml_frameworks = []
        ml_keywords = ["tensorflow", "pytorch", "keras", "scikit-learn"]
        for framework, count in frameworks.items():
            if any(keyword in framework.lower() for keyword in ml_keywords):
                ml_frameworks.append(framework)
        insights["popular_ml_frameworks"] = ml_frameworks[:5]
        
        return insights
    
    def _identify_trending_skills(self, all_skills: Dict, total_jobs: int) -> List[str]:
        """Identify trending/emerging skills based on frequency and recency."""
        
        # Skills that are considered "emerging" or "trending"
        trending_keywords = [
            "rag", "langchain", "vector database", "llm", "gpt", "claude",
            "kubernetes", "terraform", "microservices", "serverless",
            "rust", "go", "typescript", "next.js", "react native"
        ]
        
        trending_skills = []
        
        for category, skills in all_skills.items():
            for skill, count in skills.items():
                # Consider a skill trending if it's mentioned in at least 10% of jobs
                # and matches trending keywords
                if count >= max(1, total_jobs * 0.1):
                    for keyword in trending_keywords:
                        if keyword.lower() in skill.lower():
                            trending_skills.append(skill)
                            break
        
        return trending_skills[:10]
