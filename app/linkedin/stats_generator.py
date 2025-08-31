"""
Job Statistics Generator
Processes LinkedIn job data and generates statistical insights.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class JobStatsGenerator:
    """Generates comprehensive statistics from job market data."""
    
    def __init__(self):
        # Technology categories for fine-grained analysis
        self.technology_categories = {
            'AI/ML Frameworks': [
                'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Hugging Face', 
                'OpenAI API', 'LangChain', 'LlamaIndex', 'Anthropic', 'Cohere'
            ],
            'RAG & Vector Tech': [
                'RAG', 'Vector Databases', 'Pinecone', 'Weaviate', 'Chroma', 
                'Qdrant', 'Milvus', 'FAISS', 'Embeddings', 'Semantic Search'
            ],
            'LLM Technologies': [
                'Large Language Models', 'Fine-tuning', 'Prompt Engineering', 
                'BERT', 'GPT', 'T5', 'BLOOM', 'PaLM', 'LaMDA', 'Alpaca', 'Vicuna'
            ],
            'Programming Languages': [
                'Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'Go', 
                'Rust', 'R', 'Scala', 'Julia'
            ],
            'MLOps & Tools': [
                'MLflow', 'Weights & Biases', 'Neptune', 'ClearML', 'DVC', 
                'Kubeflow', 'MLOps', 'AutoML'
            ],
            'Cloud AI Services': [
                'AWS SageMaker', 'Google AI Platform', 'Azure ML', 'AWS Bedrock', 
                'Google Colab'
            ],
            'Data Processing': [
                'Apache Spark', 'Hadoop', 'Kafka', 'Airflow', 'Dask', 'Ray', 
                'Pandas', 'NumPy', 'Polars'
            ],
            'Databases': [
                'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Neo4j', 
                'ClickHouse', 'Snowflake', 'BigQuery'
            ],
            'Web Frameworks': [
                'FastAPI', 'Flask', 'Django', 'React', 'Vue.js', 'Angular', 
                'Next.js', 'Streamlit', 'Gradio'
            ],
            'DevOps & Infrastructure': [
                'Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure', 'Terraform', 
                'Jenkins', 'GitLab CI', 'GitHub Actions'
            ],
            'Computer Vision': [
                'OpenCV', 'YOLO', 'ResNet', 'EfficientNet', 'Vision Transformer', 
                'CLIP', 'Detectron'
            ],
            'AI Specializations': [
                'Computer Vision', 'Natural Language Processing', 'Speech Recognition', 
                'Reinforcement Learning', 'Generative AI', 'Deep Learning', 
                'Transfer Learning', 'Few-shot Learning', 'Zero-shot Learning'
            ],
            'Development Tools': [
                'Git', 'GitHub', 'GitLab', 'Bitbucket', 'PyTest', 'Unit Testing', 
                'A/B Testing', 'Model Testing'
            ],
            'Methodologies': [
                'Agile', 'REST API', 'GraphQL', 'Microservices', 'CI/CD'
            ]
        }
    
    def generate_comprehensive_stats(self, job_data: Dict) -> Dict:
        """
        Generate comprehensive statistics from job market data.
        
        Args:
            job_data: Job data from LinkedInJobAnalyzer
            
        Returns:
            Comprehensive statistics dictionary
        """
        logger.info(f"📊 Generating comprehensive stats for {job_data.get('total_jobs', 0)} jobs")
        
        # Check if this is intelligent LLM-based analysis
        analysis_method = job_data.get('analysis_method', 'traditional')
        
        if analysis_method == 'intelligent_llm':
            logger.info("🤖 Using LLM-powered intelligent analysis")
            stats = self._generate_intelligent_stats(job_data)
        else:
            logger.info("⚡ Using traditional keyword-based analysis")
            stats = {
                'overview': self._generate_overview_stats(job_data),
                'location_analysis': self._analyze_locations(job_data),
                'company_analysis': self._analyze_companies(job_data), 
                'skills_analysis': self._analyze_skills(job_data),
                'market_insights': self._generate_market_insights(job_data),
                'recommendations': self._generate_recommendations(job_data)
            }
        
        logger.info("✅ Comprehensive statistics generated")
        return stats
    
    def _generate_intelligent_stats(self, job_data: Dict) -> Dict:
        """Generate statistics from intelligent LLM-based analysis."""
        logger.info("🤖 Processing intelligent LLM-based analysis results")
        
        intelligent_analysis = job_data.get('intelligent_skills_analysis', {})
        
        # Convert intelligent analysis to our expected format
        overview_stats = {
            'total_jobs': job_data.get('total_jobs', 0),
            'analysis_method': 'LLM-powered intelligent extraction',
            'locations_analyzed': len(job_data.get('location_distribution', {})),
            'unique_skills_found': intelligent_analysis.get('overview', {}).get('total_unique_skills', 0),
            'experience_level': job_data.get('experience_level', 'All Levels')
        }
        
        # Process location data
        location_analysis = {
            'top_locations': job_data.get('location_distribution', {}),
            'distribution_method': 'intelligent_extraction'
        }
        
        # Process company data
        company_analysis = job_data.get('company_analysis', {})
        
        # Process intelligent skills analysis
        skills_analysis = self._process_intelligent_skills(intelligent_analysis)
        
        # Generate insights for intelligent analysis
        market_insights = self._generate_intelligent_insights(job_data, intelligent_analysis)
        
        # Generate recommendations based on intelligent analysis
        recommendations = self._generate_intelligent_recommendations(intelligent_analysis)
        
        return {
            'overview': overview_stats,
            'location_analysis': location_analysis,
            'company_analysis': company_analysis,
            'skills_analysis': skills_analysis,
            'market_insights': market_insights,
            'recommendations': recommendations,
            'raw_intelligent_data': intelligent_analysis  # For debugging
        }
    
    def _process_intelligent_skills(self, intelligent_analysis: Dict) -> Dict:
        """Process intelligent skill analysis into our expected format."""
        
        skills_by_category = intelligent_analysis.get('skills_by_category', {})
        top_skills_overall = intelligent_analysis.get('top_skills_overall', {})
        ai_ml_insights = intelligent_analysis.get('ai_ml_insights', {})
        trending_skills = intelligent_analysis.get('trending_skills', [])
        
        # Convert to our format
        processed_skills = {
            'top_skills': top_skills_overall,
            'skill_categories': {k: sum(v.values()) for k, v in skills_by_category.items()},
            'technology_breakdown': self._create_technology_breakdown(skills_by_category),
            'ai_specific_insights': ai_ml_insights,
            'trending_skills': trending_skills,
            'extraction_method': 'LLM-powered'
        }
        
        return processed_skills
    
    def _create_technology_breakdown(self, skills_by_category: Dict) -> Dict:
        """Create technology breakdown from intelligent analysis."""
        breakdown = {}
        
        for category, skills in skills_by_category.items():
            if skills:
                breakdown[category] = {
                    'total_mentions': sum(skills.values()),
                    'skills': skills,
                    'top_skill': max(skills.items(), key=lambda x: x[1]) if skills else None
                }
        
        return breakdown
    
    def _generate_intelligent_insights(self, job_data: Dict, intelligent_analysis: Dict) -> List[str]:
        """Generate insights from intelligent analysis."""
        insights = []
        
        total_jobs = job_data.get('total_jobs', 0)
        top_skills = intelligent_analysis.get('top_skills_overall', {})
        ai_insights = intelligent_analysis.get('ai_ml_insights', {})
        trending_skills = intelligent_analysis.get('trending_skills', [])
        
        insights.append(f"🤖 Analysis powered by LLM extraction from {total_jobs} real job descriptions")
        
        if top_skills:
            top_skill = max(top_skills.items(), key=lambda x: x[1])
            insights.append(f"🎯 Most demanded skill: {top_skill[0]} (mentioned in {top_skill[1]} jobs)")
        
        if ai_insights.get('rag_adoption', 0) > 0:
            insights.append(f"🔥 RAG technology adoption: {ai_insights['rag_adoption']} companies actively seeking RAG skills")
        
        if trending_skills:
            insights.append(f"📈 Trending skills identified: {', '.join(trending_skills[:3])}")
        
        return insights
    
    def _generate_intelligent_recommendations(self, intelligent_analysis: Dict) -> List[str]:
        """Generate recommendations from intelligent analysis."""
        recommendations = []
        
        top_skills = intelligent_analysis.get('top_skills_overall', {})
        ai_insights = intelligent_analysis.get('ai_ml_insights', {})
        trending_skills = intelligent_analysis.get('trending_skills', [])
        
        if top_skills:
            top_3_skills = list(top_skills.items())[:3]
            recommendations.append(f"🎯 Priority skills to learn: {', '.join([skill for skill, _ in top_3_skills])}")
        
        if ai_insights.get('llm_frameworks'):
            frameworks = ', '.join(ai_insights['llm_frameworks'][:2])
            recommendations.append(f"🚀 Focus on LLM frameworks: {frameworks}")
        
        if trending_skills:
            recommendations.append(f"📈 Consider emerging technologies: {', '.join(trending_skills[:2])}")
        
        recommendations.append("💡 Skills extracted using advanced LLM analysis - more accurate than keyword matching")
        
        return recommendations
    
    def _generate_overview_stats(self, job_data: Dict) -> Dict:
        """Generate high-level overview statistics."""
        return {
            'total_jobs': job_data.get('total_jobs', 0),
            'job_title': job_data.get('job_title', 'Unknown'),
            'recent_postings': job_data.get('recent_postings', 0),
            'locations_covered': len(job_data.get('locations', {})),
            'companies_hiring': len(job_data.get('companies', {})),
            'unique_skills_found': len(job_data.get('skills_mentioned', {})),
            'market_activity': self._calculate_market_activity(job_data)
        }
    
    def _analyze_locations(self, job_data: Dict) -> Dict:
        """Analyze job distribution by location."""
        locations = job_data.get('locations', {})
        
        if not locations:
            return {'error': 'No location data available'}
        
        total_jobs = sum(locations.values())
        
        analysis = {
            'top_locations': dict(sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10]),
            'location_percentages': {
                loc: round((count / total_jobs) * 100, 1) 
                for loc, count in locations.items()
            },
            'remote_jobs': locations.get('Remote', 0),
            'geographic_distribution': self._categorize_locations(locations)
        }
        
        return analysis
    
    def _analyze_companies(self, job_data: Dict) -> Dict:
        """Analyze hiring companies and patterns."""
        companies = job_data.get('companies', {})
        
        if not companies:
            return {'error': 'No company data available'}
        
        total_jobs = sum(companies.values())
        
        analysis = {
            'top_employers': dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:15]),
            'hiring_concentration': self._calculate_hiring_concentration(companies),
            'company_types': self._categorize_companies(companies),
            'market_competition': len(companies)
        }
        
        return analysis
    
    def _analyze_skills(self, job_data: Dict) -> Dict:
        """Analyze skill demands and trends."""
        skills = job_data.get('skills_mentioned', {})
        
        if not skills:
            return {'error': 'No skills data available'}
        
        total_mentions = sum(skills.values())
        
        analysis = {
            'top_skills': dict(sorted(skills.items(), key=lambda x: x[1], reverse=True)[:20]),
            'skill_percentages': {
                skill: round((count / total_mentions) * 100, 1)
                for skill, count in skills.items()
            },
            'skill_categories': self._categorize_skills(skills),
            'technology_breakdown': self._generate_technology_breakdown(skills),
            'emerging_skills': self._identify_emerging_skills(skills),
            'critical_skills': self._identify_critical_skills(skills, total_mentions),
            'ai_specific_insights': self._generate_ai_specific_insights(skills)
        }
        
        return analysis
    
    def _generate_market_insights(self, job_data: Dict) -> List[str]:
        """Generate actionable market insights."""
        insights = []
        
        total_jobs = job_data.get('total_jobs', 0)
        recent_postings = job_data.get('recent_postings', 0)
        locations = job_data.get('locations', {})
        skills = job_data.get('skills_mentioned', {})
        
        # Market activity insight
        if recent_postings > 0:
            activity_rate = (recent_postings / total_jobs) * 100
            if activity_rate > 30:
                insights.append(f"🔥 High market activity: {activity_rate:.1f}% of jobs posted recently")
            elif activity_rate > 15:
                insights.append(f"📈 Moderate market activity: {activity_rate:.1f}% recent postings")
            else:
                insights.append(f"📊 Stable market: {activity_rate:.1f}% recent posting rate")
        
        # Location insights
        if locations:
            top_location = max(locations.items(), key=lambda x: x[1])
            insights.append(f"🌍 Top hiring location: {top_location[0]} with {top_location[1]} jobs")
            
            if 'Remote' in locations and locations['Remote'] > 0:
                remote_percentage = (locations['Remote'] / total_jobs) * 100
                insights.append(f"🏠 {remote_percentage:.1f}% of jobs offer remote work")
        
        # Skills insights
        if skills:
            top_skill = max(skills.items(), key=lambda x: x[1])
            skill_percentage = (top_skill[1] / total_jobs) * 100
            insights.append(f"⭐ Most demanded skill: {top_skill[0]} ({skill_percentage:.1f}% of jobs)")
            
            # Technical vs soft skills analysis
            technical_skills = self._get_technical_skills(skills)
            if technical_skills:
                insights.append(f"💻 Top technical skills: {', '.join(list(technical_skills.keys())[:3])}")
        
        return insights
    
    def _generate_recommendations(self, job_data: Dict) -> List[str]:
        """Generate career recommendations based on market data."""
        recommendations = []
        
        skills = job_data.get('skills_mentioned', {})
        locations = job_data.get('locations', {})
        total_jobs = job_data.get('total_jobs', 0)
        
        if not skills or not total_jobs:
            return ["Insufficient data for recommendations"]
        
        # Skill recommendations
        top_skills = dict(sorted(skills.items(), key=lambda x: x[1], reverse=True)[:5])
        recommendations.append(f"🎯 Focus on these high-demand skills: {', '.join(top_skills.keys())}")
        
        # Location recommendations
        if locations:
            top_locations = dict(sorted(locations.items(), key=lambda x: x[1], reverse=True)[:3])
            recommendations.append(f"📍 Consider these job markets: {', '.join(top_locations.keys())}")
        
        # Market timing
        recent_postings = job_data.get('recent_postings', 0)
        if recent_postings > total_jobs * 0.2:
            recommendations.append("⏰ Good time to apply - high recent posting activity")
        
        # Skill gap analysis
        emerging_skills = self._identify_emerging_skills(skills)
        if emerging_skills:
            recommendations.append(f"🚀 Consider learning emerging skills: {', '.join(emerging_skills[:3])}")
        
        return recommendations
    
    def _calculate_market_activity(self, job_data: Dict) -> str:
        """Calculate overall market activity level."""
        total_jobs = job_data.get('total_jobs', 0)
        recent_postings = job_data.get('recent_postings', 0)
        
        if total_jobs == 0:
            return "No Data"
        
        activity_rate = (recent_postings / total_jobs) * 100
        
        if activity_rate > 30:
            return "Very High"
        elif activity_rate > 20:
            return "High"
        elif activity_rate > 10:
            return "Moderate"
        else:
            return "Low"
    
    def _categorize_locations(self, locations: Dict) -> Dict:
        """Categorize locations by region/type."""
        categories = {
            'West Coast': ['San Francisco, CA', 'Seattle, WA', 'Los Angeles, CA', 'Portland, OR'],
            'East Coast': ['New York, NY', 'Boston, MA', 'Washington, DC', 'Philadelphia, PA'],
            'Central': ['Chicago, IL', 'Austin, TX', 'Denver, CO', 'Dallas, TX'],
            'Remote': ['Remote']
        }
        
        categorized = defaultdict(int)
        
        for location, count in locations.items():
            categorized_location = False
            for category, cities in categories.items():
                if location in cities:
                    categorized[category] += count
                    categorized_location = True
                    break
            
            if not categorized_location:
                categorized['Other'] += count
        
        return dict(categorized)
    
    def _calculate_hiring_concentration(self, companies: Dict) -> Dict:
        """Calculate hiring concentration metrics."""
        total_jobs = sum(companies.values())
        company_counts = list(companies.values())
        
        # Top 10 companies concentration
        top_10_jobs = sum(sorted(company_counts, reverse=True)[:10])
        top_10_concentration = (top_10_jobs / total_jobs) * 100 if total_jobs > 0 else 0
        
        return {
            'top_10_concentration': round(top_10_concentration, 1),
            'market_fragmentation': 'Low' if top_10_concentration > 60 else 'High',
            'average_jobs_per_company': round(total_jobs / len(companies), 1) if companies else 0
        }
    
    def _categorize_companies(self, companies: Dict) -> Dict:
        """Categorize companies by type/industry.""" 
        # This is a simplified categorization - in practice, you'd use a more comprehensive mapping
        tech_giants = ['Google', 'Microsoft', 'Amazon', 'Apple', 'Meta', 'Netflix']
        startups_indicators = ['Inc.', 'LLC', 'Labs', 'Technologies']
        
        categories = defaultdict(int)
        
        for company, count in companies.items():
            if company in tech_giants:
                categories['Tech Giants'] += count
            elif any(indicator in company for indicator in startups_indicators):
                categories['Startups/Scale-ups'] += count
            else:
                categories['Other Companies'] += count
        
        return dict(categories)
    
    def _categorize_skills(self, skills: Dict) -> Dict:
        """Categorize skills by fine-grained technology categories."""
        categorized = defaultdict(int)
        
        for skill, count in skills.items():
            skill_categorized = False
            for category, skill_list in self.technology_categories.items():
                if skill in skill_list:
                    categorized[category] += count
                    skill_categorized = True
                    break
            
            if not skill_categorized:
                categorized['Other Technologies'] += count
        
        return dict(categorized)
    
    def _generate_technology_breakdown(self, skills: Dict) -> Dict:
        """Generate detailed technology breakdown by category."""
        breakdown = {}
        
        for category, skill_list in self.technology_categories.items():
            category_skills = {}
            for skill in skill_list:
                if skill in skills:
                    category_skills[skill] = skills[skill]
            
            if category_skills:
                breakdown[category] = {
                    'skills': dict(sorted(category_skills.items(), key=lambda x: x[1], reverse=True)),
                    'total_mentions': sum(category_skills.values()),
                    'top_skill': max(category_skills.items(), key=lambda x: x[1]) if category_skills else None
                }
        
        return breakdown
    
    def _generate_ai_specific_insights(self, skills: Dict) -> Dict:
        """Generate insights specific to AI/ML roles."""
        ai_insights = {
            'rag_adoption': 0,
            'llm_frameworks': [],
            'vector_tech_usage': [],
            'ai_cloud_preference': [],
            'emerging_ai_tools': []
        }
        
        # Check RAG adoption
        rag_related = ['RAG', 'Vector Databases', 'Embeddings', 'Semantic Search', 'Pinecone', 'Weaviate', 'Chroma']
        ai_insights['rag_adoption'] = sum(skills.get(tech, 0) for tech in rag_related)
        
        # LLM frameworks in use
        llm_frameworks = ['LangChain', 'LlamaIndex', 'Hugging Face', 'OpenAI API', 'Anthropic']
        ai_insights['llm_frameworks'] = [fw for fw in llm_frameworks if skills.get(fw, 0) > 0]
        
        # Vector technology usage
        vector_techs = ['Pinecone', 'Weaviate', 'Chroma', 'Qdrant', 'Milvus', 'FAISS']
        ai_insights['vector_tech_usage'] = [tech for tech in vector_techs if skills.get(tech, 0) > 0]
        
        # AI cloud service preference
        cloud_ai = ['AWS SageMaker', 'Google AI Platform', 'Azure ML', 'AWS Bedrock']
        ai_insights['ai_cloud_preference'] = [service for service in cloud_ai if skills.get(service, 0) > 0]
        
        # Emerging AI tools
        emerging_ai = ['Fine-tuning', 'Prompt Engineering', 'MLOps', 'AutoML', 'Vision Transformer']
        ai_insights['emerging_ai_tools'] = [tool for tool in emerging_ai if skills.get(tool, 0) > 0]
        
        return ai_insights
    
    def _identify_emerging_skills(self, skills: Dict) -> List[str]:
        """Identify emerging/trending skills with focus on cutting-edge technologies."""
        # Updated emerging technology keywords
        emerging_keywords = [
            'RAG', 'LangChain', 'Vector Databases', 'Fine-tuning', 'Prompt Engineering',
            'Generative AI', 'LlamaIndex', 'Embeddings', 'MLOps', 'AutoML',
            'Vision Transformer', 'CLIP', 'Zero-shot Learning', 'Few-shot Learning'
        ]
        
        emerging = []
        for skill in skills:
            if skill in emerging_keywords or any(keyword in skill for keyword in ['GPT', 'LLM', 'Transformer']):
                emerging.append(skill)
        
        # Sort by mention count and return top emerging skills
        emerging_with_counts = [(skill, skills[skill]) for skill in emerging if skill in skills]
        emerging_sorted = sorted(emerging_with_counts, key=lambda x: x[1], reverse=True)
        
        return [skill for skill, _ in emerging_sorted[:8]]
    
    def _identify_critical_skills(self, skills: Dict, total_mentions: int) -> List[str]:
        """Identify critical skills (mentioned in >50% of jobs)."""
        critical_threshold = total_mentions * 0.5
        
        critical_skills = []
        for skill, count in skills.items():
            if count >= critical_threshold:
                critical_skills.append(skill)
        
        return critical_skills
    
    def _get_technical_skills(self, skills: Dict) -> Dict:
        """Filter and return only technical skills."""
        technical_categories = ['Programming Languages', 'Frameworks', 'Databases', 'Cloud/DevOps', 'Data/AI', 'Tools']
        categorized = self._categorize_skills(skills)
        
        technical_skills = {}
        for skill, count in skills.items():
            # Check if skill belongs to technical categories
            for category in technical_categories:
                if category in categorized and skill in self._get_skills_in_category(category):
                    technical_skills[skill] = count
                    break
        
        return dict(sorted(technical_skills.items(), key=lambda x: x[1], reverse=True))
    
    def _get_skills_in_category(self, category: str) -> List[str]:
        """Get list of skills for a specific category."""
        category_mapping = {
            'Programming Languages': ['Python', 'JavaScript', 'Java', 'C++', 'TypeScript', 'Go', 'Rust', 'C#', 'PHP', 'Ruby'],
            'Frameworks': ['React', 'Angular', 'Vue', 'Django', 'Flask', 'Spring', 'Express'],
            'Databases': ['SQL', 'NoSQL', 'Redis', 'Elasticsearch'],
            'Cloud/DevOps': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'Terraform', 'CI/CD', 'DevOps'],
            'Data/AI': ['Machine Learning', 'Data Science', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'NumPy'],
            'Tools': ['Git', 'Jira', 'Slack', 'Figma', 'Postman']
        }
        
        return category_mapping.get(category, [])
