"""
LinkedIn Job Analyzer
Scrapes and analyzes LinkedIn job postings for market insights.
"""

import logging
import requests
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import re

from .intelligent_skill_extractor import IntelligentSkillExtractor
from .enhanced_job_scraper import EnhancedLinkedInScraper

logger = logging.getLogger(__name__)

class LinkedInJobAnalyzer:
    """Analyzes LinkedIn job postings and extracts market insights."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        self.base_url = "https://www.linkedin.com/jobs/search"
        
        # Initialize intelligent components
        self.skill_extractor = IntelligentSkillExtractor()
        self.enhanced_scraper = EnhancedLinkedInScraper()
        
    def search_jobs(self, 
                   job_title: str, 
                   location: str = "", 
                   max_results: int = 100,
                   experience_level: str = "") -> List[Dict]:
        """
        Search for jobs on LinkedIn and return job data.
        
        Args:
            job_title: The job title to search for
            location: Location filter (optional)
            max_results: Maximum number of results to return
            experience_level: Experience level filter (optional)
            
        Returns:
            List of job dictionaries with extracted information
        """
        logger.info(f"🔍 Searching LinkedIn for: '{job_title}' in '{location}'")
        
        jobs = []
        start = 0
        
        while len(jobs) < max_results:
            params = {
                'keywords': job_title,
                'location': location,
                'start': start,
                'f_TPR': 'r2592000',  # Last 30 days
            }
            
            if experience_level:
                experience_map = {
                    'entry': '1',
                    'associate': '2', 
                    'mid': '3',
                    'senior': '4',
                    'director': '5',
                    'executive': '6'
                }
                if experience_level.lower() in experience_map:
                    params['f_E'] = experience_map[experience_level.lower()]
            
            url = f"{self.base_url}?{urlencode(params)}"
            
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                job_cards = soup.find_all('div', {'data-entity-urn': True})
                
                if not job_cards:
                    logger.warning("No job cards found on page")
                    break
                
                for card in job_cards:
                    if len(jobs) >= max_results:
                        break
                        
                    job_data = self._extract_job_data(card)
                    if job_data:
                        jobs.append(job_data)
                
                start += 25
                time.sleep(2)  # Be respectful to LinkedIn's servers
                
            except Exception as e:
                logger.error(f"Error fetching jobs: {e}")
                break
        
        logger.info(f"✅ Found {len(jobs)} jobs for '{job_title}'")
        return jobs[:max_results]
    
    def _extract_job_data(self, job_card) -> Optional[Dict]:
        """Extract job information from a LinkedIn job card."""
        try:
            # Extract basic job information
            title_elem = job_card.find('h3', class_='base-search-card__title')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown"
            
            company_elem = job_card.find('h4', class_='base-search-card__subtitle')
            company = company_elem.get_text(strip=True) if company_elem else "Unknown"
            
            location_elem = job_card.find('span', class_='job-search-card__location')
            location = location_elem.get_text(strip=True) if location_elem else "Unknown"
            
            # Extract job URL
            link_elem = job_card.find('a', {'data-control-name': 'job_search_job_result'})
            job_url = link_elem.get('href') if link_elem else ""
            
            # Extract posting date
            date_elem = job_card.find('time')
            posted_date = date_elem.get_text(strip=True) if date_elem else ""
            
            # Extract job description snippet
            description_elem = job_card.find('p', class_='job-search-card__snippet')
            description = description_elem.get_text(strip=True) if description_elem else ""
            
            return {
                'title': title,
                'company': company,
                'location': location,
                'posted_date': posted_date,
                'description': description,
                'url': job_url,
                'extracted_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error extracting job data: {e}")
            return None
    
    def get_job_statistics(self, job_title: str, locations: List[str] = None) -> Dict:
        """
        Get comprehensive job statistics for a role across multiple locations.
        
        Args:
            job_title: The job title to analyze
            locations: List of locations to analyze (default: major US cities)
            
        Returns:
            Dictionary with job statistics and insights
        """
        if locations is None:
            locations = [
                "San Francisco, CA", "New York, NY", "Seattle, WA", 
                "Austin, TX", "Boston, MA", "Chicago, IL", "Remote"
            ]
        
        logger.info(f"📊 Generating job statistics for '{job_title}'")
        
        stats = {
            'job_title': job_title,
            'total_jobs': 0,
            'locations': {},
            'companies': {},
            'recent_postings': 0,
            'skills_mentioned': {},
            'salary_insights': {},
            'generated_at': time.time()
        }
        
        all_jobs = []
        
        for location in locations:
            jobs = self.search_jobs(job_title, location, max_results=50)
            all_jobs.extend(jobs)
            
            stats['locations'][location] = len(jobs)
            
            # Count companies
            for job in jobs:
                company = job.get('company', 'Unknown')
                stats['companies'][company] = stats['companies'].get(company, 0) + 1
        
        stats['total_jobs'] = len(all_jobs)
        
        # Analyze skills mentioned in job descriptions
        stats['skills_mentioned'] = self._extract_skills_from_jobs(all_jobs)
        
        # Count recent postings (last 7 days)
        recent_keywords = ['1 day ago', '2 days ago', '3 days ago', '4 days ago', 
                          '5 days ago', '6 days ago', '1 week ago']
        stats['recent_postings'] = sum(
            1 for job in all_jobs 
            if any(keyword in job.get('posted_date', '').lower() for keyword in recent_keywords)
        )
        
        logger.info(f"✅ Generated statistics: {stats['total_jobs']} total jobs found")
        return stats
    
    def get_company_job_details(self, company_name: str, job_title: str, max_results: int = 10) -> Dict:
        """
        Get detailed job information for a specific company.
        
        Args:
            company_name: Name of the company
            job_title: Job title to search for
            max_results: Maximum number of jobs to return
            
        Returns:
            Dictionary with company job details and potential contacts
        """
        logger.info(f"🏢 Getting job details for {company_name} - {job_title}")
        
        # Search for jobs at specific company
        company_jobs = self.search_jobs(f"{job_title} {company_name}", max_results=max_results)
        
        # Filter jobs that are actually from this company
        filtered_jobs = [job for job in company_jobs if company_name.lower() in job.get('company', '').lower()]
        
        # Get potential referral contacts
        referral_contacts = self._find_potential_contacts(company_name, job_title)
        
        return {
            'company': company_name,
            'job_title': job_title,
            'open_positions': filtered_jobs,
            'total_positions': len(filtered_jobs),
            'referral_contacts': referral_contacts,
            'generated_at': time.time()
        }
    
    def _find_potential_contacts(self, company_name: str, job_title: str) -> List[Dict]:
        """
        Find potential LinkedIn contacts for referrals.
        
        Args:
            company_name: Company name to search
            job_title: Related job title for finding relevant contacts
            
        Returns:
            List of potential contact profiles
        """
        logger.info(f"🔍 Finding potential contacts at {company_name}")
        
        # Common job titles that could provide referrals
        referral_titles = [
            'recruiter', 'hiring manager', 'hr', 'talent acquisition',
            'engineering manager', 'technical lead', 'senior engineer',
            'director', 'vp', 'head of engineering', 'people operations'
        ]
        
        # In a real implementation, you would use LinkedIn's People Search API
        # For now, we'll return structured contact suggestions
        contacts = []
        
        for title in referral_titles[:5]:  # Limit to 5 types
            contact = {
                'suggested_title': title.title(),
                'company': company_name,
                'linkedin_search_url': f"https://www.linkedin.com/search/results/people/?currentCompany=%5B%22{company_name.replace(' ', '%20')}%22%5D&keywords={title.replace(' ', '%20')}",
                'contact_strategy': self._get_contact_strategy(title),
                'message_template': self._get_message_template(title, job_title)
            }
            contacts.append(contact)
        
        return contacts
    
    def _get_contact_strategy(self, title: str) -> str:
        """Get contact strategy based on role."""
        strategies = {
            'recruiter': 'Best for direct job inquiries and application status',
            'hiring manager': 'Great for understanding role requirements and team culture',
            'hr': 'Good for company culture questions and referral process',
            'talent acquisition': 'Excellent for understanding hiring timeline and process',
            'engineering manager': 'Perfect for technical discussions and team insights',
            'technical lead': 'Great for technical requirements and project details',
            'senior engineer': 'Good for day-to-day work insights and technical questions',
            'director': 'Valuable for strategic role understanding and growth opportunities',
            'vp': 'Excellent for company direction and high-level insights',
            'head of engineering': 'Perfect for technical strategy and team structure',
            'people operations': 'Great for company culture and employee experience'
        }
        
        for key, strategy in strategies.items():
            if key in title.lower():
                return strategy
        
        return 'Good general contact for company insights'
    
    def _get_message_template(self, title: str, job_title: str) -> str:
        """Get personalized message template for different contact types."""
        if 'recruiter' in title.lower() or 'talent' in title.lower():
            return f"Hi [Name], I'm interested in the {job_title} position at [Company]. I'd love to learn more about the role and share my background. Would you be open to a brief conversation?"
        
        elif 'manager' in title.lower() or 'lead' in title.lower():
            return f"Hi [Name], I'm exploring {job_title} opportunities and am very interested in [Company]. I'd appreciate any insights you could share about the team and role requirements."
        
        elif 'engineer' in title.lower():
            return f"Hi [Name], I'm a fellow engineer interested in the {job_title} role at [Company]. Would you be open to sharing your experience working there and any advice for the application process?"
        
        else:
            return f"Hi [Name], I'm interested in joining [Company] as a {job_title}. I'd love to learn more about the company culture and any advice you might have for someone looking to join the team."

    def get_job_statistics_with_experience(self, job_title: str, locations: List[str] = None, experience_level: str = "") -> Dict:
        """
        Get comprehensive job statistics for a role across multiple locations with experience level filter.
        
        Args:
            job_title: The job title to analyze
            locations: List of locations to analyze (default: major US cities)
            experience_level: Experience level filter ('Entry Level', 'Associate', etc.)
            
        Returns:
            Dictionary with job statistics and insights
        """
        if locations is None:
            locations = [
                "San Francisco, CA", "New York, NY", "Seattle, WA", 
                "Austin, TX", "Boston, MA", "Chicago, IL", "Remote"
            ]
        
        exp_suffix = f" ({experience_level})" if experience_level else ""
        logger.info(f"📊 Generating job statistics for '{job_title}'{exp_suffix}")
        
        stats = {
            'job_title': job_title,
            'experience_level': experience_level,
            'total_jobs': 0,
            'locations': {},
            'companies': {},
            'recent_postings': 0,
            'skills_mentioned': {},
            'salary_insights': {},
            'generated_at': time.time()
        }
        
        all_jobs = []
        
        for location in locations:
            jobs = self.search_jobs(job_title, location, max_results=50, experience_level=experience_level)
            all_jobs.extend(jobs)
            
            stats['locations'][location] = len(jobs)
            
            # Count companies
            for job in jobs:
                company = job.get('company', 'Unknown')
                stats['companies'][company] = stats['companies'].get(company, 0) + 1
        
        stats['total_jobs'] = len(all_jobs)
        
        # Analyze skills mentioned in job descriptions
        stats['skills_mentioned'] = self._extract_skills_from_jobs(all_jobs)
        
        # Count recent postings (last 7 days)
        recent_keywords = ['1 day ago', '2 days ago', '3 days ago', '4 days ago', 
                          '5 days ago', '6 days ago', '1 week ago']
        stats['recent_postings'] = sum(
            1 for job in all_jobs 
            if any(keyword in job.get('posted_date', '').lower() for keyword in recent_keywords)
        )
        
        logger.info(f"✅ Generated statistics: {stats['total_jobs']} total jobs found{exp_suffix}")
        return stats
    
    def _extract_skills_from_jobs(self, jobs: List[Dict]) -> Dict[str, int]:
        """Extract and count skills mentioned in job descriptions with fine-grained technology detection."""
        
        # Fine-grained technical skills organized by categories
        skills_keywords = {
            # AI/ML Frameworks & Libraries
            'TensorFlow': ['tensorflow', 'tf'],
            'PyTorch': ['pytorch', 'torch'],
            'Keras': ['keras'],
            'Scikit-learn': ['scikit-learn', 'sklearn', 'scikit learn'],
            'Hugging Face': ['huggingface', 'hugging face', 'transformers'],
            'OpenAI API': ['openai', 'gpt', 'chatgpt', 'openai api'],
            'LangChain': ['langchain', 'lang chain'],
            'LlamaIndex': ['llamaindex', 'llama index', 'llama-index'],
            'Anthropic': ['anthropic', 'claude'],
            'Cohere': ['cohere'],
            
            # RAG & Vector Technologies
            'RAG': ['rag', 'retrieval augmented generation', 'retrieval-augmented'],
            'Vector Databases': ['vector database', 'vector db', 'vectordb'],
            'Pinecone': ['pinecone'],
            'Weaviate': ['weaviate'],
            'Chroma': ['chroma', 'chromadb'],
            'Qdrant': ['qdrant'],
            'Milvus': ['milvus'],
            'FAISS': ['faiss', 'facebook ai similarity search'],
            'Embeddings': ['embeddings', 'word embeddings', 'sentence embeddings'],
            'Semantic Search': ['semantic search', 'similarity search'],
            
            # LLM & NLP Technologies
            'Large Language Models': ['llm', 'large language model', 'language model'],
            'Fine-tuning': ['fine-tuning', 'fine tuning', 'model tuning'],
            'Prompt Engineering': ['prompt engineering', 'prompt design'],
            'BERT': ['bert', 'bidirectional encoder'],
            'GPT': ['gpt-3', 'gpt-4', 'gpt3', 'gpt4'],
            'T5': ['t5', 'text-to-text'],
            'BLOOM': ['bloom'],
            'PaLM': ['palm'],
            'LaMDA': ['lamda'],
            'Alpaca': ['alpaca'],
            'Vicuna': ['vicuna'],
            
            # Programming Languages
            'Python': ['python', 'py'],
            'JavaScript': ['javascript', 'js', 'node.js', 'nodejs'],
            'TypeScript': ['typescript', 'ts'],
            'Java': ['java'],
            'C++': ['c++', 'cpp'],
            'Go': ['golang', 'go'],
            'Rust': ['rust'],
            'R': [' r ', 'r programming', 'r language'],
            'Scala': ['scala'],
            'Julia': ['julia'],
            
            # ML/AI Specialized Tools
            'MLflow': ['mlflow', 'ml flow'],
            'Weights & Biases': ['wandb', 'weights and biases', 'weights & biases'],
            'Neptune': ['neptune.ai', 'neptune'],
            'ClearML': ['clearml'],
            'DVC': ['dvc', 'data version control'],
            'Kubeflow': ['kubeflow', 'kube flow'],
            'MLOps': ['mlops', 'ml ops', 'machine learning operations'],
            'AutoML': ['automl', 'automated machine learning'],
            
            # Cloud AI Services
            'AWS SageMaker': ['sagemaker', 'aws sagemaker'],
            'Google AI Platform': ['google ai platform', 'vertex ai', 'vertexai'],
            'Azure ML': ['azure ml', 'azure machine learning'],
            'AWS Bedrock': ['bedrock', 'aws bedrock'],
            'Google Colab': ['colab', 'google colab', 'colaboratory'],
            
            # Data Processing & Big Data
            'Apache Spark': ['spark', 'apache spark', 'pyspark'],
            'Hadoop': ['hadoop'],
            'Kafka': ['kafka', 'apache kafka'],
            'Airflow': ['airflow', 'apache airflow'],
            'Dask': ['dask'],
            'Ray': ['ray', 'ray.io'],
            'Pandas': ['pandas'],
            'NumPy': ['numpy', 'np'],
            'Polars': ['polars'],
            
            # Databases
            'PostgreSQL': ['postgresql', 'postgres'],
            'MongoDB': ['mongodb', 'mongo'],
            'Redis': ['redis'],
            'Elasticsearch': ['elasticsearch', 'elastic search'],
            'Neo4j': ['neo4j', 'neo 4j'],
            'ClickHouse': ['clickhouse', 'click house'],
            'Snowflake': ['snowflake'],
            'BigQuery': ['bigquery', 'big query'],
            
            # Web Frameworks
            'FastAPI': ['fastapi', 'fast api'],
            'Flask': ['flask'],
            'Django': ['django'],
            'React': ['react', 'reactjs'],
            'Vue.js': ['vue', 'vuejs', 'vue.js'],
            'Angular': ['angular', 'angularjs'],
            'Next.js': ['nextjs', 'next.js'],
            'Streamlit': ['streamlit'],
            'Gradio': ['gradio'],
            
            # DevOps & Infrastructure
            'Docker': ['docker', 'containerization'],
            'Kubernetes': ['kubernetes', 'k8s'],
            'AWS': ['aws', 'amazon web services'],
            'GCP': ['gcp', 'google cloud platform', 'google cloud'],
            'Azure': ['azure', 'microsoft azure'],
            'Terraform': ['terraform'],
            'Jenkins': ['jenkins'],
            'GitLab CI': ['gitlab ci', 'gitlab-ci'],
            'GitHub Actions': ['github actions', 'github-actions'],
            
            # Computer Vision
            'OpenCV': ['opencv', 'cv2'],
            'YOLO': ['yolo', 'you only look once'],
            'ResNet': ['resnet'],
            'EfficientNet': ['efficientnet'],
            'Vision Transformer': ['vit', 'vision transformer'],
            'CLIP': ['clip', 'contrastive language-image'],
            'Detectron': ['detectron', 'detectron2'],
            
            # Specialized AI Domains
            'Computer Vision': ['computer vision', 'cv', 'image processing'],
            'Natural Language Processing': ['nlp', 'natural language processing'],
            'Speech Recognition': ['speech recognition', 'asr', 'automatic speech recognition'],
            'Reinforcement Learning': ['reinforcement learning', 'rl'],
            'Generative AI': ['generative ai', 'gen ai', 'generative'],
            'Deep Learning': ['deep learning', 'dl', 'neural networks'],
            'Transfer Learning': ['transfer learning'],
            'Few-shot Learning': ['few-shot learning', 'few shot'],
            'Zero-shot Learning': ['zero-shot learning', 'zero shot'],
            
            # Version Control & Collaboration
            'Git': ['git'],
            'GitHub': ['github'],
            'GitLab': ['gitlab'],
            'Bitbucket': ['bitbucket'],
            
            # Testing & Quality
            'PyTest': ['pytest', 'py.test'],
            'Unit Testing': ['unit testing', 'unit tests'],
            'A/B Testing': ['a/b testing', 'ab testing'],
            'Model Testing': ['model testing', 'ml testing'],
            
            # Soft Skills & Methodologies
            'Agile': ['agile', 'scrum', 'kanban'],
            'REST API': ['rest', 'api', 'restful'],
            'GraphQL': ['graphql', 'graph ql'],
            'Microservices': ['microservices', 'microservice'],
            'CI/CD': ['ci/cd', 'continuous integration', 'continuous deployment'],
        }
        
        skill_counts = {}
        
        for job in jobs:
            description = job.get('description', '').lower()
            title = job.get('title', '').lower()
            
            # Search in both title and description
            full_text = f"{title} {description}"
            
            for skill, keywords in skills_keywords.items():
                for keyword in keywords:
                    if keyword in full_text:
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
                        break  # Don't double count if multiple keywords match
        
        # Sort by frequency
        return dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True))
    
    def compare_job_markets(self, job_titles: List[str], location: str = "") -> Dict:
        """
        Compare job market demand across multiple job titles.
        
        Args:
            job_titles: List of job titles to compare
            location: Location to focus comparison on
            
        Returns:
            Comparison data across job titles
        """
        logger.info(f"📊 Comparing job markets for: {job_titles}")
        
        comparison = {
            'location': location,
            'job_comparison': {},
            'generated_at': time.time()
        }
        
        for job_title in job_titles:
            jobs = self.search_jobs(job_title, location, max_results=30)
            
            comparison['job_comparison'][job_title] = {
                'total_jobs': len(jobs),
                'skills': self._extract_skills_from_jobs(jobs),
                'top_companies': self._get_top_companies(jobs),
                'sample_jobs': jobs[:5]  # Store sample for insights
            }
        
        logger.info(f"✅ Job market comparison completed")
        return comparison
    
    def _get_top_companies(self, jobs: List[Dict], top_n: int = 10) -> Dict[str, int]:
        """Get top companies hiring for this role."""
        companies = {}
        for job in jobs:
            company = job.get('company', 'Unknown')
            companies[company] = companies.get(company, 0) + 1
        
        return dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    def get_intelligent_job_analysis(self, job_title: str, locations: List[str] = None, experience_level: str = "", max_results: int = 50) -> Dict:
        """
        Get intelligent job market analysis using LLM-powered skill extraction.
        
        Args:
            job_title: The job title to analyze
            locations: List of locations to analyze (if None, uses default locations)
            experience_level: Experience level filter
            max_results: Maximum number of jobs to analyze per location
            
        Returns:
            Comprehensive analysis with LLM-extracted skills
        """
        logger.info(f"🤖 Starting intelligent analysis for '{job_title}' ({experience_level})")
        
        if locations is None:
            locations = ["San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Remote"]
        
        all_jobs_data = []
        location_stats = {}
        
        # Collect detailed job data from all locations
        for location in locations:
            try:
                logger.info(f"🔍 Getting detailed jobs for '{job_title}' in '{location}'")
                
                # Use enhanced scraper to get detailed job descriptions
                location_jobs = self.enhanced_scraper.search_jobs_with_details(
                    job_title=job_title,
                    location=location,
                    max_results=min(max_results, 20)  # Limit per location for performance
                )
                
                # Filter by experience level if specified
                if experience_level and experience_level != "All Levels":
                    filtered_jobs = []
                    for job in location_jobs:
                        job_exp_level = job.get('experience_level', '')
                        job_description = job.get('description', '').lower()
                        
                        # Check if job matches experience level
                        if self._matches_experience_level(job_description, job_exp_level, experience_level):
                            filtered_jobs.append(job)
                    
                    location_jobs = filtered_jobs
                
                all_jobs_data.extend(location_jobs)
                location_stats[location] = len(location_jobs)
                
                logger.info(f"✅ Found {len(location_jobs)} detailed jobs in {location}")
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error analyzing jobs in {location}: {str(e)}")
                location_stats[location] = 0
        
        if not all_jobs_data:
            logger.warning("⚠️ No jobs found for intelligent analysis")
            return {
                'total_jobs': 0,
                'analysis_method': 'intelligent_llm',
                'error': 'No jobs found for analysis'
            }
        
        # Perform intelligent skill extraction on all job descriptions
        logger.info(f"🧠 Performing LLM-powered skill analysis on {len(all_jobs_data)} jobs")
        
        intelligent_analysis = self.skill_extractor.analyze_multiple_jobs(all_jobs_data)
        
        # Generate company analysis
        company_stats = self._generate_company_stats(all_jobs_data)
        
        # Combine results
        comprehensive_stats = {
            'total_jobs': len(all_jobs_data),
            'analysis_method': 'intelligent_llm',
            'experience_level': experience_level,
            'locations_analyzed': list(locations),
            'location_distribution': location_stats,
            'intelligent_skills_analysis': intelligent_analysis,
            'company_analysis': company_stats,
            'generated_at': time.time(),
            'sample_jobs': all_jobs_data[:5]  # Include sample for debugging
        }
        
        logger.info(f"✅ Completed intelligent analysis: {len(all_jobs_data)} jobs, {intelligent_analysis['overview']['total_unique_skills']} unique skills")
        
        return comprehensive_stats
    
    def _matches_experience_level(self, job_description: str, job_exp_level: str, target_level: str) -> bool:
        """Check if job matches the target experience level."""
        
        level_keywords = {
            "Entry Level": ["entry level", "entry-level", "junior", "graduate", "new grad", "0-1 years", "0-2 years"],
            "Associate": ["associate", "1-3 years", "2-4 years"],
            "Mid Level": ["mid level", "mid-level", "intermediate", "3-5 years", "4-6 years"],
            "Senior": ["senior", "sr.", "5+ years", "6+ years", "experienced"],
            "Director": ["director", "head of", "lead", "principal", "staff"],
            "Executive": ["executive", "vp", "vice president", "chief", "cto", "ceo"]
        }
        
        target_keywords = level_keywords.get(target_level, [])
        
        # Check in explicit experience level field first
        if job_exp_level:
            for keyword in target_keywords:
                if keyword.lower() in job_exp_level.lower():
                    return True
        
        # Check in job description
        for keyword in target_keywords:
            if keyword.lower() in job_description:
                return True
        
        return False
    
    def _generate_company_stats(self, jobs_data: List[Dict]) -> Dict:
        """Generate company statistics from job data."""
        companies = {}
        for job in jobs_data:
            company = job.get('company', 'Unknown')
            if company not in companies:
                companies[company] = {
                    'job_count': 0,
                    'sample_jobs': []
                }
            companies[company]['job_count'] += 1
            if len(companies[company]['sample_jobs']) < 3:
                companies[company]['sample_jobs'].append({
                    'title': job.get('title', ''),
                    'url': job.get('job_url', ''),
                    'location': job.get('location', '')
                })
        
        # Sort by job count
        sorted_companies = dict(sorted(companies.items(), key=lambda x: x[1]['job_count'], reverse=True))
        
        return {
            'top_employers': {k: v['job_count'] for k, v in list(sorted_companies.items())[:15]},
            'company_details': sorted_companies
        }
