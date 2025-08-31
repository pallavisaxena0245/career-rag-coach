"""
Enhanced LinkedIn Job Scraper
Gets detailed job descriptions for intelligent skill analysis.
"""

import logging
import time
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import random
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class EnhancedLinkedInScraper:
    """Enhanced scraper for LinkedIn job data with detailed descriptions."""
    
    def __init__(self):
        self.session = requests.Session()
        
        # Rotate user agents to avoid detection
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Cache for job details to avoid repeated requests
        self.job_cache = {}
    
    def search_jobs_with_details(self, job_title: str, location: str = "", max_results: int = 50) -> List[Dict]:
        """
        Search for jobs and get detailed descriptions.
        
        Args:
            job_title: Job title to search for
            location: Location filter
            max_results: Maximum number of results to return
            
        Returns:
            List of job dictionaries with detailed descriptions
        """
        logger.info(f"🔍 Searching LinkedIn for detailed job data: '{job_title}' in '{location}'")
        
        # First get basic job listings
        basic_jobs = self._search_basic_jobs(job_title, location, max_results)
        
        if not basic_jobs:
            logger.warning("⚠️ No basic job listings found")
            return []
        
        # Then get detailed descriptions for each job
        detailed_jobs = []
        
        for i, job in enumerate(basic_jobs):
            try:
                # Add delay to avoid rate limiting
                if i > 0:
                    time.sleep(random.uniform(1, 3))
                
                detailed_job = self._get_job_details(job)
                if detailed_job:
                    detailed_jobs.append(detailed_job)
                    
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(basic_jobs)} job details")
                    
            except Exception as e:
                logger.error(f"❌ Error getting details for job {i}: {str(e)}")
                continue
        
        logger.info(f"✅ Successfully extracted detailed data from {len(detailed_jobs)} jobs")
        return detailed_jobs
    
    def _search_basic_jobs(self, job_title: str, location: str, max_results: int) -> List[Dict]:
        """Search for basic job listings."""
        jobs = []
        
        try:
            # Rotate user agent
            self.session.headers['User-Agent'] = random.choice(self.user_agents)
            
            # Build search parameters
            params = {
                'keywords': job_title,
                'location': location,
                'distance': '25',
                'f_TPR': 'r86400',  # Posted in last 24 hours
                'start': 0
            }
            
            page = 0
            while len(jobs) < max_results and page < 5:  # Limit to 5 pages
                params['start'] = page * 25
                
                search_url = f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"
                
                response = self.session.get(search_url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                job_cards = soup.find_all('div', class_='job-search-card')
                
                if not job_cards:
                    logger.warning(f"⚠️ No job cards found on page {page + 1}")
                    break
                
                for card in job_cards:
                    job_data = self._extract_basic_job_data(card)
                    if job_data:
                        jobs.append(job_data)
                        
                        if len(jobs) >= max_results:
                            break
                
                page += 1
                time.sleep(random.uniform(1, 2))  # Respectful delay
            
        except Exception as e:
            logger.error(f"❌ Error searching jobs: {str(e)}")
        
        return jobs[:max_results]
    
    def _extract_basic_job_data(self, job_card) -> Optional[Dict]:
        """Extract basic job information from a job card."""
        try:
            # Extract job title
            title_elem = job_card.find('h3', class_='base-search-card__title')
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Extract company
            company_elem = job_card.find('h4', class_='base-search-card__subtitle')
            company = company_elem.get_text(strip=True) if company_elem else ""
            
            # Extract location
            location_elem = job_card.find('span', class_='job-search-card__location')
            location = location_elem.get_text(strip=True) if location_elem else ""
            
            # Extract job URL
            link_elem = job_card.find('a', class_='base-card__full-link')
            job_url = link_elem.get('href') if link_elem else ""
            
            # Extract job ID from URL
            job_id = ""
            if job_url:
                import re
                job_id_match = re.search(r'/jobs/view/(\d+)', job_url)
                if job_id_match:
                    job_id = job_id_match.group(1)
            
            # Extract posting date
            date_elem = job_card.find('time')
            posted_date = date_elem.get('datetime') if date_elem else ""
            
            # Extract snippet
            snippet_elem = job_card.find('p', class_='job-search-card__snippet')
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            
            return {
                'title': title,
                'company': company,
                'location': location,
                'job_id': job_id,
                'job_url': job_url,
                'posted_date': posted_date,
                'snippet': snippet,
                'description': snippet  # Will be enhanced later
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting basic job data: {str(e)}")
            return None
    
    def _get_job_details(self, job: Dict) -> Optional[Dict]:
        """Get detailed job description for a job."""
        
        job_id = job.get('job_id')
        if not job_id:
            return job  # Return basic data if no ID available
        
        # Check cache first
        if job_id in self.job_cache:
            cached_data = self.job_cache[job_id]
            job.update(cached_data)
            return job
        
        try:
            # Rotate user agent
            self.session.headers['User-Agent'] = random.choice(self.user_agents)
            
            # Get job details page
            detail_url = f"https://www.linkedin.com/jobs/view/{job_id}"
            response = self.session.get(detail_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract full job description
            description = self._extract_full_description(soup)
            
            # Extract additional details
            additional_details = self._extract_additional_details(soup)
            
            # Update job with detailed information
            enhanced_job = job.copy()
            enhanced_job.update({
                'description': description or job.get('snippet', ''),
                'full_description': description,
                **additional_details
            })
            
            # Cache the result
            self.job_cache[job_id] = {
                'description': description,
                'full_description': description,
                **additional_details
            }
            
            return enhanced_job
            
        except Exception as e:
            logger.error(f"❌ Error getting job details for ID {job_id}: {str(e)}")
            return job  # Return basic data on error
    
    def _extract_full_description(self, soup) -> str:
        """Extract the full job description from job details page."""
        
        # Try multiple selectors for job description
        description_selectors = [
            '.description__text',
            '.show-more-less-html__markup',
            '.jobs-box__html-content',
            '.jobs-description__content',
            '.jobs-description-content__text'
        ]
        
        for selector in description_selectors:
            description_elem = soup.select_one(selector)
            if description_elem:
                # Get text and clean it up
                description = description_elem.get_text(separator=' ', strip=True)
                if len(description) > 100:  # Only return if substantial content
                    return description
        
        # Fallback: try to find any large text block
        text_blocks = soup.find_all('div', class_=lambda x: x and 'description' in x.lower())
        for block in text_blocks:
            text = block.get_text(separator=' ', strip=True)
            if len(text) > 200:
                return text
        
        return ""
    
    def _extract_additional_details(self, soup) -> Dict:
        """Extract additional job details like experience level, employment type, etc."""
        
        details = {}
        
        try:
            # Extract experience level
            experience_elem = soup.find('span', string=lambda x: x and 'experience level' in x.lower())
            if experience_elem:
                details['experience_level'] = experience_elem.get_text(strip=True)
            
            # Extract employment type
            employment_elem = soup.find('span', string=lambda x: x and ('full-time' in str(x).lower() or 'part-time' in str(x).lower()))
            if employment_elem:
                details['employment_type'] = employment_elem.get_text(strip=True)
            
            # Extract company size
            company_info = soup.find('div', class_='jobs-company__details')
            if company_info:
                size_text = company_info.get_text()
                if 'employees' in size_text.lower():
                    details['company_size'] = size_text.strip()
            
            # Extract salary information if available
            salary_elem = soup.find('span', string=lambda x: x and '$' in str(x))
            if salary_elem:
                details['salary_info'] = salary_elem.get_text(strip=True)
                
        except Exception as e:
            logger.error(f"❌ Error extracting additional details: {str(e)}")
        
        return details
    
    def get_company_specific_jobs(self, company_name: str, job_title: str, max_results: int = 10) -> List[Dict]:
        """Get jobs from a specific company."""
        logger.info(f"🏢 Searching jobs at {company_name} for '{job_title}'")
        
        try:
            # Search with company filter
            params = {
                'keywords': job_title,
                'f_C': company_name,  # Company filter
                'distance': '25',
                'start': 0
            }
            
            search_url = f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"
            
            self.session.headers['User-Agent'] = random.choice(self.user_agents)
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            job_cards = soup.find_all('div', class_='job-search-card')
            
            jobs = []
            for card in job_cards[:max_results]:
                job_data = self._extract_basic_job_data(card)
                if job_data and company_name.lower() in job_data.get('company', '').lower():
                    # Get detailed description
                    detailed_job = self._get_job_details(job_data)
                    if detailed_job:
                        jobs.append(detailed_job)
                
                # Rate limiting
                time.sleep(random.uniform(0.5, 1.5))
            
            logger.info(f"✅ Found {len(jobs)} jobs at {company_name}")
            return jobs
            
        except Exception as e:
            logger.error(f"❌ Error searching company jobs: {str(e)}")
            return []
