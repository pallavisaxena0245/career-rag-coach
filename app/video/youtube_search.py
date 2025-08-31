from typing import List, Dict, Optional
import json
import re
import logging
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

from yt_dlp import YoutubeDL
logger.info("✅ yt-dlp imported successfully")


class YouTubeVideoSearcher:
    
    def __init__(self):
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch',
            'max_downloads': 10,
        }
    
    def search_videos(self, query: str, max_results: int = 5) -> List[Dict]:
        logger.info(f"🔍 Searching videos for query: '{query}' (max: {max_results})")
        
        logger.info("🔄 Initializing YouTube search...") #
        with YoutubeDL(self.ydl_opts) as ydl:
            search_query = f"ytsearch{max_results}:{query}"
            logger.info(f"🔍 Search query: {search_query}") #
            
            logger.info("🔄 Extracting video info...")
            results = ydl.extract_info(search_query, download=False)
            
            if not results or 'entries' not in results:
                logger.warning("⚠️ No results found, using mock results")
                return self._mock_video_results(query, max_results)
            
            logger.info(f"📺 Found {len(results.get('entries', []))} raw results") 
            logger.info(f"📺 Results: {results}") #
            
            videos = []
            for entry in results['entries']:
                if entry:
                    video_info = {
                        'title': entry.get('title', 'Unknown Title'),
                        'url': f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                        'duration': entry.get('duration', 0),
                        'view_count': entry.get('view_count', 0),
                        'like_count': entry.get('like_count', 0),
                        'uploader': entry.get('uploader', 'Unknown'),
                        'upload_date': entry.get('upload_date', ''),
                        'description': entry.get('description', '')[:200] + '...' if entry.get('description') else '',
                        'thumbnail': entry.get('thumbnail', ''),
                    }
                    videos.append(video_info)
            
            logger.info(f"📺 Processed {len(videos)} videos")
            
            videos.sort(key=lambda x: (x['view_count'], x['like_count']), reverse=True)
            logger.info(f"✅ Returning {len(videos[:max_results])} videos")
            logger.info(f"📺 Videos: {videos[:max_results]}") #
            return videos[:max_results]
    
    def _mock_video_results(self, query: str, max_results: int) -> List[Dict]:
        return [
            {
                'title': f'Mock Video for {query} - Part {i+1}',
                'url': f'https://www.youtube.com/watch?v=mock{i}',
                'duration': 1200 + i * 300,  # 20-35 minutes
                'view_count': 100000 - i * 10000,
                'like_count': 5000 - i * 500,
                'uploader': f'Mock Channel {i+1}',
                'upload_date': '20240101',
                'description': f'This is a mock video description for {query}. This would be a real YouTube video in production.',
                'thumbnail': 'https://via.placeholder.com/320x180',
            }
            for i in range(max_results)
        ]
    
    def search_videos_for_skills(self, skills: List[str], max_videos_per_skill: int = 3, max_skills: int = 8) -> Dict[str, List[Dict]]:
        """
        Search for videos for multiple skills.
        
        Args:
            skills (List[str]): List of skills to search for
            max_videos_per_skill (int): Maximum videos per skill
            max_skills (int): Maximum number of skills to search for (to avoid too many API calls)
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping skills to their video lists
        """
        # Limit the number of skills to avoid too many API calls
        limited_skills = skills[:max_skills]
        logger.info(f"🎥 Starting video search for {len(limited_skills)} skills (limited from {len(skills)})")
        logger.info(f"📋 Skills: {limited_skills}")
        
        results = {}
        
        for skill in limited_skills:
            logger.info(f"🔍 Searching videos for skill: {skill}")
            
            # Generate fewer search queries for efficiency
            queries = [
                f"{skill} tutorial for beginners",
                f"{skill} interview preparation"
            ]
            
            logger.info(f"🔍 Search queries for {skill}: {queries}")
            
            all_videos = []
            for query in queries:
                logger.info(f"🔍 Searching: {query}")
                videos = self.search_videos(query, max_videos_per_skill)
                logger.info(f"📺 Found {len(videos)} videos for query: {query}")
                all_videos.extend(videos)
            
            unique_videos = self._remove_duplicate_videos(all_videos)
            logger.info(f"📺 Unique videos: {unique_videos}") #
            logger.info(f"📺 After deduplication: {len(unique_videos)} videos for {skill}")
            
            unique_videos.sort(key=lambda x: (x['view_count'], x['like_count']), reverse=True)
            
            results[skill] = unique_videos[:max_videos_per_skill]
            logger.info(f"✅ Final result for {skill}: {len(results[skill])} videos")
        
        logger.info(f"🎯 Total results: {len(results)} skills with videos")
        return results
    
    def _remove_duplicate_videos(self, videos: List[Dict]) -> List[Dict]:
        seen_urls = set()
        unique_videos = []
        
        for video in videos:
            video_id = self._extract_video_id(video['url'])
            if video_id and video_id not in seen_urls:
                seen_urls.add(video_id)
                unique_videos.append(video)
        
        return unique_videos
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def format_video_info(self, video: Dict) -> str:
        duration_min = video.get('duration', 0) // 60
        duration_sec = video.get('duration', 0) % 60
        duration_str = f"{duration_min}:{duration_sec:02d}" if duration_min > 0 else f"{duration_sec}s"
        
        view_count = video.get('view_count', 0)
        like_count = video.get('like_count', 0)
        
        return f"""
**{video.get('title', 'Unknown Title')}**
- **Duration**: {duration_str}
- **Views**: {view_count:,}
- **Likes**: {like_count:,}
- **Channel**: {video.get('uploader', 'Unknown')}
- **URL**: {video.get('url', 'N/A')}
        """.strip()


def search_top_videos_for_plan(plan_data: Dict, max_videos_per_skill: int = 3) -> Dict[str, List[Dict]]:
    logger.info(f"🎥 Starting video search for plan")
    logger.info(f"📋 Plan data keys: {list(plan_data.keys()) if plan_data else 'None'}")
    
    searcher = YouTubeVideoSearcher()
    
    skills = set()
    
    daily_plan = plan_data.get("daily_plan", [])
    logger.info(f"📅 Daily plan has {len(daily_plan)} days")
    
    for day_data in daily_plan:
        topics = day_data.get("topics", [])
        logger.info(f"📚 Day has {len(topics)} topics")
        for topic in topics:
            skill = topic.get("skill", "")
            if skill:
                skills.add(skill)
                logger.info(f"➕ Added skill: {skill}")
    
    priority_skills = plan_data.get("priority_skills", [])
    logger.info(f"🎯 Priority skills: {priority_skills}")
    skills.update(priority_skills)
    
    logger.info(f"📋 Total unique skills found: {len(skills)}")
    logger.info(f"📋 Skills: {list(skills)}")
    
    result = searcher.search_videos_for_skills(list(skills), max_videos_per_skill, max_skills=8)
    logger.info(f"🎯 Video search completed, found videos for {len(result)} skills")
    return result
