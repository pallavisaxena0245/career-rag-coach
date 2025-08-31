import os
import logging
from typing import Dict
import streamlit as st

from app.config import require_packages_ready, get_openai_api_key
from app.graph.flow import build_graph
from app.video import YouTubeVideoSearcher
from app.linkedin import LinkedInJobAnalyzer, JobStatsGenerator, SkillVisualization
from app.study import MarketBasedStudyPlanner
from app.agents import CareerLearningAgent

# -----------------------
# Helpers
# -----------------------
def stable_key(prompt: str) -> str:
    """Create a stable key from the job prompt (hash ensures uniqueness)."""
    return str(abs(hash(prompt)))

def skill_selector(job_prompt: str, clean_skills: list):
    """Render the multiselect for skills and return selected ones."""
    skill_key = f"selected_skills_{stable_key(job_prompt)}"

    # Render multiselect — Streamlit handles persistence with key
    st.multiselect(
        "Choose skills for video recommendations:",
        options=clean_skills,
        key=skill_key,
        help="Select the skills you want to learn more about"
    )

    # Always read current state
    selected_skills = st.session_state.get(skill_key, [])

    # Fallback to last selection if empty
    if not selected_skills and "last_selected_skills" in st.session_state:
        selected_skills = st.session_state["last_selected_skills"]

    return selected_skills

# -----------------------
# App Config
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Career RAG Coach", layout="wide")
st.title("🚀 Career RAG Coach - Skills & Market Analysis")
st.caption("Extract skills from job descriptions, find YouTube tutorials, and analyze the job market trends.")

with st.sidebar:
    st.subheader("Configuration")
    model_name = st.text_input("OpenAI model", value="gpt-4o-mini")
    st.caption("Requires OPENAI_API_KEY in environment or .env file.")

    # Debug
    debug_enabled = st.checkbox("🔧 Show Debug Information")
    st.session_state['show_debug'] = debug_enabled

    # Session management
    st.subheader("Session Management")
    if st.button("🗑️ Clear All Session Data"):
        st.session_state.clear()
        st.rerun()

# Add navigation tabs
tab1, tab2, tab3 = st.tabs(["🎯 Skills & Videos", "📊 Market Analysis", "🤖 AI Career Coach"])

# Skills and Videos Tab (existing functionality)
with tab1:
    # -----------------------
    # Job Prompt Input
    # -----------------------
job_prompt = st.text_input("Describe the job/role (e.g., 'Senior Data Scientist in fintech')")
run_btn = st.button("Extract Skills")

missing = require_packages_ready()
if missing:
    st.warning("Missing packages: " + ", ".join(missing) + ". Install requirements and reload.")

    # -----------------------
    # Extract Skills
    # -----------------------
if run_btn:
    if not job_prompt.strip():
        st.error("Please enter a job description or title.")
            st.stop()

        if not get_openai_api_key():
            st.warning("OPENAI_API_KEY not found. Skill extraction will not run.")
            st.stop()

        if missing:
            st.stop()

        graph = build_graph()
        if graph is None:
            st.error("LangGraph is not available. Please install dependencies.")
            st.stop()

        skills_cache_key = f"skills_cache_{stable_key(job_prompt)}"
        if skills_cache_key in st.session_state:
            skills = st.session_state[skills_cache_key]
            logger.info("✅ Using cached skills")
        else:
        with st.spinner("Extracting skills..."):
                from app.extract.skills import extract_skills_for_job
                skills = extract_skills_for_job(job_prompt, model_name=model_name)
                st.session_state[skills_cache_key] = skills
                logger.info("✅ Skills extracted and cached")

    # -----------------------
    # Display Skills (Always show if cached)
    # -----------------------
    if job_prompt.strip():
        skills_cache_key = f"skills_cache_{stable_key(job_prompt)}"
        if skills_cache_key in st.session_state:
            skills = st.session_state[skills_cache_key]
            clean_skills = [str(s).strip() for s in skills if s and str(s).strip()]
            
            st.subheader("Extracted Skills")
            if clean_skills:
                st.success(f"✅ Found {len(clean_skills)} skills")
                st.write("\n".join(f"- {s}" for s in clean_skills))

                # -----------------------
                # YouTube Section
                # -----------------------
                st.subheader("🎥 YouTube Video Recommendations")
                selected_skills = skill_selector(job_prompt, clean_skills)

                if selected_skills:
                    st.write(f"🎯 Selected skills: {', '.join(selected_skills)}")

                    if st.button("🔍 Search YouTube Videos", type="primary", key=f"search_btn_{stable_key(job_prompt)}"):
                        # Cache last selection
                        st.session_state["last_selected_skills"] = selected_skills

                        with st.spinner("Searching for top YouTube videos..."):
                            try:
                                searcher = YouTubeVideoSearcher()
                                video_results = searcher.search_videos_for_skills(
                                    selected_skills, max_skills=len(selected_skills)
                                )

                                video_key = f"video_results_{stable_key(job_prompt)}"
                                st.session_state[video_key] = video_results

                                if video_results:
                                    for skill, videos in video_results.items():
                                        with st.expander(f"📺 Videos for {skill}"):
                                            for i, video in enumerate(videos, 1):
                                                st.markdown(f"**{i}. {video['title']}**")
                                                st.markdown(f"- [Watch here]({video['url']})")
                                                st.markdown("---")
                                else:
                                    st.info("No video recommendations available.")
            except Exception as e:
                                st.error(f"Error searching videos: {e}")
                                
                    # Display previously found videos if they exist
                    video_key = f"video_results_{stable_key(job_prompt)}"
                    if video_key in st.session_state and st.session_state[video_key]:
                        st.subheader("📺 Previously Found Videos")
                        
                        video_results = st.session_state[video_key]
                        for skill, videos in video_results.items():
                            with st.expander(f"📺 Videos for {skill}"):
                                for i, video in enumerate(videos, 1):
                                    st.markdown(f"**{i}. {video['title']}**")
                                    st.markdown(f"- [Watch here]({video['url']})")
                                    st.markdown("---")
        else:
                    st.info("Please select at least one skill to get YouTube recommendations.")
            else:
                st.info("No valid skills found for this role.")

# Market Analysis Tab (new functionality)
with tab2:
    st.subheader("📊 LinkedIn Job Market Analysis")
    st.caption("AI-powered analysis of job market trends, skill demands, and hiring patterns")
    st.info("🤖 This analysis now uses LLM to read actual job descriptions for accurate skill extraction (not hardcoded keywords)")
    
    # Input section for market analysis
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analysis_job_title = st.text_input(
            "Job Title for Market Analysis", 
            placeholder="e.g., Python Developer, Data Scientist, Product Manager"
        )
    
    with col2:
        analysis_location = st.selectbox(
            "Location Filter",
            ["All Locations", "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Boston, MA", "Chicago, IL", "Remote"]
        )
    
    with col3:
        experience_level = st.selectbox(
            "Experience Level",
            ["All Levels", "Entry Level", "Associate", "Mid Level", "Senior", "Director", "Executive"],
            help="Filter jobs by required experience level"
        )
    
    # Analysis options
    st.write("**Analysis Options:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analyze_skills = st.checkbox("📈 Skill Demand Analysis", value=True)
    with col2:
        analyze_locations = st.checkbox("🗺️ Location Analysis", value=True)  
    with col3:
        analyze_companies = st.checkbox("🏢 Company Analysis", value=True)
    
    # Run analysis button
    run_analysis = st.button("🔍 Run Market Analysis", type="primary", disabled=not analysis_job_title.strip())
    
    # Analysis logic (only run when button pressed)
    if run_analysis and analysis_job_title.strip():
        analysis_cache_key = f"linkedin_analysis_{stable_key(analysis_job_title)}_{stable_key(analysis_location)}_{stable_key(experience_level)}"
        
        # Check if analysis is cached
        if analysis_cache_key in st.session_state:
            st.info("📋 Using cached analysis results")
            job_data = st.session_state[analysis_cache_key]
        else:
            with st.spinner("🤖 AI is analyzing job descriptions... This may take a few minutes."):
                try:
                    analyzer = LinkedInJobAnalyzer()
                    location_filter = analysis_location if analysis_location != "All Locations" else ""
                    experience_filter = experience_level if experience_level != "All Levels" else ""
                    
                    # Use intelligent LLM-powered analysis for accurate results  
                    job_data = analyzer.get_intelligent_job_analysis(
                        job_title=analysis_job_title,
                        locations=[location_filter] if location_filter else None,
                        experience_level=experience_filter,
                        max_results=30  # Limit for LLM processing
                    )
                    
                    # Cache the results
                    st.session_state[analysis_cache_key] = job_data
                    logger.info(f"✅ LinkedIn analysis completed for '{analysis_job_title}' ({experience_level})")
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing job market: {str(e)}")
                    st.info("💡 Note: LinkedIn analysis requires web scraping which may be limited by rate limits or access restrictions.")
                    job_data = None
    
    # Display analysis results (always show if we have cached data for current inputs)
    if analysis_job_title.strip():
        analysis_cache_key = f"linkedin_analysis_{stable_key(analysis_job_title)}_{stable_key(analysis_location)}_{stable_key(experience_level)}"
        
        if analysis_cache_key in st.session_state:
            job_data = st.session_state[analysis_cache_key]
            
            if job_data and job_data.get('total_jobs', 0) > 0:
                # Generate comprehensive statistics
                stats_generator = JobStatsGenerator()
                comprehensive_stats = stats_generator.generate_comprehensive_stats(job_data)
                
                # Create visualizations
                visualizer = SkillVisualization()
                
                # Display overview metrics
                st.subheader("📊 Market Overview")
                if experience_level != "All Levels":
                    st.info(f"🎯 Showing results filtered for: **{experience_level}** positions")
                overview = comprehensive_stats.get('overview', {})
                visualizer.create_metrics_cards(overview)
                
                # Market insights
                if 'market_insights' in comprehensive_stats:
                    st.subheader("💡 Market Insights")
                    for insight in comprehensive_stats['market_insights']:
                        st.info(insight)
                
                # Recommendations
                if 'recommendations' in comprehensive_stats:
                    st.subheader("🎯 Recommendations")
                    for recommendation in comprehensive_stats['recommendations']:
                        st.success(recommendation)
                
                # Create and display charts based on selected options
                if analyze_skills and 'skills_analysis' in comprehensive_stats:
                    skills_data = comprehensive_stats['skills_analysis'].get('top_skills', {})
                    if skills_data:
                        st.subheader("📈 Skills Demand Analysis")
                        
                        # Main skills pie chart
                        skills_fig = visualizer.create_skills_pie_chart(
                            skills_data, 
                            f"Most Demanded Skills - {analysis_job_title}"
                        )
                        st.plotly_chart(skills_fig, use_container_width=True)
                        
                        # Technology breakdown
                        technology_breakdown = comprehensive_stats['skills_analysis'].get('technology_breakdown', {})
                        if technology_breakdown:
                            st.subheader("🔧 Technology Stack Breakdown")
                            
                            # Display top technologies by category
                            cols = st.columns(2)
                            col_index = 0
                            
                            for category, data in technology_breakdown.items():
                                if data['skills']:
                                    with cols[col_index % 2]:
                                        st.write(f"**{category}**")
                                        top_skills = list(data['skills'].items())[:5]
                                        for skill, count in top_skills:
                                            percentage = round((count / data['total_mentions']) * 100, 1)
                                            st.write(f"• {skill}: {count} jobs ({percentage}%)")
                                        st.write("")
                                    col_index += 1
                        
                        # AI-specific insights for AI/ML roles
                        ai_insights = comprehensive_stats['skills_analysis'].get('ai_specific_insights', {})
                        if ai_insights and any(ai_insights.values()):
                            st.subheader("🤖 AI/ML Technology Insights")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if ai_insights.get('rag_adoption', 0) > 0:
                                    st.metric("RAG Technology Adoption", f"{ai_insights['rag_adoption']} mentions")
                                
                                if ai_insights.get('llm_frameworks'):
                                    st.write("**🔗 LLM Frameworks in Demand:**")
                                    for framework in ai_insights['llm_frameworks'][:5]:
                                        st.write(f"• {framework}")
                            
                            with col2:
                                if ai_insights.get('vector_tech_usage'):
                                    st.write("**🔍 Vector Technologies:**")
                                    for tech in ai_insights['vector_tech_usage'][:5]:
                                        st.write(f"• {tech}")
                                
                                if ai_insights.get('ai_cloud_preference'):
                                    st.write("**☁️ AI Cloud Services:**")
                                    for service in ai_insights['ai_cloud_preference'][:3]:
                                        st.write(f"• {service}")
                        
                        # Skill categories chart
                        skill_categories = comprehensive_stats['skills_analysis'].get('skill_categories', {})
                        if skill_categories:
                            categories_fig = visualizer.create_skill_categories_chart(
                                skill_categories,
                                "Technology Categories Distribution"
                            )
                            st.plotly_chart(categories_fig, use_container_width=True)
                
                if analyze_locations and 'location_analysis' in comprehensive_stats:
                    location_data = comprehensive_stats['location_analysis'].get('top_locations', {})
                    if location_data:
                        st.subheader("🗺️ Geographic Distribution")
                        location_fig = visualizer.create_location_bar_chart(
                            location_data,
                            "Job Distribution by Location"
                        )
                        st.plotly_chart(location_fig, use_container_width=True)
                
                if analyze_companies and 'company_analysis' in comprehensive_stats:
                    company_data = comprehensive_stats['company_analysis'].get('top_employers', {})
                    if company_data:
                        st.subheader("🏢 Top Hiring Companies")
                        company_fig = visualizer.create_companies_chart(
                            company_data,
                            "Companies Actively Hiring"
                        )
                        st.plotly_chart(company_fig, use_container_width=True)
                        
                        # Company details section (simplified)
                        st.subheader("💼 Company Explorer")
                        selected_company = st.selectbox(
                            "Explore a company:",
                            options=["Choose a company..."] + list(company_data.keys())[:10],
                            key="company_explorer"
                        )
                        
                        if selected_company and selected_company != "Choose a company...":
                            st.info(f"💡 **{selected_company}** is actively hiring for {analysis_job_title} roles!")
                            st.write("**Next Steps:**")
                            st.write(f"• 🔍 Search LinkedIn for '{selected_company} {analysis_job_title}' jobs")
                            st.write(f"• 🤝 Find contacts at {selected_company} for referrals") 
                            st.write(f"• 📝 Research {selected_company}'s engineering culture and values")
                            
                            # LinkedIn search links
                            company_search_url = f"https://www.linkedin.com/jobs/search/?keywords={analysis_job_title.replace(' ', '%20')}&location=&geoId=&f_C={selected_company.replace(' ', '%20')}"
                            st.markdown(f"🔗 [Search {selected_company} jobs on LinkedIn]({company_search_url})")
                            
                            people_search_url = f"https://www.linkedin.com/search/results/people/?currentCompany=%5B%22{selected_company.replace(' ', '%20')}%22%5D&keywords=recruiter%20OR%20hiring%20manager"
                            st.markdown(f"🔗 [Find {selected_company} recruiters on LinkedIn]({people_search_url})")
                
                # Market-Based Study Plan (simplified)
                st.subheader("📚 Smart Study Plan")
                st.caption("Get learning recommendations based on market demand")
                
                if st.button("🎯 Generate Study Recommendations", key="study_recommendations"):
                    try:
                        planner = MarketBasedStudyPlanner()
                        study_plan = planner.generate_study_plan(comprehensive_stats, target_weeks=8)
                        
                        if 'learning_path' in study_plan and study_plan['learning_path']:
                            st.write("**🎯 Top 5 Skills to Learn (in priority order):**")
                            
                            for i, item in enumerate(study_plan['learning_path'][:5], 1):
                                with st.expander(f"{i}. {item['skill']} ({item['difficulty']}) - {item['market_demand']} job mentions"):
                                    st.write(f"**Learning Time:** {item['duration_weeks']} weeks")
                                    st.write("**Why Learn This:**")
                                    for objective in item['learning_objectives'][:3]:
                                        st.write(f"• {objective}")
                                    
                                    # YouTube search link
                                    skill_search = f"https://www.youtube.com/results?search_query={item['skill'].replace(' ', '+').replace('/', '%2F')}+tutorial"
                                    st.markdown(f"🎥 [Search YouTube for {item['skill']} tutorials]({skill_search})")
                        
                        if 'market_insights' in study_plan:
                            st.write("**💡 Study Plan Insights:**")
                            for insight in study_plan['market_insights'][:3]:
                                st.info(insight)
                                
                    except Exception as e:
                        st.error(f"Error generating study plan: {str(e)}")
                
                # Raw data section (collapsible)
                with st.expander("📋 View Raw Analysis Data"):
                    st.json(comprehensive_stats)
                    
            elif job_data:
                st.warning(f"⚠️ No job postings found for '{analysis_job_title}' in the specified location.")
                st.info("💡 Try adjusting your search terms or location filter.")
    
    elif run_analysis:
        st.error("Please enter a job title to analyze.")

# AI Career Coach Tab
with tab3:
    st.subheader("🤖 AI Career Coach")
    st.caption("Your intelligent career advisor that learns and adapts")
    st.info("🚧 AI Career Coach is ready! Full interface coming soon...")
    st.write("**Features:**")
    st.write("• Personalized career recommendations")
    st.write("• Learning path generation")
    st.write("• Market intelligence integration")
    st.write("• Continuous learning from feedback")
