"""
Skill Visualization Module
Creates charts and graphs for LinkedIn job market analysis.
"""

import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple
import streamlit as st

logger = logging.getLogger(__name__)

class SkillVisualization:
    """Creates interactive visualizations for job market data."""
    
    def __init__(self):
        self.color_schemes = {
            'skills': px.colors.qualitative.Set3,
            'locations': px.colors.qualitative.Pastel,
            'companies': px.colors.qualitative.Dark24
        }
    
    def create_skills_pie_chart(self, skills_data: Dict[str, int], title: str = "Skills Distribution") -> go.Figure:
        """
        Create an interactive pie chart showing skill distribution.
        
        Args:
            skills_data: Dictionary of skill names and their counts
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        logger.info(f"📊 Creating skills pie chart with {len(skills_data)} skills")
        
        if not skills_data:
            return self._create_empty_chart("No skills data available")
        
        # Prepare data - limit to top 15 skills for better readability
        top_skills = dict(sorted(skills_data.items(), key=lambda x: x[1], reverse=True)[:15])
        
        # Group smaller skills into "Others" category
        total_count = sum(skills_data.values())
        top_count = sum(top_skills.values())
        others_count = total_count - top_count
        
        if others_count > 0:
            top_skills['Others'] = others_count
        
        skills = list(top_skills.keys())
        counts = list(top_skills.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=skills,
            values=counts,
            hole=0.3,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>' +
                         'Jobs: %{value}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>',
            marker=dict(
                colors=self.color_schemes['skills'][:len(skills)],
                line=dict(color='#FFFFFF', width=2)
            )
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2E4057'}
            },
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            margin=dict(l=20, r=150, t=70, b=20),
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def create_location_bar_chart(self, location_data: Dict[str, int], title: str = "Jobs by Location") -> go.Figure:
        """
        Create a horizontal bar chart showing job distribution by location.
        
        Args:
            location_data: Dictionary of locations and job counts
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        logger.info(f"📊 Creating location bar chart with {len(location_data)} locations")
        
        if not location_data:
            return self._create_empty_chart("No location data available")
        
        # Sort by job count
        sorted_locations = dict(sorted(location_data.items(), key=lambda x: x[1], reverse=True))
        
        locations = list(sorted_locations.keys())
        counts = list(sorted_locations.values())
        
        fig = go.Figure([go.Bar(
            x=counts,
            y=locations,
            orientation='h',
            marker=dict(
                color=counts,
                colorscale='Blues',
                colorbar=dict(title="Job Count")
            ),
            text=counts,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Jobs: %{x}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2E4057'}
            },
            xaxis_title="Number of Jobs",
            yaxis_title="Location",
            height=max(400, len(locations) * 30),
            margin=dict(l=150, r=50, t=70, b=50),
            yaxis=dict(categoryorder='total ascending')
        )
        
        return fig
    
    def create_companies_chart(self, company_data: Dict[str, int], title: str = "Top Hiring Companies") -> go.Figure:
        """
        Create a bar chart showing top hiring companies.
        
        Args:
            company_data: Dictionary of companies and job counts
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        logger.info(f"📊 Creating companies chart with {len(company_data)} companies")
        
        if not company_data:
            return self._create_empty_chart("No company data available")
        
        # Take top 15 companies
        top_companies = dict(sorted(company_data.items(), key=lambda x: x[1], reverse=True)[:15])
        
        companies = list(top_companies.keys())
        counts = list(top_companies.values())
        
        fig = go.Figure([go.Bar(
            x=companies,
            y=counts,
            marker=dict(
                color=counts,
                colorscale='Viridis',
                colorbar=dict(title="Job Count")
            ),
            text=counts,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'Jobs: %{y}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2E4057'}
            },
            xaxis_title="Company",
            yaxis_title="Number of Jobs",
            height=500,
            margin=dict(l=50, r=50, t=70, b=150),
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_skill_categories_chart(self, skill_categories: Dict[str, int], title: str = "Skills by Category") -> go.Figure:
        """
        Create a donut chart showing skill distribution by category.
        
        Args:
            skill_categories: Dictionary of skill categories and counts
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        logger.info(f"📊 Creating skill categories chart with {len(skill_categories)} categories")
        
        if not skill_categories:
            return self._create_empty_chart("No skill category data available")
        
        categories = list(skill_categories.keys())
        counts = list(skill_categories.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=counts,
            hole=0.4,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>' +
                         'Skills: %{value}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>',
            marker=dict(
                colors=self.color_schemes['skills'][:len(categories)],
                line=dict(color='#FFFFFF', width=2)
            )
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2E4057'}
            },
            showlegend=True,
            annotations=[dict(text='Skill<br>Categories', x=0.5, y=0.5, font_size=16, showarrow=False)],
            height=500,
            margin=dict(l=20, r=20, t=70, b=20)
        )
        
        return fig
    
    def create_market_overview_dashboard(self, stats_data: Dict) -> List[go.Figure]:
        """
        Create a comprehensive dashboard with multiple charts.
        
        Args:
            stats_data: Comprehensive statistics data
            
        Returns:
            List of Plotly figure objects
        """
        logger.info("📊 Creating comprehensive market overview dashboard")
        
        figures = []
        
        # Skills analysis charts
        if 'skills_analysis' in stats_data and stats_data['skills_analysis'].get('top_skills'):
            skills_fig = self.create_skills_pie_chart(
                stats_data['skills_analysis']['top_skills'], 
                f"Most Demanded Skills - {stats_data.get('overview', {}).get('job_title', 'Unknown Role')}"
            )
            figures.append(skills_fig)
        
        # Location analysis chart
        if 'location_analysis' in stats_data and stats_data['location_analysis'].get('top_locations'):
            location_fig = self.create_location_bar_chart(
                stats_data['location_analysis']['top_locations'],
                "Job Distribution by Location"
            )
            figures.append(location_fig)
        
        # Company analysis chart
        if 'company_analysis' in stats_data and stats_data['company_analysis'].get('top_employers'):
            company_fig = self.create_companies_chart(
                stats_data['company_analysis']['top_employers'],
                "Top Hiring Companies"
            )
            figures.append(company_fig)
        
        # Skill categories chart
        if 'skills_analysis' in stats_data and stats_data['skills_analysis'].get('skill_categories'):
            categories_fig = self.create_skill_categories_chart(
                stats_data['skills_analysis']['skill_categories'],
                "Skills Distribution by Category"
            )
            figures.append(categories_fig)
        
        logger.info(f"✅ Created {len(figures)} dashboard charts")
        return figures
    
    def create_comparison_chart(self, comparison_data: Dict, metric: str = "total_jobs") -> go.Figure:
        """
        Create a comparison chart between multiple job titles.
        
        Args:
            comparison_data: Job comparison data
            metric: Metric to compare ('total_jobs', 'skills', etc.)
            
        Returns:
            Plotly figure object
        """
        logger.info(f"📊 Creating comparison chart for metric: {metric}")
        
        job_comparison = comparison_data.get('job_comparison', {})
        
        if not job_comparison:
            return self._create_empty_chart("No comparison data available")
        
        if metric == "total_jobs":
            job_titles = list(job_comparison.keys())
            job_counts = [data.get('total_jobs', 0) for data in job_comparison.values()]
            
            fig = go.Figure([go.Bar(
                x=job_titles,
                y=job_counts,
                marker=dict(
                    color=job_counts,
                    colorscale='Blues',
                    colorbar=dict(title="Job Count")
                ),
                text=job_counts,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                             'Jobs: %{y}<br>' +
                             '<extra></extra>'
            )])
            
            fig.update_layout(
                title={
                    'text': "Job Market Comparison",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2E4057'}
                },
                xaxis_title="Job Title",
                yaxis_title="Number of Jobs",
                height=500,
                margin=dict(l=50, r=50, t=70, b=100),
                xaxis=dict(tickangle=45)
            )
            
            return fig
        
        return self._create_empty_chart(f"Comparison for {metric} not implemented")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color='gray')
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
    
    def display_charts_in_streamlit(self, figures: List[go.Figure]):
        """
        Display multiple charts in Streamlit with proper layout.
        
        Args:
            figures: List of Plotly figures to display
        """
        logger.info(f"📊 Displaying {len(figures)} charts in Streamlit")
        
        for i, fig in enumerate(figures):
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
    
    def create_metrics_cards(self, overview_stats: Dict) -> None:
        """
        Create metric cards for key statistics.
        
        Args:
            overview_stats: Overview statistics dictionary
        """
        logger.info("📊 Creating metrics cards")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Jobs", 
                value=overview_stats.get('total_jobs', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Recent Postings", 
                value=overview_stats.get('recent_postings', 0),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Locations", 
                value=overview_stats.get('locations_covered', 0),
                delta=None
            )
        
        with col4:
            st.metric(
                label="Skills Found", 
                value=overview_stats.get('unique_skills_found', 0),
                delta=None
            )
