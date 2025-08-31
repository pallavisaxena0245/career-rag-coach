# �� Career RAG Coach - AI-Powered Career Guidance System

> **Your intelligent career advisor that learns, adapts, and provides personalized guidance using real job market data**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)](https://openai.com/)

## ✨ **What is Career RAG Coach?**

Career RAG Coach is a **next-generation AI career guidance system** that combines:

- 🤖 **LLM-Powered Skill Extraction** - Reads actual job descriptions (no hardcoded keywords!)
- 🧠 **Learning Agent System** - Adapts and improves from user feedback
- 📊 **Real Market Intelligence** - LinkedIn job analysis with AI insights
- 🎥 **Smart Video Recommendations** - YouTube tutorials based on market demand
- 🛤️ **Personalized Learning Paths** - AI-generated study plans

## 🎯 **Key Features**

### **1. 🎯 Skills & Videos Tab**
- **AI-Powered Skill Extraction**: Uses GPT to understand job context and extract relevant skills
- **YouTube Integration**: Finds top-rated tutorials for selected skills
- **LangGraph Workflow**: Robust state management and error handling

### **2. 📊 Market Analysis Tab** 
- **LLM-Powered LinkedIn Analysis**: Reads actual job descriptions for accurate insights
- **Experience Level Filtering**: Entry, Associate, Mid, Senior, Director, Executive
- **Interactive Visualizations**: Pie charts, bar charts, company analysis
- **Real-Time Data**: No hardcoded values - everything is AI-analyzed

### **3. 🤖 AI Career Coach Tab**
- **Learning Agent**: Remembers interactions and improves over time
- **Personalized Recommendations**: Tailored to your background and goals
- **Market-Based Study Plans**: Learning paths based on real job demand
- **Feedback Loop**: Agent learns from your ratings and suggestions

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   LangGraph      │    │   LLM Agents    │
│                 │    │   Workflow       │    │                 │
│ • Skills Tab    │◄──►│ • State Mgmt     │◄──►│ • Skill Extractor│
│ • Market Tab    │    │ • Node Pipeline  │    │ • Career Coach  │
│ • AI Coach Tab  │    │ • Error Handling │    │ • Market Analyzer│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.10+
- OpenAI API Key
- Git

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/career-rag-coach.git
   cd career-rag-coach
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 🎨 **Usage Examples**

### **Example 1: Extract Skills from Job Description**
```
Input: "Senior AI Engineer with focus on RAG systems and LLM applications"
Output: 
✅ Python, LangChain, Vector Databases, Pinecone, RAG, 
   Prompt Engineering, Docker, AWS, MLOps
```

### **Example 2: Market Analysis for UI Developer**
```
Input: "UI Developer" 
Output:
🎨 Top Skills: Figma (46 mentions), Sketch (46), 
   Adobe Creative Suite (45), HTML (43), CSS (43)
📊 Market Strength: Strong (50+ active positions)
```

### **Example 3: AI Career Coach Guidance**
```
Input: "I'm a UI developer wanting to transition to AI engineering"
Output:
🔴 HIGH PRIORITY: Learn Python (2-3 months)
🟡 MEDIUM PRIORITY: Master ML frameworks (3-4 months)  
🛤️ Learning Path: 3-phase structured plan with market insights
```

## 🛠️ **Technical Stack**

### **Core Technologies**
- **Frontend**: Streamlit (Python web framework)
- **AI/LLM**: OpenAI GPT-4o-mini, LangChain
- **Workflow**: LangGraph (state management & agent orchestration)
- **Data Processing**: Pandas, BeautifulSoup4
- **Visualization**: Plotly, Streamlit components

### **Key Libraries**
```python
# AI & LLM
langchain-openai>=0.1.0
langgraph>=0.6.0

# Web Scraping & Data
beautifulsoup4>=4.12.0
requests>=2.31.0
pandas>=2.0.0

# Visualization
plotly>=5.17.0
streamlit>=1.28.0

# Utilities
python-dotenv>=1.0.0
yt-dlp>=2023.12.30
```

## 🏗️ **Project Structure**

```
career-rag-coach/
├── app/
│   ├── agents/                 # AI Learning Agents
│   │   ├── career_learning_agent.py
│   │   └── __init__.py
│   ├── extract/                # Skill Extraction
│   │   └── skills.py
│   ├── graph/                  # LangGraph Workflows
│   │   └── flow.py
│   ├── linkedin/               # Market Analysis
│   │   ├── job_analyzer.py
│   │   ├── enhanced_job_scraper.py
│   │   ├── intelligent_skill_extractor.py
│   │   ├── stats_generator.py
│   │   └── visualization.py
│   ├── study/                  # Study Planning
│   │   └── market_based_planner.py
│   └── video/                  # YouTube Integration
│       └── youtube_search.py
├── streamlit_app.py            # Main Application
├── requirements.txt            # Dependencies
├── .env.example               # Environment Variables Template
└── README.md                  # This File
```

## 🔧 **Configuration**

### **Environment Variables**
```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

### **Customization Options**
- **Model Selection**: Change OpenAI model in sidebar
- **Analysis Depth**: Adjust max_results for LinkedIn analysis
- **Cache Settings**: Modify caching behavior for performance
- **Rate Limiting**: Adjust delays for web scraping

## 🚀 **Advanced Features**

### **1. Intelligent Skill Extraction**
- **Context Understanding**: LLM reads full job descriptions
- **Technology Detection**: Identifies specific frameworks and tools
- **Skill Categorization**: Programming, AI/ML, Tools, Databases, etc.
- **Fallback Generation**: Creates realistic descriptions when scraping fails

### **2. Learning Agent System**
- **Memory Persistence**: Remembers user interactions across sessions
- **Feedback Processing**: Learns from user ratings and suggestions
- **Adaptive Recommendations**: Improves suggestions over time
- **Market Integration**: Uses real job data for guidance

### **3. Market Intelligence**
- **Real-Time Analysis**: Scrapes current LinkedIn job postings
- **Experience Filtering**: Entry, Associate, Mid, Senior, Director, Executive
- **Geographic Analysis**: Location-based job market insights
- **Company Intelligence**: Top employers and application strategies

## 📊 **Performance & Scalability**

### **Optimizations**
- **Intelligent Caching**: Results cached to reduce API calls
- **Batch Processing**: LLM analysis in batches for efficiency
- **Rate Limiting**: Respectful web scraping with delays
- **Error Handling**: Graceful fallbacks when services fail

### **Scalability Features**
- **Modular Architecture**: Easy to add new analysis types
- **Plugin System**: Extensible agent and tool system
- **State Management**: Robust session handling
- **Async Support**: Ready for concurrent processing

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

### **Development Setup**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### **Areas for Contribution**
- 🎨 **UI/UX Improvements**: Better Streamlit components
- 🤖 **Agent Enhancements**: New learning algorithms
- 📊 **Data Sources**: Additional job market data
- 🌐 **Internationalization**: Multi-language support
- 📱 **Mobile Optimization**: Responsive design improvements

## 🐛 **Troubleshooting**

### **Common Issues**

**1. "No jobs found" for valid roles**
- Check if LinkedIn is blocking requests
- Verify job title spelling
- Try different location filters

**2. OpenAI API errors**
- Verify API key in `.env` file
- Check API quota and billing
- Ensure model name is correct

**3. Scraping failures**
- LinkedIn may have updated their HTML structure
- Check network connectivity
- Verify user agent strings

### **Debug Mode**
Enable debug logging in the sidebar to see detailed information about the analysis process.

## 📈 **Roadmap**

### **Phase 1: Core Features** ✅
- [x] LLM-powered skill extraction
- [x] LinkedIn market analysis
- [x] YouTube video recommendations
- [x] Basic study planning

### **Phase 2: Intelligence** ✅
- [x] Learning agent system
- [x] Market-based study plans
- [x] Intelligent skill categorization
- [x] Feedback learning system

### **Phase 3: Advanced Features** 🚧
- [ ] Resume parsing and analysis
- [ ] Interview question generation
- [ ] Salary insights and negotiation
- [ ] Career path visualization
- [ ] Multi-language support

### **Phase 4: Enterprise** 📋
- [ ] Team collaboration features
- [ ] Advanced analytics dashboard
- [ ] API endpoints for integration
- [ ] White-label solutions

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **OpenAI** for GPT-4o-mini API access
- **LangChain** for the excellent LLM framework
- **Streamlit** for the amazing web app framework
- **LinkedIn** for job market data (used respectfully)
- **YouTube** for educational content integration

## 📞 **Support & Contact**

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/career-rag-coach/issues)
- **Discussions**: [Join the community](https://github.com/yourusername/career-rag-coach/discussions)
- **Email**: your.email@example.com

---

<div align="center">

**Made with ❤️ by [Your Name]**

*Empowering careers through AI-powered insights*

[⭐ Star this repo](https://github.com/yourusername/career-rag-coach) | [🚀 Deploy](https://github.com/yourusername/career-rag-coach#deployment) | [📖 Docs](https://github.com/yourusername/career-rag-coach#documentation)

</div>
