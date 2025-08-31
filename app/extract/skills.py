from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
logger.info("✅ LangChain OpenAI imported successfully")

from app.config import get_openai_api_key


def extract_skills_for_job(job_prompt: str, model_name: str = "gpt-4o-mini") -> List[str]:
    
    logger.info(f"🔍 Starting skills extraction for job: '{job_prompt}'")
    logger.info(f"📋 Using model: {model_name}")
    
    logger.info("✅ ChatOpenAI is available")
    
    api_key = get_openai_api_key()
    if not api_key:
        logger.error("❌ No OpenAI API key found")
        return []
    
    logger.info("✅ OpenAI API key found")
    
    llm = ChatOpenAI(model=model_name, temperature=0)
    logger.info(f"✅ LLM initialized with model: {model_name}")
    
    prompt = f"""Extract skills for this job: {job_prompt}

List the most important technical skills, soft skills, and tools needed for this role. Return only skill names, one per line, no bullets or numbers."""
    
    logger.info(f"📝 Sending prompt to LLM: {prompt[:100]}...")
    
    logger.info("🔄 Invoking LLM...")
    result = llm.invoke(prompt)
    logger.info("✅ LLM invocation successful")
    
    content = getattr(result, "content", "")
    logger.info(f"📄 Raw LLM response length: {len(content)} characters")
    logger.info(f"📄 Raw LLM response: {repr(content)}")
    
    if not content:
        logger.error("❌ LLM returned empty content")
        return []
    
    lines = [line.strip("- •\t ") for line in content.splitlines()]
    logger.info(f"📋 Parsed {len(lines)} lines from response")
    
    skills = [s for s in lines if len(s) > 1 and s.strip()]
    logger.info(f"🔧 Found {len(skills)} potential skills after filtering")
    logger.info(f"🔧 Skills found: {skills}")
    
    seen = set()
    deduped: List[str] = []
    for s in skills:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(s.strip())
    
    logger.info(f"🎯 Final deduplicated skills count: {len(deduped)}")
    logger.info(f"🎯 Final skills: {deduped}")
    
    return deduped[:50]

