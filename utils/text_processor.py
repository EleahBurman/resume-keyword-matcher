"""
Text Processor Module for Resume Keyword Matcher

This module handles text processing and keyword extraction using:
- NLTK for basic text processing
- spaCy for advanced NLP (when available)
- Custom keyword extraction for resume/job description matching

Author: Resume Keyword Matcher
"""

import re
import string
from collections import Counter

# Try to import NLP libraries, handle gracefully if not available
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    
    # Download required NLTK data (only if not already downloaded)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not installed. Using basic text processing.")

try:
    import spacy
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        print("Warning: spaCy English model not found. Run: python -m spacy download en_core_web_sm")
        
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not installed. Using NLTK for text processing.")


class TextProcessor:
    """
    Handles text processing and keyword extraction for resume/job description matching
    """
    
    def __init__(self):
        """Initialize the TextProcessor with available NLP tools"""
        self.use_spacy = SPACY_AVAILABLE
        self.use_nltk = NLTK_AVAILABLE
        
        # Initialize NLTK components if available
        if self.use_nltk:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except:
                self.use_nltk = False
                print("Warning: NLTK data not available. Using basic processing.")
        
        # Define custom stop words for resume/job processing
        self.custom_stop_words = {
            'experience', 'work', 'job', 'position', 'role', 'company', 'team',
            'responsible', 'responsibilities', 'duties', 'tasks', 'required',
            'preferred', 'qualification', 'qualifications', 'skills', 'ability',
            'knowledge', 'understanding', 'familiarity', 'proficient', 'expert',
            'years', 'year', 'month', 'months', 'time', 'full', 'part',
            'looking', 'seeking', 'candidate', 'applicant', 'resume', 'cv'
        }
        
        # Common technical skills and keywords to prioritize
        self.technical_keywords = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php',
            'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'nodejs',
            'express', 'django', 'flask', 'spring', 'laravel', 'rails',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'oracle', 'sqlite', 'database', 'nosql',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'github', 'gitlab', 'ci/cd', 'devops', 'terraform', 'ansible',
            
            # Data Science & AI
            'machine learning', 'deep learning', 'ai', 'tensorflow', 'pytorch',
            'pandas', 'numpy', 'scikit-learn', 'data science', 'analytics',
            
            # Other Technical
            'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum',
            'testing', 'junit', 'pytest', 'selenium', 'automation'
        }
    
    def extract_keywords(self, text, max_keywords=50):
        """
        Extract important keywords from text
        
        Args:
            text (str): Input text to process
            max_keywords (int): Maximum number of keywords to return
            
        Returns:
            list: List of important keywords with weights
        """
        if not text or not text.strip():
            return []
        
        # Clean the text first
        cleaned_text = self._clean_text(text)
        
        # Use the best available method for keyword extraction
        if self.use_spacy:
            keywords = self._extract_with_spacy(cleaned_text)
        elif self.use_nltk:
            keywords = self._extract_with_nltk(cleaned_text)
        else:
            keywords = self._extract_basic(cleaned_text)
        
        # Post-process and rank keywords
        ranked_keywords = self._rank_keywords(keywords, text)
        
        # Return top keywords
        return ranked_keywords[:max_keywords]
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and line breaks
        text = re.sub(r'\s+', ' ', text)
        
        # Remove some punctuation but keep important ones like . and -
        text = re.sub(r'[^\w\s\.\-\+\#]', ' ', text)
        
        return text.strip()
    
    def _extract_with_spacy(self, text):
        """Extract keywords using spaCy NLP"""
        doc = nlp(text)
        keywords = []
        
        # Extract entities, noun phrases, and important words
        for token in doc:
            # Skip stop words, punctuation, and spaces
            if (token.is_stop or token.is_punct or token.is_space or 
                len(token.text) < 2):
                continue
            
            # Prioritize nouns, adjectives, and proper nouns
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                keywords.append(token.lemma_.lower())
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'SKILL']:
                keywords.append(ent.text.lower())
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit phrase length
                keywords.append(chunk.text.lower())
        
        return keywords
    
    def _extract_with_nltk(self, text):
        """Extract keywords using NLTK"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and punctuation
        keywords = []
        for token in tokens:
            if (len(token) >= 2 and 
                token.lower() not in self.stop_words and
                token not in string.punctuation):
                
                # Lemmatize the token
                lemma = self.lemmatizer.lemmatize(token.lower())
                keywords.append(lemma)
        
        return keywords
    
    def _extract_basic(self, text):
        """Basic keyword extraction without external libraries"""
        # Simple word splitting and filtering
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        
        # Basic stop words
        basic_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        keywords = []
        for word in words:
            word_lower = word.lower()
            if (len(word_lower) >= 2 and 
                word_lower not in basic_stop_words):
                keywords.append(word_lower)
        
        return keywords
    
    def _rank_keywords(self, keywords, original_text):
        """Rank keywords by importance"""
        if not keywords:
            return []
        
        # Count frequency of each keyword
        keyword_counts = Counter(keywords)
        
        # Calculate scores based on multiple factors
        scored_keywords = []
        original_lower = original_text.lower()
        
        for keyword, count in keyword_counts.items():
            score = count  # Base score is frequency
            
            # Boost technical keywords
            if keyword in self.technical_keywords:
                score *= 2.0
            
            # Boost keywords that appear as complete words
            if re.search(r'\b' + re.escape(keyword) + r'\b', original_lower):
                score *= 1.5
            
            # Reduce score for very common words
            if keyword in self.custom_stop_words:
                score *= 0.5
            
            # Boost longer keywords (likely more specific)
            if len(keyword) > 6:
                score *= 1.2
            
            scored_keywords.append({
                'keyword': keyword,
                'count': count,
                'score': score
            })
        
        # Sort by score (descending)
        scored_keywords.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_keywords
    
    def extract_skills(self, text):
        """
        Extract technical skills specifically
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of identified technical skills
        """
        if not text:
            return []
        
        text_lower = text.lower()
        found_skills = []
        
        # Look for technical keywords
        for skill in self.technical_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        
        # Look for skill patterns (e.g., "3+ years of Python")
        skill_patterns = [
            r'(\d+\+?\s*years?\s+(?:of\s+)?(\w+))',
            r'(experience\s+(?:with\s+|in\s+)?(\w+))',
            r'(proficient\s+(?:in\s+|with\s+)?(\w+))',
            r'(expert\s+(?:in\s+|with\s+)?(\w+))'
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if len(match.groups()) > 1:
                    skill = match.group(2)
                    if len(skill) > 2:
                        found_skills.append(skill)
        
        return list(set(found_skills))  # Remove duplicates
    
    def get_text_stats(self, text):
        """Get basic statistics about the text"""
        if not text:
            return {}
        
        words = len(text.split())
        characters = len(text)
        sentences = len(re.findall(r'[.!?]+', text))
        
        return {
            'word_count': words,
            'character_count': characters,
            'sentence_count': max(sentences, 1),  # At least 1
            'avg_words_per_sentence': words / max(sentences, 1)
        }


# Test function
def test_text_processor():
    """Test the TextProcessor functionality"""
    print("Testing TextProcessor...")
    
    processor = TextProcessor()
    
    # Test text
    test_text = """
    Senior Python Developer with 5+ years of experience in web development.
    Proficient in Django, Flask, and FastAPI. Experience with PostgreSQL,
    Redis, and Docker. Strong background in machine learning and data science
    using pandas, numpy, and scikit-learn. Familiar with AWS cloud services
    and CI/CD pipelines using Jenkins and GitHub Actions.
    """
    
    try:
        # Test keyword extraction
        keywords = processor.extract_keywords(test_text)
        print(f"\nExtracted {len(keywords)} keywords:")
        for i, kw in enumerate(keywords[:10]):  # Show top 10
            print(f"  {i+1}. {kw['keyword']} (score: {kw['score']:.1f}, count: {kw['count']})")
        
        # Test skill extraction
        skills = processor.extract_skills(test_text)
        print(f"\nExtracted skills: {skills}")
        
        # Test text stats
        stats = processor.get_text_stats(test_text)
        print(f"\nText statistics: {stats}")
        
        print("\n✅ TextProcessor test passed!")
        
    except Exception as e:
        print(f"❌ TextProcessor test failed: {e}")


if __name__ == "__main__":
    test_text_processor()