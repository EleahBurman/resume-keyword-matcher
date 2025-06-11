"""
Keyword Matcher Module for Resume Keyword Matcher

This module compares keywords from resumes and job descriptions
to calculate compatibility scores and identify matches.

Author: Resume Keyword Matcher
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher


class KeywordMatcher:
    """
    Handles keyword matching and scoring between resumes and job descriptions
    """
    
    def __init__(self):
        """Initialize the KeywordMatcher"""
        self.similarity_threshold = 0.8  # Minimum similarity for fuzzy matching
        
        # Weight multipliers for different match types
        self.match_weights = {
            'exact_match': 2.0,      # Exact keyword matches
            'fuzzy_match': 1.5,      # Similar keywords (e.g., "develop" vs "developer")
            'skill_match': 2.5,      # Technical skill matches
            'phrase_match': 1.8      # Multi-word phrase matches
        }
    
    def match_keywords(self, resume_keywords, job_keywords):
        """
        Compare resume keywords with job description keywords
        
        Args:
            resume_keywords (list): Keywords extracted from resume
            job_keywords (list): Keywords extracted from job description
            
        Returns:
            dict: Comprehensive matching results with scores and details
        """
        if not resume_keywords or not job_keywords:
            return self._empty_result()
        
        # Convert keyword lists to more usable format
        resume_kw_dict = self._keywords_to_dict(resume_keywords)
        job_kw_dict = self._keywords_to_dict(job_keywords)
        
        # Find different types of matches
        exact_matches = self._find_exact_matches(resume_kw_dict, job_kw_dict)
        fuzzy_matches = self._find_fuzzy_matches(resume_kw_dict, job_kw_dict, exact_matches)
        skill_matches = self._find_skill_matches(resume_kw_dict, job_kw_dict)
        
        # Calculate overall compatibility score
        compatibility_score = self._calculate_compatibility_score(
            resume_kw_dict, job_kw_dict, exact_matches, fuzzy_matches, skill_matches
        )
        
        # Generate detailed analysis
        analysis = self._generate_analysis(
            resume_kw_dict, job_kw_dict, exact_matches, fuzzy_matches, skill_matches
        )
        
        # Compile comprehensive results
        results = {
            'compatibility_score': round(compatibility_score, 1),
            'total_resume_keywords': len(resume_kw_dict),
            'total_job_keywords': len(job_kw_dict),
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches,
            'skill_matches': skill_matches,
            'match_summary': {
                'exact_match_count': len(exact_matches),
                'fuzzy_match_count': len(fuzzy_matches),
                'skill_match_count': len(skill_matches),
                'total_matches': len(exact_matches) + len(fuzzy_matches) + len(skill_matches)
            },
            'analysis': analysis,
            'recommendations': self._generate_recommendations(
                resume_kw_dict, job_kw_dict, exact_matches, fuzzy_matches
            )
        }
        
        return results
    
    def _keywords_to_dict(self, keyword_list):
        """Convert keyword list to dictionary for easier processing"""
        keyword_dict = {}
        for kw in keyword_list:
            if isinstance(kw, dict) and 'keyword' in kw:
                keyword_dict[kw['keyword']] = kw
            elif isinstance(kw, str):
                keyword_dict[kw] = {'keyword': kw, 'score': 1.0, 'count': 1}
        return keyword_dict
    
    def _find_exact_matches(self, resume_kw, job_kw):
        """Find exact keyword matches between resume and job description"""
        exact_matches = []
        
        for resume_word, resume_data in resume_kw.items():
            if resume_word in job_kw:
                job_data = job_kw[resume_word]
                match = {
                    'keyword': resume_word,
                    'resume_score': resume_data.get('score', 1.0),
                    'resume_count': resume_data.get('count', 1),
                    'job_score': job_data.get('score', 1.0),
                    'job_count': job_data.get('count', 1),
                    'match_strength': min(resume_data.get('score', 1.0), job_data.get('score', 1.0))
                }
                exact_matches.append(match)
        
        # Sort by match strength
        exact_matches.sort(key=lambda x: x['match_strength'], reverse=True)
        return exact_matches
    
    def _find_fuzzy_matches(self, resume_kw, job_kw, exact_matches):
        """Find similar keywords using fuzzy string matching"""
        fuzzy_matches = []
        exact_match_words = {match['keyword'] for match in exact_matches}
        
        for resume_word, resume_data in resume_kw.items():
            if resume_word in exact_match_words:
                continue  # Skip words that already have exact matches
            
            best_match = None
            best_similarity = 0
            
            for job_word, job_data in job_kw.items():
                if job_word in exact_match_words:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(resume_word, job_word)
                
                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'resume_keyword': resume_word,
                        'job_keyword': job_word,
                        'similarity': similarity,
                        'resume_score': resume_data.get('score', 1.0),
                        'job_score': job_data.get('score', 1.0),
                        'match_strength': similarity * min(resume_data.get('score', 1.0), job_data.get('score', 1.0))
                    }
            
            if best_match:
                fuzzy_matches.append(best_match)
        
        # Sort by match strength
        fuzzy_matches.sort(key=lambda x: x['match_strength'], reverse=True)
        return fuzzy_matches
    
    def _find_skill_matches(self, resume_kw, job_kw):
        """Find technical skill matches with special handling"""
        technical_skills = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'django',
            'flask', 'sql', 'postgresql', 'mysql', 'mongodb', 'aws', 'azure',
            'docker', 'kubernetes', 'git', 'jenkins', 'tensorflow', 'pytorch',
            'machine learning', 'data science', 'api', 'rest', 'graphql'
        }
        
        skill_matches = []
        
        for skill in technical_skills:
            resume_has_skill = skill in resume_kw
            job_wants_skill = skill in job_kw
            
            if resume_has_skill and job_wants_skill:
                match = {
                    'skill': skill,
                    'resume_score': resume_kw[skill].get('score', 1.0),
                    'job_score': job_kw[skill].get('score', 1.0),
                    'importance': 'high',
                    'match_strength': (resume_kw[skill].get('score', 1.0) + job_kw[skill].get('score', 1.0)) / 2
                }
                skill_matches.append(match)
        
        # Sort by match strength
        skill_matches.sort(key=lambda x: x['match_strength'], reverse=True)
        return skill_matches
    
    def _calculate_similarity(self, word1, word2):
        """Calculate similarity between two words"""
        return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
    
    def _calculate_compatibility_score(self, resume_kw, job_kw, exact_matches, fuzzy_matches, skill_matches):
        """Calculate overall compatibility score (0-100)"""
        if not job_kw:
            return 0.0
        
        total_score = 0
        max_possible_score = 0
        
        # Score exact matches
        for match in exact_matches:
            total_score += match['match_strength'] * self.match_weights['exact_match']
        
        # Score fuzzy matches
        for match in fuzzy_matches:
            total_score += match['match_strength'] * self.match_weights['fuzzy_match']
        
        # Score skill matches (heavily weighted)
        for match in skill_matches:
            total_score += match['match_strength'] * self.match_weights['skill_match']
        
        # Calculate maximum possible score based on job requirements
        for job_word, job_data in job_kw.items():
            max_possible_score += job_data.get('score', 1.0) * self.match_weights['exact_match']
        
        # Calculate percentage score
        if max_possible_score > 0:
            compatibility_percentage = min((total_score / max_possible_score) * 100, 100)
        else:
            compatibility_percentage = 0
        
        return compatibility_percentage
    
    def _generate_analysis(self, resume_kw, job_kw, exact_matches, fuzzy_matches, skill_matches):
        """Generate detailed analysis of the matching results"""
        analysis = {
            'strengths': [],
            'gaps': [],
            'technical_skills_coverage': 0,
            'keyword_coverage': 0
        }
        
        # Identify strengths
        if exact_matches:
            top_matches = [match['keyword'] for match in exact_matches[:5]]
            analysis['strengths'].append(f"Strong keyword matches: {', '.join(top_matches)}")
        
        if skill_matches:
            technical_skills = [match['skill'] for match in skill_matches[:5]]
            analysis['strengths'].append(f"Technical skills alignment: {', '.join(technical_skills)}")
        
        # Identify gaps
        job_keywords_set = set(job_kw.keys())
        resume_keywords_set = set(resume_kw.keys())
        matched_keywords = {match['keyword'] for match in exact_matches}
        
        missing_keywords = job_keywords_set - resume_keywords_set - matched_keywords
        if missing_keywords:
            # Get top missing keywords based on their importance in job description
            important_missing = sorted(
                missing_keywords, 
                key=lambda x: job_kw[x].get('score', 0), 
                reverse=True
            )[:5]
            analysis['gaps'] = important_missing
        
        # Calculate coverage metrics
        total_job_keywords = len(job_kw)
        matched_job_keywords = len(exact_matches) + len(fuzzy_matches)
        analysis['keyword_coverage'] = round((matched_job_keywords / total_job_keywords) * 100, 1) if total_job_keywords > 0 else 0
        
        # Technical skills coverage
        job_technical_skills = [kw for kw in job_kw.keys() if self._is_technical_skill(kw)]
        matched_technical_skills = len(skill_matches)
        if job_technical_skills:
            analysis['technical_skills_coverage'] = round((matched_technical_skills / len(job_technical_skills)) * 100, 1)
        
        return analysis
    
    def _generate_recommendations(self, resume_kw, job_kw, exact_matches, fuzzy_matches):
        """Generate recommendations for improving resume match"""
        recommendations = []
        
        # Find most important missing keywords
        job_keywords_set = set(job_kw.keys())
        resume_keywords_set = set(resume_kw.keys())
        matched_keywords = {match['keyword'] for match in exact_matches}
        
        missing_important = []
        for job_word, job_data in job_kw.items():
            if job_word not in resume_keywords_set and job_word not in matched_keywords:
                if job_data.get('score', 0) > 2.0:  # Important keywords
                    missing_important.append(job_word)
        
        if missing_important[:3]:  # Top 3 missing
            recommendations.append({
                'type': 'add_keywords',
                'priority': 'high',
                'suggestion': f"Consider adding these important keywords: {', '.join(missing_important[:3])}"
            })
        
        # Skill-specific recommendations
        technical_gaps = []
        technical_skills = ['python', 'java', 'react', 'sql', 'aws', 'docker']
        for skill in technical_skills:
            if skill in job_kw and skill not in resume_kw:
                technical_gaps.append(skill)
        
        if technical_gaps:
            recommendations.append({
                'type': 'technical_skills',
                'priority': 'high',
                'suggestion': f"Highlight experience with: {', '.join(technical_gaps[:3])}"
            })
        
        return recommendations
    
    def _is_technical_skill(self, keyword):
        """Check if keyword is a technical skill"""
        technical_indicators = [
            'python', 'java', 'javascript', 'react', 'angular', 'sql', 'aws',
            'docker', 'kubernetes', 'api', 'database', 'framework', 'library'
        ]
        return keyword.lower() in technical_indicators
    
    def _empty_result(self):
        """Return empty result structure"""
        return {
            'compatibility_score': 0.0,
            'total_resume_keywords': 0,
            'total_job_keywords': 0,
            'exact_matches': [],
            'fuzzy_matches': [],
            'skill_matches': [],
            'match_summary': {
                'exact_match_count': 0,
                'fuzzy_match_count': 0,
                'skill_match_count': 0,
                'total_matches': 0
            },
            'analysis': {
                'strengths': [],
                'gaps': [],
                'technical_skills_coverage': 0,
                'keyword_coverage': 0
            },
            'recommendations': []
        }


# Test function
def test_keyword_matcher():
    """Test the KeywordMatcher functionality"""
    print("Testing KeywordMatcher...")
    
    matcher = KeywordMatcher()
    
    # Sample resume keywords (format from TextProcessor)
    resume_keywords = [
        {'keyword': 'python', 'score': 4.0, 'count': 3},
        {'keyword': 'django', 'score': 3.0, 'count': 2},
        {'keyword': 'database', 'score': 2.5, 'count': 2},
        {'keyword': 'development', 'score': 2.0, 'count': 1},
        {'keyword': 'api', 'score': 3.5, 'count': 2}
    ]
    
    # Sample job description keywords
    job_keywords = [
        {'keyword': 'python', 'score': 5.0, 'count': 4},
        {'keyword': 'flask', 'score': 3.0, 'count': 2},
        {'keyword': 'database', 'score': 3.0, 'count': 3},
        {'keyword': 'developer', 'score': 2.0, 'count': 1},
        {'keyword': 'rest', 'score': 2.5, 'count': 1},
        {'keyword': 'sql', 'score': 4.0, 'count': 2}
    ]
    
    try:
        # Test matching
        results = matcher.match_keywords(resume_keywords, job_keywords)
        
        print(f"\n📊 Compatibility Score: {results['compatibility_score']}%")
        print(f"Resume Keywords: {results['total_resume_keywords']}")
        print(f"Job Keywords: {results['total_job_keywords']}")
        
        print(f"\n🎯 Exact Matches ({len(results['exact_matches'])}):")
        for match in results['exact_matches']:
            print(f"  • {match['keyword']} (strength: {match['match_strength']:.1f})")
        
        print(f"\n🔍 Fuzzy Matches ({len(results['fuzzy_matches'])}):")
        for match in results['fuzzy_matches']:
            print(f"  • {match['resume_keyword']} ≈ {match['job_keyword']} ({match['similarity']:.1f})")
        
        print(f"\n💻 Technical Skills ({len(results['skill_matches'])}):")
        for match in results['skill_matches']:
            print(f"  • {match['skill']} (strength: {match['match_strength']:.1f})")
        
        print(f"\n📈 Analysis:")
        analysis = results['analysis']
        print(f"  Keyword Coverage: {analysis['keyword_coverage']}%")
        print(f"  Technical Skills Coverage: {analysis['technical_skills_coverage']}%")
        
        if analysis['strengths']:
            print(f"  Strengths: {analysis['strengths']}")
        
        if analysis['gaps']:
            print(f"  Missing Keywords: {', '.join(analysis['gaps'][:3])}")
        
        print("\n✅ KeywordMatcher test passed!")
        
    except Exception as e:
        print(f"❌ KeywordMatcher test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_keyword_matcher()