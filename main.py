"""
Intelligent Recommendation System (Local RAG-like Pipeline)

 Multi-Query Retriever
- STAGE 1 (Extract): Extracts features (level, duration) AND a "cleaned"
                     tech-only string.
- STAGE 2 (Retrieve): Parses the tech-string into a LIST of queries
                     (e.g., ["Python", "SQL", "JavaScript"]).
                     Runs a separate semantic search for EACH query.
                     Runs a final search on the FULL query for context.
                     Combines all results.
- STAGE 3 (Rank):     Uses the V6 multi-signal ranker to score the
                     high-quality combined list.
"""
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import requests
from bs4 import BeautifulSoup

class IntelligentRecommender:
    def __init__(self, assessments_file='shl_individual_tests.json', embeddings_file='assessment_embeddings-mpnet.pkl'):
        print("Loading Local-Only Intelligent RAG System ( Multi-Query Retriever)...")
        with open(assessments_file, 'r', encoding='utf-8') as f:
            self.assessments = json.load(f)
        
        print(f"Loaded {len(self.assessments)} individual test assessments")
        
        print("Loading sentence transformer model (all-mpnet-base-v2)...")
        self.model = SentenceTransformer('all-mpnet-base-v2') 

        self.feature_keywords = {
            'soft': ['collaborat', 'team', 'teamwork', 'lead', 'manag', 'supervisor', 'personality', 'behaviour', 'communication', 'opq'],
            'level': ['entry', 'junior', 'graduate', 'intern', 'senior', 'lead', 'expert', 'principal'],
            'duration': ['minute', 'min', 'mins', 'less', 'maximum', 'max', 'under', 'within', 'duration', 'time']
        }
        
        try:
            print(f"Loading pre-computed embeddings ({embeddings_file})...")
            with open(embeddings_file, 'rb') as f:
                self.assessment_embeddings = pickle.load(f)
            print(f"Loaded embeddings with shape: {self.assessment_embeddings.shape}")
        except FileNotFoundError:
            print(f"Embeddings file not found. Creating new embeddings with '{self.model._get_name()}'...")
            self._create_embeddings(embeddings_file)
        
        print("✓ System ready. Running 100% locally.")

    def _create_embeddings(self, embeddings_file):
        """Create embeddings for all assessments"""
        print("Creating embeddings for assessments...")
        
        assessment_texts = []
        for assessment in self.assessments:
            text = f"{assessment['name']}. "
            text += f"{assessment.get('description', '')}. "
            text += f"Test types: {' '.join(assessment.get('test_type', []))}. "
            text += f"Duration: {assessment.get('duration', 30)} minutes. "
            text += f"Adaptive: {assessment.get('adaptive_support', 'No')}."
            assessment_texts.append(text)
        
        self.assessment_embeddings = self.model.encode(
            assessment_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.assessment_embeddings, f)
        print(f"Saved embeddings to {embeddings_file}")

    def _fetch_text_from_url(self, url):
        """Fetches and parses the main text from a job description URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('main') or soup.find('article') or soup.find('body')
            if content:
                text = content.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text)
                print(f"✓ Successfully fetched {len(text)} characters from URL.")
                return text
            else:
                return soup.get_text(separator=' ', strip=True)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None

    def _parse_tech_queries(self, cleaned_tech_query):
        """
        (NEW Helper for V7) Splits a cleaned query into multiple sub-queries
        if it detects a list pattern (commas, "and").
        """
        # Split by comma OR the word "and"
        # "Python, SQL and JavaScript" -> ["Python", "SQL", "JavaScript"]
        queries = re.split(r',\s*|\s+and\s+', cleaned_tech_query, flags=re.IGNORECASE)
        
        # Clean up whitespace and remove empty strings
        queries = [q.strip() for q in queries if q.strip()]
        
        if not queries:
            return [cleaned_tech_query] # Fallback
            
        print(f" 	Detected {len(queries)} distinct tech queries: {queries}")
        return queries

    def _extract_features_locally(self, query):
        """
        (RAG Stage 1 - Local)
        Extract features AND create a "cleaned" tech-only query.
        """
        query_lower = query.lower()
        cleaned_tech_query = query
        
        features = {
            'soft_skill_requested': False,
            'role_level': 'mid',
            'max_duration': None
        }
        
        # --- Check for Soft Skill intent ---
        if any(w in query_lower for w in self.feature_keywords['soft']):
            features['soft_skill_requested'] = True

        # --- Role Level ---
        if any(w in query_lower for w in self.feature_keywords['level']):
             if any(w in query_lower for w in ['entry', 'junior', 'graduate', 'intern']):
                features['role_level'] = 'entry'
             elif any(w in query_lower for w in ['senior', 'lead', 'expert', 'principal']):
                features['role_level'] = 'senior'
            
        # --- Duration ---
        duration_matches = re.findall(r'(\d+)\s*(?:minute|min|minutes|mins).*?(less|maximum|max|under|within)', query_lower)
        duration_less_than = re.findall(r'(?:less|maximum|max|under|within).*?(\d+)\s*(?:minute|min|minutes|mins)', query_lower)
        
        if duration_matches:
            features['max_duration'] = int(duration_matches[0][0])
        elif duration_less_than:
            features['max_duration'] = int(duration_less_than[0])

        # --- Create the cleaned_tech_query ---
        all_noise_words = self.feature_keywords['soft'] + self.feature_keywords['level'] + self.feature_keywords['duration']
        pattern = r'\b(' + '|'.join(re.escape(w) for w in all_noise_words) + r')\b'
        cleaned_tech_query = re.sub(pattern, '', cleaned_tech_query, flags=re.IGNORECASE)

        # --- V7: Better cleaning ---
        non_skill_words = r'\b(looking|to|hire|who|are|proficient|in|need|an|assessment|package|that|can|test|all|skills|with|for|a)\b'
        cleaned_tech_query = re.sub(non_skill_words, '', cleaned_tech_query, flags=re.IGNORECASE)
        
        cleaned_tech_query = re.sub(r'\s+', ' ', cleaned_tech_query).strip()

        # If cleaning removed everything, fall back to original query
        if not cleaned_tech_query:
             cleaned_tech_query = query
            
        print(f"Extracted features: {features}")
        print(f"Cleaned tech-only query string: \"{cleaned_tech_query[:100]}...\"")
            
        return features, cleaned_tech_query

    def _validate_and_rank_locally(self, features, candidates, cleaned_tech_query, final_k):
        """
        (RAG Stage 3/4 - Local - Unchanged from V6)
        Re-scores candidates with a second "tech-only" query,
        then ranks using a multi-signal model.
        """
        print("Starting local validation & ranking (RAG Stage 3/4)...")
        print("--- BEGIN LOCAL TRACE ---")

        WEIGHTS = {
            'tech_sim': 0.60,      # 60% (Pure tech-only score)
            'soft_skill': 0.20,    # 20% (Is this a soft-skill doc?)
            'context_sim': 0.10,   # 10% (Original hybrid query score)
            'level': 0.10          # 10% (Role level match)
        }
        
        # --- STAGE 3: RE-SCORING ---
        print(f" 	Re-scoring {len(candidates)} candidates against tech-only query: \"{cleaned_tech_query}\"")
        
        candidate_indices = [c['doc_index'] for c in candidates]
        candidate_embeddings = self.assessment_embeddings[candidate_indices]
        
        tech_embedding = self.model.encode([cleaned_tech_query], convert_to_numpy=True)
        
        tech_similarities = cosine_similarity(tech_embedding, candidate_embeddings)[0]
        
        scored_candidates = []
        filter_log = {'removed_duration': 0, 'total_candidates': len(candidates)}

        # --- STAGE 4: RANKING ---
        for i, candidate in enumerate(candidates):
            # 1. HARD FILTERS
            if features.get('max_duration') is not None:
                if candidate['duration'] > features['max_duration']:
                    print(f" 	ⓘ FILTERED (Duration): '{candidate['name']}' ({candidate['duration']}m > {features['max_duration']}m)")
                    filter_log['removed_duration'] += 1
                    continue
            
            # 2. WEIGHTED SCORING
            name_lower = candidate['name'].lower()
            desc_lower = candidate['description'].lower()
            test_types = candidate.get('test_type', [])
            
            is_soft_skill_doc = 0.0
            collab_keywords = ['personality', 'behaviour', 'communication', 'team', 'opq']
            if any(kw in name_lower or kw in desc_lower for kw in collab_keywords):
                is_soft_skill_doc = 1.0
            if 'Personality & Behaviour' in test_types:
                is_soft_skill_doc = 1.0

            level_score = 0.5 # Neutral
            if features.get('role_level') == 'entry':
                if 'entry' in name_lower: level_score = 1.0
                elif 'advanced' in name_lower or 'senior' in name_lower: level_score = 0.0
            elif features.get('role_level') == 'senior':
                if 'entry' in name_lower: level_score = 0.0
                elif 'advanced' in name_lower or 'senior' in name_lower: level_score = 1.0

            scores = {
                'tech_sim': tech_similarities[i],
                'soft_skill': is_soft_skill_doc,
                'context_sim': candidate['context_similarity'],
                'level': level_score
            }
            
            final_score = (
                (scores['tech_sim'] * WEIGHTS['tech_sim']) +
                (scores['soft_skill'] * WEIGHTS['soft_skill']) +
                (scores['context_sim'] * WEIGHTS['context_sim']) +
                (scores['level'] * WEIGHTS['level'])
            )

            if features['soft_skill_requested'] and is_soft_skill_doc:
                 final_score *= 1.10
            
            scored_candidates.append({**candidate, 'final_score': final_score})
        
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        passed_count = len(scored_candidates)
        print(f" 	✓ RE-RANKED {passed_count} candidates using multi-signal score.")
        print("--- END LOCAL TRACE ---")

        reasoning = f"Local Validation Log: Processed {filter_log['total_candidates']} candidates. "
        reasoning += f"Filtered {filter_log['removed_duration']} for duration. "
        reasoning += f"Re-ranked {passed_count} using a multi-signal score (60% tech, 20% soft, 10% context, 10% level)."
        
        final_recommendations = scored_candidates[:final_k]
        for rec in final_recommendations:
            if 'final_score' in rec: del rec['final_score']
            if 'context_similarity' in rec: del rec['context_similarity']
            if 'doc_index' in rec: del rec['doc_index']

        return {"recommendations": final_recommendations, "reasoning": reasoning}

    def recommend(self, query, top_k=40, final_k=5):
        """
        Main Local RAG Pipeline ( Multi-Query)
        
        --- UPDATE: Default top_k=40, final_k=5 ---
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        print(f"\n{'='*80}")
        print("STEP 1: Feature Extraction & Query Cleaning (Local Rules)")
        print(f"{'='*80}")
        
        features, cleaned_tech_query = self._extract_features_locally(query)
        
        print(f"\n{'='*80}")
        print("STEP 2: Retrieval (Multi-Query Semantic Search)")
        print(f"{'='*80}")
        
        tech_queries = self._parse_tech_queries(cleaned_tech_query)
        
        all_candidates = {} # Use a dict to store unique candidates by doc_index
        
        k_per_query = max(10, (top_k // len(tech_queries)) + 2) # --- UPDATE: min 10 ---
        print(f"Running {len(tech_queries)} searches, retrieving {k_per_query} candidates each...")
        
        for tech_query in tech_queries:
            query_embedding = self.model.encode([tech_query], convert_to_numpy=True)
            similarities = cosine_similarity(query_embedding, self.assessment_embeddings)[0]
            top_indices_per_query = np.argsort(similarities)[::-1][:k_per_query]
            
            for idx in top_indices_per_query:
                idx = int(idx)
                if idx not in all_candidates:
                    all_candidates[idx] = {
                        **self.assessments[idx], 
                        'context_similarity': float(similarities[idx]),
                        'doc_index': idx
                    }
                else:
                    all_candidates[idx]['context_similarity'] = max(
                        all_candidates[idx]['context_similarity'], 
                        float(similarities[idx])
                    )

        print(f"Running 1 final search on FULL query to catch soft-skills...")
        full_query_embedding = self.model.encode([query], convert_to_numpy=True)
        full_similarities = cosine_similarity(full_query_embedding, self.assessment_embeddings)[0]
        top_indices_full = np.argsort(full_similarities)[::-1][:top_k]
        
        for idx in top_indices_full:
            idx = int(idx)
            if idx not in all_candidates:
                all_candidates[idx] = {
                    **self.assessments[idx], 
                    'context_similarity': float(full_similarities[idx]),
                    'doc_index': idx
                }
            else:
                all_candidates[idx]['context_similarity'] = max(
                    all_candidates[idx]['context_similarity'], 
                    float(full_similarities[idx])
                )

        candidates = list(all_candidates.values())
        print(f"✓ Retrieved {len(candidates)} unique candidates from all searches")
        
        print(f"\n{'='*80}")
        print("STEP 3 & 4: Re-scoring & Ranking (Local Rules)")
        print(f"{'='*80}")
        
        result_data = self._validate_and_rank_locally(features, candidates, cleaned_tech_query, final_k)

        return result_data

    def recommend_from_url(self, url, top_k=40, final_k=5):
        """
        Fetches a job description from a URL and runs the recommendation pipeline.
        
        --- UPDATE: Default top_k=40, final_k=5 ---
        """
        print(f"\n{'='*80}")
        print(f"Running recommendation from URL: {url}")
        print(f"{'='*80}")
        job_description_text = self._fetch_text_from_url(url)
        if job_description_text:
            return self.recommend(job_description_text, top_k, final_k)
        else:
            print("Failed to get text from URL. Aborting recommendation.")
            return {"recommendations": [], "reasoning": "Failed to fetch or parse URL."}

# --- EVALUATION AND PRINTING FUNCTIONS ---

def evaluate_recommendation(results, expected_urls):
    """
    Demonstrates a simple evaluation (evals) methodology.
    """
    print(f"\n{'='*88}")
    print("RUNNING EVALUATION (EVALS) METHODOLOGY")
    print(f"{'='*88}")
    
    predicted_urls = set([rec['url'] for rec in results])
    expected_urls = set(expected_urls)
    
    if not predicted_urls and not expected_urls:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
    
    true_positives = len(predicted_urls.intersection(expected_urls))
    
    precision = true_positives / len(predicted_urls) if predicted_urls else 0.0
    recall = true_positives / len(expected_urls) if expected_urls else 0.0
    
    f1_score = 0
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        
    print(f"Ground Truth (Expected): {len(expected_urls)} URLs")
    print(f"Recommendations (Predicted): {len(predicted_urls)} URLs")
    print(f"True Positives (Correct): {true_positives}")
    print("---")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:   {recall:.2%}")
    print(f"F1-Score: {f1_score:.2%}")
    
    return {"precision": precision, "recall": recall, "f1_score": f1_score}


def print_results(result_data, expected_urls=None):
    """Helper function to print recommendations and run evals."""
    print("\n" + "="*88)
    print("FINAL RECOMMENDATIONS (from Local RAG Pipeline)")
    print("="*88)
    
    results = result_data["recommendations"]
    reasoning = result_data["reasoning"]
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['name']}")
            print(f"   URL: {result['url']}")
            print(f"   Test Type: {', '.join(result['test_type'])}")
            print(f"   Duration: {result['duration']} min")
            print(f"   Description: {result['description'][:120]}...")
        
        print("\n" + "---" * 29)
        print("Qualitative Eval (Programmatic Reasoning):")
        print(f"> {reasoning}")
        print("---" * 29)
        
        if expected_urls:
            eval_metrics = evaluate_recommendation(results, expected_urls)
    else:
        print("No recommendations found matching the criteria.")


# --- Main execution block ---
if __name__ == "__main__":
    recommender = IntelligentRecommender()
    
    print("\n" + "="*88)
    print("TESTING LOCAL-ONLY RAG RECOMMENDER (MULTI-QUERY RETRIEVER)")
    print("="*88)
    
    # --- Test Case 1: Hybrid Query (Tech + Soft Skill) ---
    TEST_QUERY_1 = "Need a Java developer with good collaboration skills. Test duration should be less than 30 minutes"
    print(f"\nQuery 1: {TEST_QUERY_1}\n")
    
    EXPECTED_URLS_1 = [
        "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/core-java-advanced-level-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level/",
        "https://www.shl.com/solutions/products/product-catalog/view/occupational-personality-questionnaire-opq/",
        "https://www.shl.com/products/product-catalog/view/global-skills-development-report/"


    ]
    
    try:
        # --- UPDATE: Requesting 10 results ---
        result_data_1 = recommender.recommend(TEST_QUERY_1, top_k=40, final_k=5)
        print_results(result_data_1, EXPECTED_URLS_1)
        
    except Exception as e:
        print(f"\nAn error occurred during Test Case 1: {e}")

    # --- Test Case 2: Multi-Tech Query (Your Example) ---
    TEST_QUERY_2 = "Looking to hire who are proficient in Python, SQL, JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes."
    print(f"\nQuery 2: {TEST_QUERY_2}\n")
    
    EXPECTED_URLS_2 = [
        "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/sql-server-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/javascript-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/data-warehousing-concepts/",
        "https://www.shl.com/solutions/products/product-catalog/view/microsoft-excel-365-new/",
        "https://www.shl.com/solutions/products/product-catalog/view/drupal-new/"
    ]

    try:
        # --- UPDATE: Requesting 10 results ---
        result_data_2 = recommender.recommend(TEST_QUERY_2, top_k=40, final_k=5)
        print_results(result_data_2, EXPECTED_URLS_2) 
        
    except Exception as e:
        print(f"\nAn error occurred during Test Case 2: {e}")

    # --- Test Case 3: URL-based Query ---
    TEST_URL_3 = "https://www.linkedin.com/jobs/view/4295882584" # Example: Moloco ML Engineer
    print(f"\nQuery 3: Running from URL: {TEST_URL_3}\n")

    try:
        # --- UPDATE: Requesting 10 results ---
        result_data_3 = recommender.recommend_from_url(TEST_URL_3, top_k=40, final_k=5)
        print_results(result_data_3, expected_urls=None)
        
    except Exception as e:
        print(f"\nAn error occurred during Test Case 3 (URL): {e}")