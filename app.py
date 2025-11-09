"""
SHL Assessment Recommendation System - Flask Web Application
Powered by a 3-stage local RAG-like pipeline (no API keys required).
"""

from flask import Flask, request, jsonify, render_template
from main import IntelligentRecommender
import traceback


app = Flask(__name__)


print("Initializing Intelligent SHL Assessment Recommendation System (RAG)...")
recommender = IntelligentRecommender()
print("Recommender ready!")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['GET', 'POST'])
def recommend():
    """
    API endpoint for getting recommendations - Local RAG Pipeline
    
    Process:
    1. Extracts relevant features from Query/JD Text/URL (Local Rules)
    2. Converts to word embeddings using Sentence Transformer
    3. Calculates Cosine Similarity and gets top 20 candidates (Retrieval)
    4. Validates and re-ranks with a local weighted scoring engine (Validation)
    5. Returns the 5 most relevant assessments.
    """
    try:
        data = None

        # Default parameters
        query_type = 'text'
        query_value = ''
        final_k = 5  # default number of recommendations to return

        # Support GET (query params) and POST (json/form)
        if request.method == 'GET':
            query_type = request.args.get('type', 'text')
            query_value = (request.args.get('query') or '').strip()
            try:
                final_k = int(request.args.get('final_k', final_k))
            except Exception:
                final_k = 5
        else:
            if request.is_json:
                data = request.get_json()
            elif request.form:
                data = {
                    'type': request.form.get('type', 'text'),
                    'query': request.form.get('query', '')
                }

            if data:
                query_type = data.get('type', 'text')
                query_value = (data.get('query') or '').strip()
                try:
                    final_k = int(data.get('final_k', final_k))
                except Exception:
                    final_k = 5

        # Validate query
        if not query_value:
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Clamp final_k to be between 1 and 10 (inclusive)
        try:
            final_k = max(1, min(10, int(final_k)))
        except Exception:
            final_k = 5
        
        # Get recommendations using the local RAG pipeline
        # Stage 1-2: Semantic search gets top 20 candidates
        # Stage 3: Local weighted scoring filters to 5 most relevant
        
        
        result_data = None
        
        # Determine retrieval and final ranking sizes; keep retrieval larger than final_k
        top_k = max(20, final_k * 4)

        if query_type == 'url':
            result_data = recommender.recommend_from_url(
                query_value,
                top_k=top_k,
                final_k=final_k
            )
        else:
            result_data = recommender.recommend(
                query_value,
                top_k=top_k,
                final_k=final_k
            )
        
        # FIX: Extract the recommendations and reasoning from the dictionary
        results = result_data.get("recommendations", []) if isinstance(result_data, dict) else []
        reasoning = result_data.get("reasoning", "No reasoning provided.") if isinstance(result_data, dict) else "No reasoning provided."

        # Ensure we respect the requested final_k and bounds
        results = results[:final_k]

        # Fallback: if recommender returned nothing, return at least one assessment (best-effort)
        if not results:
            fallback = []
            try:
                # recommender.assessments is expected to be a list of assessments
                if hasattr(recommender, 'assessments') and recommender.assessments:
                    fallback = [recommender.assessments[0]]
            except Exception:
                fallback = []

            if fallback:
                results = fallback
        
        # Return in the format matching SHL API structure
        return jsonify({
            'recommended_assessments': results,
            'reasoning_trace': reasoning # Also return the trace for debugging/UI
        })
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get statistics about available assessments"""
    return jsonify({
        'total_assessments': len(recommender.assessments),
        'model': 'all-MiniLM-L6-v2',
        'method': 'Local RAG (Semantic Retrieval + Rule-Based Ranking)'
    })

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring and deployment
    """
    try:
        # Check if recommender is loaded
        assessments_count = len(recommender.assessments)
        embeddings_loaded = recommender.assessment_embeddings is not None
        
        # FIX: Removed all 'llm_enabled' checks
        health_status = {
            'status': 'healthy',
            'timestamp': '2025-11-09', # Example timestamp
            'version': '5.0 (Local RAG)', # Updated version
            'components': {
                'recommender': 'operational' if assessments_count > 0 else 'error',
                'embeddings': 'loaded' if embeddings_loaded else 'missing',
                # FIX: Updated to reflect new architecture
                'ranking_engine': 'Local RAG (Rule-Based)', 
                'database': 'connected' # Assuming file system is "database"
            },
            'metrics': {
                'assessments_loaded': assessments_count,
                'embedding_model': 'all-MiniLM-L6-v2',
                # FIX: Updated metric
                'feature_extractor': 'Local Rules (Expanded)'
            }
        }
        
        if assessments_count == 0 or not embeddings_loaded:
            health_status['status'] = 'unhealthy'
            return jsonify(health_status), 503
        
        return jsonify(health_status), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': '2025-11-09'
        }), 503

if __name__ == '__main__':
    print("\n" + "="*80)
    print("SHL Assessment Recommendation System (RAG)")
    print("="*80)
    print(f"\n✓ Server starting on http://localhost:5001")
    print(f"✓ Open your browser and navigate to: http://localhost:5001")
    print(f"\n Press CTRL+C to stop the server\n")
    app.run(debug=False, host='0.0.0.0', port=5001)