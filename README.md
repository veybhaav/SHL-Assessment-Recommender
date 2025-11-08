# SHL Assessment Recommendation System

A **Generative AI-powered** web application that **reasons through** natural language queries or job descriptions to recommend the most relevant SHL assessments.

## Features

- **Natural Language Queries**: Simply describe your hiring needs in plain English.
- **Job Description Support**: Paste a complete job description or provide a URL.
- **Generative AI Reasoning**: Leverages a Large Language Model (LLM) to understand context, infer needs, and provide human-like recommendations.
- **Smart Recommendations**: Returns the most relevant assessments, often **with a justification** for *why* each was chosen.
- **User-Friendly Interface**: Clean, modern web interface with responsive design.
- **Dynamic Processing**: Queries the LLM in real-time to provide the most context-aware answers.

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- A valid API key for your chosen LLM (e.g., Google AI Studio or OpenAI) set as an environment variable.

### Setup

1.  **Clone or navigate to the project directory:**
    ```bash
    cd /Users/soc_team/Desktop/SHL
    ```

2.  **Set your API Key:**
    ```bash
    # For macOS/Linux
    export GOOGLE_API_KEY='YOUR_API_KEY_HERE'
    
    # For Windows
    set GOOGLE_API_KEY='YOUR_API_KEY_HERE'
    ```

3.  **Install dependencies** (already installed in .venv):
    ```bash
    # Note: 'sentence-transformers' is replaced with the LLM's client library
    pip install pandas numpy scikit-learn flask beautifulsoup4 requests openpyxl google-generativeai
    # or 'openai' if using OpenAI
    ```

4.  **Verify data files exist:**
    - `shl_assessments.json` - Assessment database (this is now the primary data source)

## Usage

### Starting the Application

1.  **Run the Flask server:**
    ```bash
    python app.py
    ```

2.  **Open your browser and navigate to:**
    ```
    http://localhost:5001
    ```

Note: If port 5001 is in use, you can change it in `app.py` (line 81).

### Using the Web Interface

**Option 1: Natural Language Query / Job Description**
1.  Select the "Natural Language Query / Job Description" tab
2.  Enter your requirements or paste a job description
3.  Click "Get Recommendations"

**Example queries:**
- "I need to hire a Java developer with strong problem-solving skills"
- "Looking for a sales representative who can communicate effectively"
- "Need to assess leadership skills for senior management position"

**Option 2: Job Description URL**
1.  Select the "Job Description URL" tab
2.  Enter a URL containing a job description
3.  Click "Get Recommendations"

### Using the API Programmatically

**Recommend from text:**
```python
from recommender import AssessmentRecommender

recommender = AssessmentRecommender()

### Get recommendations
results = recommender.recommend(
    "I am hiring for Java developers who can collaborate with business teams",
    top_k=5
)
print(results)
results = recommender.recommend_from_url(
    "[https://example.com/job-posting](https://example.com/job-posting)",
    top_k=5
)

### API Endpoints
POST /api/recommend
Get assessment recommendations

Request Body:

JSON

{
  "type": "text",  // or "url"
  "query": "Your query or URL here",
  "top_k": 5
}
Response:

JSON

{
  "success": true,
  "count": 5,
  "recommendations": [
    {
      "Assessment Name": "Java 8",
      "URL": "[https://www.shl.com/](https://www.shl.com/)...",
      "Description": "...",
      "Reasoning": "This assessment is ideal for validating the specific 'Java' technical skill mentioned in your query."
    }
  ]
}
GET /api/stats
Get system statistics

Response:

JSON

{
  "total_assessments": 54,
  "model": "Google Gemini Pro (or similar LLM)",
  "method": "Generative AI (LLM-based Reasoning)"
}