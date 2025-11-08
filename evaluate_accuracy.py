"""
Evaluation System for SHL Assessment Recommendation
Measures Mean Recall@K and Recommendation Balance
"""

import json
import numpy as np
from main import LLMEnhancedRecommender

# Test queries with ground truth relevant assessments
TEST_QUERIES = [
    {
        "query": "Need a Java developer who is good in collaborating with external teams and stakeholders",
        "relevant_assessments": [
            "Java 8 (New)",
            "Core Java (Entry Level) (New)",
            "Core Java (Advanced Level) (New)",
            "OPQ Leadership Report",
            "Interpersonal Communications",
            "Business Communication (adaptive)",
            "SHL Verify Interactive - Inductive Reasoning"
        ],
        "expected_balance": {
            "technical": 0.4,  # 40% Java/technical
            "behavioral": 0.4,  # 40% communication/collaboration
            "cognitive": 0.2   # 20% reasoning/problem-solving
        }
    },
    {
        "query": "Python developer with machine learning and data analysis skills",
        "relevant_assessments": [
            "Python (New)",
            "Data Warehousing Concepts",
            "Tableau (New)",
            "SHL Verify Interactive - Inductive Reasoning",
            "Verify - Numerical Ability",
            "SHL Verify Interactive Numerical Calculation"
        ],
        "expected_balance": {
            "technical": 0.6,
            "analytical": 0.4
        }
    },
    {
        "query": "SQL database developer for data engineering team",
        "relevant_assessments": [
            "SQL Server (New)",
            "Data Warehousing Concepts",
            "Automata - SQL (New)",
            "SHL Verify Interactive - Inductive Reasoning",
            "Verify - Numerical Ability"
        ],
        "expected_balance": {
            "technical": 0.7,
            "analytical": 0.3
        }
    },
    {
        "query": "Front-end web developer with strong UI/UX skills",
        "relevant_assessments": [
            "JavaScript (New)",
            "HTML & CSS (New)",
            "CSS3 (New)",
            "SHL Verify Interactive - Inductive Reasoning"
        ],
        "expected_balance": {
            "technical": 0.8,
            "cognitive": 0.2
        }
    },
    {
        "query": "Team leader with excellent communication and management skills",
        "relevant_assessments": [
            "OPQ Leadership Report",
            "Enterprise Leadership Report 2.0",
            "OPQ Team Types and Leadership Styles Report",
            "Interpersonal Communications",
            "Business Communication (adaptive)",
            "Occupational Personality Questionnaire OPQ32r"
        ],
        "expected_balance": {
            "behavioral": 0.7,
            "leadership": 0.3
        }
    },
    {
        "query": "QA automation engineer with Selenium experience",
        "relevant_assessments": [
            "Selenium (New)",
            "Manual Testing (New)",
            "Automata - Fix (New)",
            "Java 8 (New)",
            "Python (New)",
            "SHL Verify Interactive - Inductive Reasoning"
        ],
        "expected_balance": {
            "technical": 0.7,
            "cognitive": 0.3
        }
    },
    {
        "query": "Data analyst with Excel and visualization skills",
        "relevant_assessments": [
            "Microsoft Excel 365 (New)",
            "Tableau (New)",
            "Data Warehousing Concepts",
            "Verify - Numerical Ability",
            "SHL Verify Interactive Numerical Calculation"
        ],
        "expected_balance": {
            "technical": 0.5,
            "analytical": 0.5
        }
    },
    {
        "query": ".NET developer for enterprise applications",
        "relevant_assessments": [
            ".NET Framework 4.5",
            ".NET MVC (New)",
            "ASP .NET with C# (New)",
            "ADO.NET (New)",
            "C# (New)",
            "SHL Verify Interactive - Inductive Reasoning"
        ],
        "expected_balance": {
            "technical": 0.8,
            "cognitive": 0.2
        }
    },
    {
        "query": "Customer service representative with strong English communication",
        "relevant_assessments": [
            "Business Communication (adaptive)",
            "Interpersonal Communications",
            "English Comprehension (New)",
            "Written English v1",
            "SVAR - Spoken English (Indian Accent)  (New)",
            "Occupational Personality Questionnaire OPQ32r"
        ],
        "expected_balance": {
            "communication": 0.6,
            "behavioral": 0.4
        }
    },
    {
        "query": "Full-stack developer with database and front-end skills",
        "relevant_assessments": [
            "JavaScript (New)",
            "SQL Server (New)",
            "HTML & CSS (New)",
            "Python (New)",
            "Java 8 (New)",
            ".NET MVC (New)"
        ],
        "expected_balance": {
            "technical": 0.9,
            "cognitive": 0.1
        }
    }
]


def calculate_recall_at_k(recommended, relevant, k):
    """
    Calculate Recall@K for a single query
    
    Recall@K = (Number of relevant assessments in top K) / (Total relevant assessments)
    """
    # Get top K recommendations
    top_k = recommended[:k]
    top_k_names = [r['name'] for r in top_k]
    
    # Count how many relevant assessments are in top K
    relevant_in_top_k = sum(1 for name in relevant if name in top_k_names)
    
    # Calculate recall
    recall = relevant_in_top_k / len(relevant) if len(relevant) > 0 else 0
    
    return recall, relevant_in_top_k, len(relevant)


def calculate_mean_recall_at_k(test_results, k):
    """
    Calculate Mean Recall@K across all test queries
    
    MeanRecall@K = (1/N) * Σ Recall@K_i
    where N is the number of test queries
    """
    recalls = []
    
    for result in test_results:
        recall, _, _ = calculate_recall_at_k(
            result['recommendations'],
            result['relevant_assessments'],
            k
        )
        recalls.append(recall)
    
    mean_recall = np.mean(recalls)
    return mean_recall, recalls


def analyze_recommendation_balance(recommendations, query_info):
    """
    Analyze if recommendations are balanced according to query requirements
    """
    # Categorize each recommendation
    categorized = {
        'technical': [],
        'behavioral': [],
        'cognitive': [],
        'analytical': [],
        'communication': [],
        'leadership': []
    }
    
    for rec in recommendations:
        name_lower = rec['name'].lower()
        desc_lower = rec['description'].lower()
        
        # Technical skills
        technical_keywords = ['python', 'java', 'javascript', 'sql', '.net', 'c#', 'c++', 
                             'html', 'css', 'selenium', 'programming', 'coding', 'development']
        if any(kw in name_lower or kw in desc_lower for kw in technical_keywords):
            categorized['technical'].append(rec['name'])
        
        # Behavioral/Personality
        behavioral_keywords = ['personality', 'opq', 'behaviour', 'behavior', 'team', 'work style']
        if any(kw in name_lower or kw in desc_lower for kw in behavioral_keywords):
            categorized['behavioral'].append(rec['name'])
        
        # Leadership
        leadership_keywords = ['leadership', 'manager', 'management', 'enterprise']
        if any(kw in name_lower or kw in desc_lower for kw in leadership_keywords):
            categorized['leadership'].append(rec['name'])
        
        # Cognitive/Reasoning
        cognitive_keywords = ['verify', 'reasoning', 'inductive', 'deductive', 'cognitive', 'ability']
        if any(kw in name_lower or kw in desc_lower for kw in cognitive_keywords):
            categorized['cognitive'].append(rec['name'])
        
        # Analytical
        analytical_keywords = ['data', 'analysis', 'numerical', 'excel', 'tableau', 'warehouse']
        if any(kw in name_lower or kw in desc_lower for kw in analytical_keywords):
            categorized['analytical'].append(rec['name'])
        
        # Communication
        communication_keywords = ['communication', 'english', 'writing', 'spoken', 'interpersonal']
        if any(kw in name_lower or kw in desc_lower for kw in communication_keywords):
            categorized['communication'].append(rec['name'])
    
    # Calculate actual distribution
    total = len(recommendations)
    actual_distribution = {
        category: len(items) / total if total > 0 else 0
        for category, items in categorized.items()
    }
    
    return categorized, actual_distribution


def evaluate_system():
    """
    Main evaluation function
    """
    print("="*80)
    print("SHL ASSESSMENT RECOMMENDATION SYSTEM - ACCURACY EVALUATION")
    print("="*80)
    
    # Initialize recommender
    print("\n🔧 Initializing recommender system...")
    recommender = LLMEnhancedRecommender()
    
    # Test all queries
    test_results = []
    
    print("\n" + "="*80)
    print("TESTING QUERIES")
    print("="*80)
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Query: {test_case['query']}")
        
        # Get recommendations
        recommendations = recommender.recommend(test_case['query'], top_k=10, final_k=10)
        
        # Calculate Recall@K for different K values
        recall_at_5, relevant_in_5, total_relevant = calculate_recall_at_k(
            recommendations, test_case['relevant_assessments'], 5
        )
        recall_at_10, relevant_in_10, _ = calculate_recall_at_k(
            recommendations, test_case['relevant_assessments'], 10
        )
        
        # Analyze balance
        categorized, actual_dist = analyze_recommendation_balance(
            recommendations[:10], 
            test_case
        )
        
        result = {
            'query': test_case['query'],
            'recommendations': recommendations,
            'relevant_assessments': test_case['relevant_assessments'],
            'recall_at_5': recall_at_5,
            'recall_at_10': recall_at_10,
            'relevant_in_5': relevant_in_5,
            'relevant_in_10': relevant_in_10,
            'total_relevant': total_relevant,
            'categorized': categorized,
            'actual_distribution': actual_dist,
            'expected_balance': test_case.get('expected_balance', {})
        }
        
        test_results.append(result)
        
        print(f"   ✓ Recall@5: {recall_at_5:.2%} ({relevant_in_5}/{total_relevant})")
        print(f"   ✓ Recall@10: {recall_at_10:.2%} ({relevant_in_10}/{total_relevant})")
        print(f"   ✓ Recommended: {[r['name'] for r in recommendations[:5]]}")
    
    # Calculate overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    mean_recall_5, recalls_5 = calculate_mean_recall_at_k(test_results, 5)
    mean_recall_10, recalls_10 = calculate_mean_recall_at_k(test_results, 10)
    
    print(f"\n📊 Mean Recall@5: {mean_recall_5:.2%}")
    print(f"📊 Mean Recall@10: {mean_recall_10:.2%}")
    
    print(f"\n📈 Individual Recall@10 scores:")
    for i, (test_case, recall) in enumerate(zip(TEST_QUERIES, recalls_10), 1):
        print(f"   {i}. {recall:.2%} - {test_case['query'][:60]}...")
    
    # Analyze balance across all queries
    print("\n" + "="*80)
    print("RECOMMENDATION BALANCE ANALYSIS")
    print("="*80)
    
    for i, result in enumerate(test_results, 1):
        print(f"\n[{i}] {result['query'][:70]}...")
        print(f"\n   Actual Distribution:")
        for category, percentage in result['actual_distribution'].items():
            if percentage > 0:
                print(f"      • {category.capitalize()}: {percentage:.1%} ({int(percentage * 10)} tests)")
        
        if result['expected_balance']:
            print(f"\n   Expected Balance:")
            for category, percentage in result['expected_balance'].items():
                print(f"      • {category.capitalize()}: {percentage:.1%}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n✅ Total Test Queries: {len(TEST_QUERIES)}")
    print(f"✅ Mean Recall@5: {mean_recall_5:.2%}")
    print(f"✅ Mean Recall@10: {mean_recall_10:.2%}")
    print(f"✅ Min Recall@10: {min(recalls_10):.2%}")
    print(f"✅ Max Recall@10: {max(recalls_10):.2%}")
    print(f"✅ Std Dev: {np.std(recalls_10):.2%}")
    
    # Performance rating
    if mean_recall_10 >= 0.8:
        rating = "EXCELLENT"
        emoji = "🌟"
    elif mean_recall_10 >= 0.6:
        rating = "GOOD"
        emoji = "✅"
    elif mean_recall_10 >= 0.4:
        rating = "FAIR"
        emoji = "⚠️"
    else:
        rating = "NEEDS IMPROVEMENT"
        emoji = "❌"
    
    print(f"\n{emoji} Performance Rating: {rating}")
    print(f"{emoji} System Accuracy: {mean_recall_10:.1%}")
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_results = []
        for result in test_results:
            serializable_result = {
                'query': result['query'],
                'recall_at_5': float(result['recall_at_5']),
                'recall_at_10': float(result['recall_at_10']),
                'relevant_in_5': int(result['relevant_in_5']),
                'relevant_in_10': int(result['relevant_in_10']),
                'total_relevant': int(result['total_relevant']),
                'categorized': result['categorized'],
                'actual_distribution': {k: float(v) for k, v in result['actual_distribution'].items()},
                'expected_balance': result['expected_balance'],
                'top_5_recommendations': [r['name'] for r in result['recommendations'][:5]]
            }
            serializable_results.append(serializable_result)
        
        summary = {
            'mean_recall_at_5': float(mean_recall_5),
            'mean_recall_at_10': float(mean_recall_10),
            'min_recall_10': float(min(recalls_10)),
            'max_recall_10': float(max(recalls_10)),
            'std_dev': float(np.std(recalls_10)),
            'performance_rating': rating,
            'test_results': serializable_results
        }
        
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Detailed results saved to evaluation_results.json")
    
    return test_results, mean_recall_10


if __name__ == "__main__":
    test_results, mean_recall = evaluate_system()
    
    print("\n" + "="*80)
    print(f"🎯 FINAL SCORE: Mean Recall@10 = {mean_recall:.2%}")
    print("="*80)
