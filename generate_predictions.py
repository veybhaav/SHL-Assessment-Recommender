#!/usr/bin/env python3
"""
Generate predictions for test set queries
Reads test_new_data.xlsx - Test-Set.csv and generates 10 predictions per query.
Output: predictions_latest.csv with columns [query, assesment_url]
"""

import pandas as pd
import requests
import json
import time

# Configuration

TEST_FILE = "test_new_data.xlsx - Test-Set.csv"
OUTPUT_FILE = "predictions_latest.csv" 
API_URL = "http://localhost:5001/api/recommend"

def get_predictions(query):
    """
    Get 10 assessment URL predictions from the API.
    Returns a list of URLs.
    """
    try:
        headers = {"Content-Type": "application/json"}
        # Add final_k=5 to the payload
        payload = {
            "type": "text", 
            "query": query,
            "final_k": 10
        }
        
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            assessments = data.get('recommended_assessments', [])
            
            # Return a list of URLs, not a string
            urls = [assessment.get('url', '') for assessment in assessments]
            return urls if urls else []
        else:
            print(f"  ⚠️  API error {response.status_code}")
            return []
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []

def main():
    print("="*80)
    print("SHL Assessment Prediction Generator (10 per query)")
    print("="*80)
    
    # Load test queries
    print(f"\n📂 Loading test queries from {TEST_FILE}...")
    try:
        # --- FIX 2: Added encoding='latin1' to handle the error ---
        df_test = pd.read_csv(TEST_FILE, encoding='latin1')
        print(f"✓ Loaded {len(df_test)} test queries")
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return
    
    # Check API connection
    print(f"\n🔌 Testing API connection...")
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            print("✓ API server is running")
        else:
            print("⚠️  API server may not be healthy")
    except:
        print("❌ Cannot connect to API server!")
        print("Please start the server with: python app.py")
        return
    
    # Generate predictions
    print(f"\n🎯 Generating predictions...")
    print("="*80)
    
    results = []
    
    for idx, row in df_test.iterrows():
        # Check if 'Query' column exists
        if 'Query' not in row:
            print(f"  ❌ Skipping row {idx+1}: 'Query' column not found.")
            continue
            
        query = row['Query']
        
        # Handle potential empty queries
        if not query or pd.isna(query):
            print(f"  ⚠️ Skipping row {idx+1}: Query is empty.")
            continue
            
        # Display progress
        display_query = str(query)[:80] + "..." if len(str(query)) > 80 else str(query)
        print(f"\n📝 Query {idx + 1}/{len(df_test)}: {display_query}")
        
        # Get list of 10 URLs
        urls = get_predictions(query)
        
        if urls:
            print(f"   ✓ Generated {len(urls)} predictions")
            # Add one row to results for EACH URL
            for url in urls:
                results.append({
                    'query': query,
                    'assesment_url': url  # New column name
                })
        else:
            print(f"   ❌ No predictions generated")
            # Add a single row with an empty URL to show the query was processed
            results.append({
                'query': query,
                'assesment_url': ''
            })
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    # Save results
    print(f"\n💾 Saving predictions to {OUTPUT_FILE}...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✓ Saved {len(df_results)} total rows")
    
    # New summary logic for the "long" format
    total_queries_processed = df_results['query'].nunique()
    total_recommendations = df_results['assesment_url'].str.len().gt(0).sum()
    
    print(f"\n📊 Summary:")
    print(f"   - Total unique queries processed: {total_queries_processed}")
    print(f"   - Total recommendation rows saved: {len(df_results)}")
    print(f"   - Total non-empty URLs generated: {total_recommendations}")
    if total_queries_processed > 0:
        avg_urls = total_recommendations / total_queries_processed
        print(f"   - Average URLs per query: {avg_urls:.1f}")
    
    print("\n" + "="*80)
    print("✅ Predictions generated successfully!")
    print(f"📄 Output file: {OUTPUT_FILE}")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()