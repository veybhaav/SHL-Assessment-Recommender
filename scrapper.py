"""
SHL Individual Test Solutions Scraper
Scrapes ONLY Individual Test Solutions from https://www.shl.com/solutions/products/product-catalog/
Excludes Pre-packaged Job Solutions
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import re
from urllib.parse import urljoin, urlparse

class SHLIndividualTestScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com"
        self.catalog_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.assessments = []
        self.visited_urls = set()
        
        # Keywords to identify INDIVIDUAL TESTS (not job solutions)
        self.individual_test_indicators = [
            'python', 'java', 'javascript', 'sql', 'c++', 'c#', '.net',
            'html', 'css', 'selenium', 'testing', 'qa', 'automata',
            'tableau', 'excel', 'powerpoint', 'word', 'outlook',
            'opq', 'personality', 'verify', 'reasoning', 'ability', 'cognitive',
            'verbal', 'numerical', 'inductive', 'deductive',
            'english', 'language', 'communication', 'writing',
            'accounting', 'bookkeeping', 'finance',
            'marketing', 'seo', 'advertising', 'digital',
            'leadership report', 'team types', 'enterprise leadership',
            'global skills', 'data warehousing', 'database',
            'drupal', 'wordpress', 'web development'
        ]
        
        # Keywords to EXCLUDE (job solution packages)
        self.exclude_keywords = [
            'solution', 'job focused', 'jfa', 'short form',
            'manager solution', 'agent solution', 'clerk solution',
            'representative solution', 'associate solution',
            'professional solution', 'specialist solution',
            'supervisor solution', 'coordinator solution'
        ]
    
    def is_individual_test(self, name, url):
        """Determine if this is an individual test (not a job solution package)"""
        name_lower = name.lower()
        url_lower = url.lower()
        
        # Exclude if it contains job solution keywords
        for exclude in self.exclude_keywords:
            if exclude in name_lower:
                # Exception: Keep "Global Skills Assessment" even though it has no "solution"
                if 'global skills' in name_lower:
                    return True
                return False
        
        # Include if it matches individual test patterns
        for indicator in self.individual_test_indicators:
            if indicator in name_lower or indicator in url_lower:
                return True
        
        return False
    
    def crawl_catalog_pages(self):
        """Crawl all pages in the catalog to find assessment links"""
        print("="*80)
        print("CRAWLING SHL CATALOG FOR ALL PRODUCTS (INDIVIDUAL TEST SOLUTIONS)")
        print("="*80)
        
        assessment_urls = set()
        
        # Crawl all pages with pagination - type=2 seems to have all products
        base_url = "https://www.shl.com/solutions/products/product-catalog/"
        pages_to_visit = [f"{base_url}?type=2&start={start}" for start in range(0, 500, 12)]
        visited_pages = set()
        
        while pages_to_visit:
            current_page = pages_to_visit.pop(0)
            
            if current_page in visited_pages:
                continue
            
            visited_pages.add(current_page)
            print(f"\n📄 Crawling page: {current_page}")
            
            try:
                response = requests.get(current_page, headers=self.headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all links on the page
                all_links = soup.find_all('a', href=True)
                
                products_found = 0
                for link in all_links:
                    href = link.get('href', '')
                    full_url = urljoin(self.base_url, href)
                    
                    # Check if it's an assessment detail page
                    if '/product-catalog/view/' in full_url:
                        # Get the assessment name from link text or title
                        name = link.get_text(strip=True)
                        if not name:
                            name = link.get('title', '')
                        
                        # Add URL first, then check if it's an individual test
                        if full_url not in assessment_urls:
                            assessment_urls.add(full_url)
                            products_found += 1
                            
                            # Check if it's an individual test (not job solution)
                            if self.is_individual_test(name, full_url):
                                print(f"   ✓ Individual Test: {name}")
                            else:
                                print(f"   - Job Solution: {name}")
                
                print(f"   📊 Found {products_found} new products on this page")
                
                # Check if there are more products (if we found 12+, there might be more pages)
                if products_found >= 12:
                    print(f"   → Page has products, continuing pagination...")
                else:
                    print(f"   → Reached end of products")
                    break  # Stop when we hit a page with no products
                
                time.sleep(1)  # Be respectful
                
            except Exception as e:
                print(f"   ✗ Error crawling {current_page}: {e}")
                continue
        
        # Now filter to only individual tests
        individual_test_urls = []
        print(f"\n{'='*80}")
        print(f"FILTERING TO INDIVIDUAL TEST SOLUTIONS ONLY")
        print(f"{'='*80}\n")
        
        for url in assessment_urls:
            # Extract name from URL for filtering
            url_slug = url.rstrip('/').split('/')[-1]
            name_from_url = url_slug.replace('-', ' ').title()
            
            if self.is_individual_test(name_from_url, url):
                individual_test_urls.append(url)
        
        print(f"\n✓ Total products found: {len(assessment_urls)}")
        print(f"✓ Individual Test Solutions: {len(individual_test_urls)}")
        print(f"✓ Job Solution Packages filtered out: {len(assessment_urls) - len(individual_test_urls)}")
        print(f"✓ Crawled {len(visited_pages)} catalog pages")
        
        return individual_test_urls
    
    def extract_assessment_details(self, url):
        """Extract detailed information from an individual test page"""
        try:
            time.sleep(1)  # Be respectful
            
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Initialize assessment data
            assessment = {
                'url': url,
                'name': '',
                'description': '',
                'adaptive_support': 'No',
                'remote_support': 'Yes',
                'test_type': [],
                'duration': None
            }
            
            # Extract name
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text()
                title_text = re.sub(r'\s*\|\s*SHL.*$', '', title_text)
                title_text = re.sub(r'\s*-\s*SHL.*$', '', title_text)
                assessment['name'] = title_text.strip()
            
            h1_tag = soup.find('h1')
            if h1_tag and not assessment['name']:
                assessment['name'] = h1_tag.get_text().strip()
            
            # Extract from URL if still no name
            if not assessment['name']:
                url_parts = url.rstrip('/').split('/')
                name_from_url = url_parts[-1].replace('-', ' ').title()
                assessment['name'] = name_from_url
            
            # Filter out if this is a job solution
            if not self.is_individual_test(assessment['name'], url):
                return None
            
            # Extract description
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc:
                desc = meta_desc.get('content', '').strip()
                if desc and len(desc) > 50 and 'We recommend' not in desc:
                    assessment['description'] = desc
            
            # Look for description in content
            if not assessment['description']:
                # Try to find main content area
                content_selectors = [
                    soup.find('div', class_=re.compile(r'description|overview|content|summary', re.I)),
                    soup.find('section', class_=re.compile(r'description|overview|content', re.I)),
                    soup.find('div', id=re.compile(r'description|overview|content', re.I))
                ]
                
                for content_area in content_selectors:
                    if content_area:
                        paragraphs = content_area.find_all('p', limit=5)
                        for p in paragraphs:
                            text = p.get_text().strip()
                            if len(text) > 100 and not text.startswith('We recommend'):
                                assessment['description'] = text
                                break
                    if assessment['description']:
                        break
            
            # Get page text for analysis
            page_text = soup.get_text().lower()
            
            # Determine test type based on content
            test_types = []
            
            # Check for personality/behavior
            if any(word in page_text for word in ['personality', 'behaviour', 'behavior', 'opq', 'trait', 'work style', 'preference']):
                test_types.append('Personality & Behaviour')
            
            # Check for competencies/abilities
            if any(word in page_text for word in ['competenc', 'ability', 'aptitude', 'reasoning', 'cognitive', 'verify', 'thinking']):
                test_types.append('Competencies')
            
            # Check for knowledge/skills
            if any(word in page_text for word in ['knowledge', 'skill', 'programming', 'technical', 'coding', 'language', 'software', 'test', 'proficiency']):
                test_types.append('Knowledge & Skills')
            
            # Default to Knowledge & Skills if nothing matched
            if not test_types:
                test_types.append('Knowledge & Skills')
            
            assessment['test_type'] = test_types
            
            # Extract duration - SHL specific pattern
            duration_patterns = [
                r'approximate\s+completion\s+time\s+in\s+minutes\s*=\s*(\d+)',  # SHL specific format
                r'completion\s+time[:\s]*(\d+)\s*(?:minute|min)',
                r'duration[:\s]*(\d+)\s*(?:minute|min)',
                r'takes?\s+(?:approximately\s+)?(\d+)\s*(?:minute|min)',
                r'(\d+)\s*(?:minute|min|mins)(?!\s*(?:per|each))'
            ]
            
            for pattern in duration_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                if matches:
                    try:
                        duration = int(matches[0])
                        if 5 <= duration <= 180:  # Reasonable range
                            assessment['duration'] = duration
                            break
                    except:
                        continue
            
            # Check for adaptive support
            if any(word in page_text for word in ['adaptive', 'adapts to', 'tailored', 'personalized']):
                assessment['adaptive_support'] = 'Yes'
            
            # Assume remote support unless stated otherwise
            if 'on-site only' in page_text or 'not remote' in page_text or 'in-person only' in page_text:
                assessment['remote_support'] = 'No'
            
            return assessment
            
        except Exception as e:
            print(f"   ✗ Error extracting from {url}: {e}")
            return None
    
    def enhance_data(self):
        """Enhance and clean scraped data"""
        print("\n" + "="*80)
        print("ENHANCING DATA")
        print("="*80)
        
        for assessment in self.assessments:
            # Generate description if missing or poor quality
            if not assessment['description'] or len(assessment['description']) < 50:
                assessment['description'] = self.generate_smart_description(
                    assessment['name'], 
                    assessment['test_type']
                )
            
            # Estimate duration if not found
            if not assessment['duration']:
                assessment['duration'] = self.estimate_duration(
                    assessment['name'],
                    assessment['test_type']
                )
        
        print(f"✓ Enhanced {len(self.assessments)} assessments")
    
    def generate_smart_description(self, name, test_types):
        """Generate intelligent description based on assessment name"""
        name_lower = name.lower()
        
        # Programming & Technical
        if 'python' in name_lower:
            return "Multi-choice assessment evaluating Python programming proficiency including core language features, data structures, object-oriented programming concepts, standard libraries, and database integration. For software development roles requiring Python expertise."
        elif 'java' in name_lower and 'javascript' not in name_lower:
            if 'entry' in name_lower:
                return "Entry-level Java programming assessment covering fundamental concepts including basic syntax, object-oriented principles, exception handling, and file I/O operations. For junior Java developer positions."
            elif 'advanced' in name_lower:
                return "Advanced Java development assessment measuring expertise in complex programming concepts including multithreading, generics, advanced collections, and design patterns. For senior Java roles."
            else:
                return "Comprehensive Java programming assessment evaluating knowledge of Java features including lambda expressions, stream API, collections framework, and modern development practices. For Java developer positions."
        elif 'javascript' in name_lower or 'js' in name_lower:
            return "JavaScript proficiency assessment covering ES6+ features, asynchronous programming, DOM manipulation, event handling, and modern JavaScript development practices. For front-end and full-stack developers."
        elif 'sql' in name_lower:
            return "SQL database assessment measuring query writing proficiency, database design principles, joins, filtering, aggregation techniques, and query optimization. For database developers and data analysts."
        elif 'c++' in name_lower or 'cpp' in name_lower:
            return "C++ programming assessment evaluating knowledge of object-oriented programming, memory management, templates, STL, and modern C++ features. For systems and application developers."
        elif 'c#' in name_lower or 'csharp' in name_lower:
            return "C# programming assessment measuring proficiency in .NET framework, object-oriented concepts, LINQ, async programming, and modern C# features. For .NET developers."
        elif '.net' in name_lower:
            return ".NET framework assessment evaluating knowledge of ASP.NET, MVC patterns, web services, Entity Framework, and modern .NET development practices. For .NET application developers."
        
        # Web Development
        elif 'html' in name_lower or 'css' in name_lower:
            return "Web development assessment evaluating HTML5 markup, CSS3 styling, responsive design principles, cross-browser compatibility, and modern front-end practices. For web developers and UI developers."
        elif 'selenium' in name_lower:
            return "Selenium test automation assessment evaluating knowledge of WebDriver architecture, test frameworks, element locators, wait mechanisms, and automation best practices. For QA automation engineers."
        elif 'drupal' in name_lower:
            return "Drupal CMS assessment measuring knowledge of Drupal architecture, content management, module development, theming, and site administration. For Drupal developers and CMS administrators."
        
        # Data & Analytics
        elif 'tableau' in name_lower:
            return "Tableau data visualization assessment evaluating skills in dashboard creation, interactive visualizations, calculations, filters, and data analysis using Tableau. For data analysts and BI professionals."
        elif 'excel' in name_lower:
            if 'essentials' in name_lower:
                return "Microsoft Excel fundamentals assessment covering basic formulas, data entry, cell formatting, simple calculations, and essential spreadsheet operations. For administrative positions."
            else:
                return "Advanced Microsoft Excel assessment measuring proficiency in complex formulas, data analysis tools, pivot tables, VLOOKUP, macros, and advanced data manipulation. For analyst roles."
        elif 'data warehousing' in name_lower:
            return "Data warehousing assessment evaluating understanding of ETL processes, dimensional modeling, star/snowflake schemas, OLAP systems, and data warehouse architecture. For data engineers and architects."
        
        # Testing & QA
        elif 'testing' in name_lower or 'qa' in name_lower:
            if 'manual' in name_lower:
                return "Manual software testing assessment measuring understanding of testing lifecycle, test case design, defect management, testing methodologies, and QA principles. For QA testers."
            else:
                return "Software testing assessment covering testing methodologies, test automation concepts, quality assurance practices, and testing tools. For QA professionals."
        elif 'automata' in name_lower:
            if 'fix' in name_lower:
                return "Hands-on coding assessment testing debugging skills through identifying and fixing code defects in multiple programming languages. For software developer roles."
            elif 'sql' in name_lower:
                return "Practical SQL coding assessment evaluating ability to write complex queries, perform database operations, and optimize query performance. For database developers."
            else:
                return "Automated coding assessment measuring practical programming skills through hands-on coding challenges and algorithm implementation. For software developers."
        
        # Personality & Behavior
        elif 'opq' in name_lower or 'occupational personality' in name_lower:
            return "Comprehensive occupational personality questionnaire measuring 32 behavioral dimensions across relationships, thinking styles, feelings, and work approaches. Industry-leading personality assessment."
        elif 'leadership' in name_lower:
            if 'report' in name_lower:
                return "Leadership assessment report analyzing leadership competencies, management style, strategic thinking, decision-making approaches, and influence skills. For management positions."
            else:
                return "Leadership evaluation measuring leadership capabilities, team management skills, strategic thinking, and executive presence. For leadership and management roles."
        elif 'team types' in name_lower:
            return "Team dynamics assessment examining preferred team roles, collaborative approaches, leadership flexibility, and contribution styles within teams. Based on personality profiling."
        
        # Cognitive Abilities
        elif 'verify' in name_lower or 'reasoning' in name_lower or 'ability' in name_lower:
            if 'verbal' in name_lower:
                return "Verbal reasoning assessment measuring reading comprehension, critical analysis, logical inference, argument evaluation, and text-based problem-solving. For professional roles."
            elif 'numerical' in name_lower:
                return "Numerical reasoning assessment evaluating ability to interpret numerical data, graphs, statistics, perform calculations, and solve quantitative problems. For analytical roles."
            elif 'inductive' in name_lower:
                return "Inductive reasoning assessment measuring pattern recognition, logical thinking, abstract reasoning, and capacity to generalize principles. For problem-solving roles."
            else:
                return "Cognitive ability assessment measuring reasoning skills and mental aptitude relevant to workplace performance. For professional competency evaluation."
        
        # Communication & Language
        elif 'english' in name_lower:
            if 'written' in name_lower:
                return "Written English proficiency assessment evaluating grammar, spelling, punctuation, sentence structure, and professional writing quality. For roles requiring written communication."
            elif 'spoken' in name_lower or 'svar' in name_lower:
                return "Spoken English evaluation measuring pronunciation, fluency, intonation, vocabulary, and verbal communication effectiveness. For customer-facing roles."
            else:
                return "English comprehension assessment measuring vocabulary, reading comprehension, grammar understanding, and language proficiency. For English language competency."
        elif 'communication' in name_lower:
            return "Communication skills assessment measuring verbal and non-verbal effectiveness, active listening, professional interaction, and workplace communication abilities. For collaborative roles."
        elif 'writing' in name_lower or 'email' in name_lower:
            return "Business writing assessment evaluating professional correspondence, email communication, persuasive writing, grammar, and effective written communication. For sales and professional roles."
        
        # Marketing & Digital
        elif 'marketing' in name_lower:
            return "Marketing knowledge assessment covering marketing principles, consumer behavior, brand management, digital channels, market research, and campaign management. For marketing professionals."
        elif 'seo' in name_lower:
            return "Search Engine Optimization assessment evaluating SEO fundamentals, keyword research, on-page/off-page optimization, link building, and search algorithms. For SEO specialists."
        elif 'advertising' in name_lower or 'adwords' in name_lower:
            return "Digital advertising assessment measuring expertise in online advertising platforms, campaign management, bid strategies, performance metrics, and ROI optimization. For digital marketers."
        
        # Office & Administrative
        elif 'computer literacy' in name_lower:
            return "Basic computer literacy assessment evaluating fundamental computer skills including OS navigation, file management, applications, internet, and email operations. For entry-level positions."
        elif 'accounting' in name_lower or 'bookkeeping' in name_lower:
            return "Accounting fundamentals assessment covering bookkeeping principles, financial transactions, ledger management, and basic accounting practices. For accounting and finance roles."
        
        # Global & Comprehensive
        elif 'global skills' in name_lower:
            return "Comprehensive global skills assessment evaluating 96 discrete behavioral dimensions across cognitive abilities, personality traits, and professional competencies. For diverse roles worldwide."
        
        # Default
        else:
            return f"Professional assessment measuring knowledge, skills, and competencies relevant to {name} positions. Evaluation tool designed to predict job performance and identify qualified candidates."
    
    def estimate_duration(self, name, test_types):
        """Estimate duration based on assessment characteristics"""
        name_lower = name.lower()
        
        # Personality assessments
        if 'Personality' in test_types or 'Behaviour' in ' '.join(test_types):
            if 'leadership' in name_lower and 'report' in name_lower:
                return 45
            return 25
        
        # Coding/Automata
        if 'automata' in name_lower or 'coding' in name_lower:
            return 45
        
        # Comprehensive assessments
        if 'global' in name_lower:
            return 90
        
        # Cognitive tests
        if 'verify' in name_lower or 'reasoning' in name_lower:
            if 'verbal' in name_lower or 'numerical' in name_lower:
                return 18
            return 25
        
        # Technical skills
        if 'advanced' in name_lower:
            return 45
        elif 'entry' in name_lower or 'essentials' in name_lower or 'basic' in name_lower:
            return 25
        
        # Default
        return 30
    
    def scrape_all(self):
        """Main scraping workflow"""
        print("\n" + "="*80)
        print("SHL INDIVIDUAL TEST SOLUTIONS SCRAPER")
        print("="*80)
        
        # Step 1: Crawl catalog to find all individual test URLs
        assessment_urls = self.crawl_catalog_pages()
        
        # If crawling didn't find enough, use smart fallback
        if len(assessment_urls) < 20:
            print("\n⚠️  Found fewer assessments than expected. Adding known individual tests...")
            fallback_urls = self.get_individual_test_fallback_urls()
            for url in fallback_urls:
                if url not in assessment_urls:
                    assessment_urls.append(url)
            print(f"✓ Total URLs after fallback: {len(assessment_urls)}")
        
        # Step 2: Scrape each assessment
        print("\n" + "="*80)
        print("SCRAPING INDIVIDUAL TEST DETAILS")
        print("="*80)
        
        for i, url in enumerate(assessment_urls, 1):
            print(f"\n[{i}/{len(assessment_urls)}] {url.split('/')[-2]}")
            assessment = self.extract_assessment_details(url)
            
            if assessment:
                self.assessments.append(assessment)
                print(f"   ✓ {assessment['name']}")
                print(f"   ✓ Test Type: {', '.join(assessment['test_type'])}")
            else:
                print(f"   ✗ Skipped (not an individual test or extraction failed)")
        
        # Step 3: Enhance data
        self.enhance_data()
        
        return self.assessments
    
    def get_individual_test_fallback_urls(self):
        """Fallback list of known INDIVIDUAL TEST URLs (not job solutions)"""
        base = "https://www.shl.com/solutions/products/product-catalog/view/"
        
        individual_tests = [
            # Programming Languages
            "python-new/", "java-8-new/", "core-java-entry-level-new/", "core-java-advanced-level-new/",
            "javascript-new/", "c-plus-plus-new/", "c-sharp-new/",
            
            # .NET Technologies
            "net-framework-4-5/", "net-mvc-new/", "net-mvvm-new/", "net-wcf-new/", 
            "net-wpf-new/", "net-xaml-new/", "ado-net-new/",
            
            # Databases & Data
            "sql-server-new/", "automata-sql-new/", "tableau-new/", "data-warehousing-concepts/",
            
            # Web Development
            "html-css-new/", "css3-new/", "javascript-new/", "drupal-new/",
            
            # Office Applications
            "microsoft-excel-365-new/", "microsoft-excel-365-essentials-new/",
            "microsoft-word-365-new/", "microsoft-powerpoint-365-new/",
            "microsoft-outlook-365-new/", "basic-computer-literacy-windows-10-new/",
            
            # Testing & QA
            "selenium-new/", "manual-testing-new/", "automata-fix-new/",
            
            # Marketing & Digital
            "marketing-new/", "digital-advertising-new/", "search-engine-optimization-new/",
            
            # Accounting & Finance
            "accounts-payable-new/", "accounts-receivable-new/", "bookkeeping-new/",
            
            # Communication & Language
            "interpersonal-communications/", "business-communication-adaptive/",
            "english-comprehension-new/", "written-english-v1/", 
            "svar-spoken-english-indian-accent-new/", "writex-email-writing-sales-new/",
            
            # Personality & Behavior
            "occupational-personality-questionnaire-opq32r/", "opq-leadership-report/",
            "opq-team-types-and-leadership-styles-report/", "enterprise-leadership-report-2-0/",
            
            # Cognitive Abilities
            "verify-verbal-ability-next-generation/", "verify-numerical-ability/",
            "shl-verify-interactive-inductive-reasoning/", "shl-verify-interactive-numerical-calculation/",
            
            # Comprehensive
            "global-skills-assessment/"
        ]
        
        return [base + test for test in individual_tests]
    
    def save_to_csv(self, filename='shl_individual_tests.csv'):
        """Save to CSV"""
        if not self.assessments:
            print("No assessments to save!")
            return None
        
        df = pd.DataFrame(self.assessments)
        df['test_type'] = df['test_type'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        columns = ['name', 'url', 'description', 'test_type', 'duration', 'adaptive_support', 'remote_support']
        df = df[columns]
        
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n✓ Saved {len(df)} assessments to {filename}")
        return df
    
    def save_to_json(self, filename='shl_individual_tests.json'):
        """Save to JSON"""
        if not self.assessments:
            print("No assessments to save!")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.assessments, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(self.assessments)} assessments to {filename}")


if __name__ == "__main__":
    scraper = SHLIndividualTestScraper()
    assessments = scraper.scrape_all()
    
    print("\n" + "="*80)
    print("SAVING DATA")
    print("="*80)
    
    df = scraper.save_to_csv('shl_individual_tests.csv')
    scraper.save_to_json('shl_individual_tests.json')
    
    print("\n" + "="*80)
    print("✅ SCRAPING COMPLETE!")
    print("="*80)
    print(f"\n📊 Total Individual Test Assessments: {len(assessments)}")
    print(f"📄 CSV: shl_individual_tests.csv")
    print(f"📄 JSON: shl_individual_tests.json")
    
    if df is not None and len(df) > 0:
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(f"\nTest Type Distribution:")
        print(df['test_type'].value_counts().head(10))
        print(f"\nDuration Statistics:")
        print(f"  Average: {df['duration'].mean():.1f} minutes")
        print(f"  Min: {df['duration'].min()} min")
        print(f"  Max: {df['duration'].max()} min")
        print(f"\nAdaptive Support: {(df['adaptive_support'] == 'Yes').sum()} tests")
        print(f"Remote Support: {(df['remote_support'] == 'Yes').sum()} tests")
