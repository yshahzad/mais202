import requests
import xml.etree.ElementTree as ET
import pandas as pd
import random
import time
import os

SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def make_request(url, params, retries=3, wait_time=60):
    """
    Makes an API request with automatic retries for errors or rate limits.
    """
    for attempt in range(retries):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            if "API rate limit exceeded" in response.text:
                print(f"⚠️ API rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue  # Retry after waiting
            return response  # Successful response
        else:
            print(f"⚠️ API request failed (attempt {attempt+1}/{retries}): {response.status_code}")
            time.sleep(5)  # Wait before retrying
    print("❌ Max retries reached. Skipping request.")
    return None  # Failed request after all retries

def chunked_search_pubmed(query, start_year=1900, end_year=2025):
    """
    Retrieves ALL PubMed PMIDs year-by-year, handling rate limits and large queries.
    """
    all_pmids = []

    for year in range(start_year, end_year + 1):
        print(f"🔎 Searching year {year} ...")
        year_pmids = []
        retstart = 0
        batch_size = 100

        while True:
            term_str = f"{query} AND {year}[PDAT]"
            params = {
                "db": "pubmed",
                "term": term_str,
                "retmode": "xml",
                "retmax": batch_size,
                "retstart": retstart
            }

            response = make_request(SEARCH_URL, params)
            if response is None:
                break  # Skip this year if all retries failed

            if not response.text.startswith("<?xml"):
                print(f"❌ Invalid API response (not XML) for {year}. Skipping this year.")
                break  # Skip year if response is not valid XML

            root = ET.fromstring(response.text)
            batch_pmids = [id_tag.text for id_tag in root.findall(".//Id")]

            if not batch_pmids:
                break  # No more results for this year

            year_pmids.extend(batch_pmids)
            retstart += batch_size
            time.sleep(0.5)  # Prevent API rate-limiting

            if len(year_pmids) >= 10000:
                print(f"⚠️ Reached 10k limit for year {year}. Moving on...")
                break

        print(f"✅ Found {len(year_pmids)} PMIDs for {year}.")
        all_pmids.extend(year_pmids)

    print(f"📊 Total PMIDs collected: {len(all_pmids)}")
    return all_pmids

def fetch_titles_dates_authors(pmids):
    """
    Fetches article details (Title, Date, Authors, Abstract) for each PMID.
    """
    results = []
    batch_size = 100

    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i:i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch_pmids),
            "retmode": "xml"
        }

        response = make_request(FETCH_URL, params)
        if response is None or not response.text.startswith("<?xml"):
            print(f"❌ Skipping batch due to invalid API response.")
            continue  # Skip batch if response is invalid

        root = ET.fromstring(response.text)

        for article in root.findall(".//PubmedArticle"):
            title_tag = article.find(".//ArticleTitle")
            title = title_tag.text if title_tag is not None else "N/A"

            # Extract publication date
            pub_date_tag = article.find(".//PubDate")
            if pub_date_tag is not None:
                year = pub_date_tag.find("Year")
                month = pub_date_tag.find("Month")
                day = pub_date_tag.find("Day")
                pub_date = f"{year.text if year is not None else 'Unknown'}-{month.text if month is not None else ''}-{day.text if day is not None else ''}".strip("-")
            else:
                pub_date = "Unknown"

            # Extract authors
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")

            author_list = ", ".join(authors) if authors else "No authors listed"

            # Extract abstract
            abstract_tag = article.find(".//AbstractText")
            abstract_text = abstract_tag.text if abstract_tag is not None else "No abstract available"

            results.append({
                "Title": title,
                "Date": pub_date,
                "Authors": author_list,
                "Abstract": abstract_text
            })

        time.sleep(0.5)  # Prevent API rate-limiting

    return results

def scrape_pubmed(keyword, sample_size=200, start_year=1900, end_year=2025):
    """
    1) Retrieve PMIDs year-by-year while handling rate limits.
    2) Shuffle & sample PMIDs from across all years.
    3) Fetch metadata and save to CSV.
    """
    all_pmids = chunked_search_pubmed(keyword, start_year, end_year)

    if not all_pmids:
        print("❌ No PMIDs found. Exiting.")
        return None

    # Shuffle & pick a random sample
    random.shuffle(all_pmids)
    selected_pmids = all_pmids[:sample_size]
    print(f"🔀 Selected {len(selected_pmids)} random PMIDs.")

    # Fetch article details
    articles = fetch_titles_dates_authors(selected_pmids)

    # Save to CSV
    df = pd.DataFrame(articles)
    csv_filename = "pubmed_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"📂 Results saved to {csv_filename}")

    # Open CSV automatically (Mac)
    os.system(f"open {csv_filename}")

    return csv_filename

# Example usage
if __name__ == "__main__":
    keyword = "Epigenetics"
    scrape_pubmed(keyword, sample_size=20000, start_year=1980, end_year=2025)
