import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time  # Prevents API rate limiting

# Base URLs for PubMed API (e-utilities)
SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def search_pubmed(query, max_results=10000):
    """Searches PubMed and returns a list of PMIDs, handling pagination properly."""
    pmids = []
    batch_size = 100  # PubMed allows up to 100 results per request
    retstart = 0  # Offset for pagination

    while len(pmids) < max_results:
        remaining = max_results - len(pmids)
        retmax = min(batch_size, remaining)  # Fetch up to batch_size but no more than needed

        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": retmax,  # Number of results per batch
            "retstart": retstart  # Start index for pagination
        }

        response = requests.get(SEARCH_URL, params=params)
        root = ET.fromstring(response.text)

        # Extract PMIDs
        batch_pmids = [id_tag.text for id_tag in root.findall(".//Id")]
        if not batch_pmids:
            break  # No more results to fetch

        pmids.extend(batch_pmids)
        retstart += retmax  # Move to next batch

        time.sleep(0.5)  # Prevent API rate limits

    return pmids[:max_results]

def fetch_titles_dates_authors(pmids):
    """Fetches article details and retains all results, even if abstracts are missing."""
    if not pmids:
        return []

    results = []
    batch_size = 100  # PubMed allows fetching up to 100 articles at once

    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i:i+batch_size]  # Process in batches of 100
        params = {"db": "pubmed", "id": ",".join(batch_pmids), "retmode": "xml"}
        response = requests.get(FETCH_URL, params=params)
        root = ET.fromstring(response.text)

        # Loop through all articles in the XML
        for article in root.findall(".//PubmedArticle"):
            title = article.find(".//ArticleTitle")
            title = title.text if title is not None else "N/A"

            # Extract publication date (Year, Month, Day if available)
            pub_date_tag = article.find(".//PubDate")
            if pub_date_tag is not None:
                year = pub_date_tag.find("Year").text if pub_date_tag.find("Year") is not None else "Unknown"
                month = pub_date_tag.find("Month").text if pub_date_tag.find("Month") is not None else ""
                day = pub_date_tag.find("Day").text if pub_date_tag.find("Day") is not None else ""
                pub_date = f"{year}-{month}-{day}".strip("-")  # Format as YYYY-MM-DD
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

            # Extract abstract (if available)
            abstract_tag = article.find(".//AbstractText")
            abstract_text = abstract_tag.text if abstract_tag is not None else "No abstract available"

            # Add the article to results, even if it doesn't have an abstract
            results.append({
                "Title": title,
                "Date": pub_date,
                "Authors": author_list,
                "Abstract": abstract_text
            })

        time.sleep(0.5)  # Prevent API rate limits

    return results

def scrape_pubmed(keyword, max_results=10000):
    """Finds articles and returns a DataFrame (including articles without abstracts)."""
    pmids = search_pubmed(keyword, max_results)  # Fetch PMIDs using PubMed API
    print(f"Total PMIDs Retrieved: {len(pmids)}")  # Debugging step
    articles = fetch_titles_dates_authors(pmids)  # Retrieve all article details
    return pd.DataFrame(articles)

# Example usage
keyword = "machine learning"
articles_df = scrape_pubmed(keyword, max_results=200)

# Save results to a CSV file
articles_df.to_csv("pubmed_results.csv", index=False)
print("Results saved to pubmed_results.csv")

# Optional: Open the file automatically (Mac only)
import os
os.system("open pubmed_results.csv")
