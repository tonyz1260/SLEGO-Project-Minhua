import arxiv
import csv
from typing import List, Dict
import requests

def download_papers_from_arxiv_csv(filename: str = "dataspace/latest_papers.csv", 
                                    download_folder: str = "dataspace/papers/") -> None:
    """
    Downloads papers listed in a CSV file from arXiv.

    Args:
    filename (str): The path to the CSV file containing paper metadata.
    download_folder (str): The directory where the downloaded papers will be saved.

    The CSV file must contain at least the 'url' and 'title' fields.
    """

    try:
        # Create a directory to save downloaded papers
        os.makedirs(download_folder, exist_ok=True)
        
        # Read the CSV file to get the URLs
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                url = row['url']
                paper_title = row['title'].replace('/', '_')  # Replace any slashes in title to avoid file path errors
                file_path = f"{download_folder}{paper_title}.pdf"
                
                # Download the paper
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded '{paper_title}' successfully.")
                else:
                    print(f"Failed to download '{paper_title}'. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def search_arxiv_papers(search_query: str='machine learning', 
                           filename: str = "dataspace/latest_papers.csv", 
                           max_results: int = 5):

    """
    Searches for papers on arXiv and saves the results to a CSV file.

    Args:
    search_query (str): The query term to search for on arXiv.
    filename (str): Path to save the CSV file containing the search results.
    max_results (int): Maximum number of results to fetch and save.

    Returns:
    list: A list of dictionaries, each containing paper details.

    The function saves a CSV with fields: title, authors, abstract, published, url.
    """

    # Create a search object
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    # Fetch the results
    results = list(search.results())
    
    # Format the results into a readable list and write to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ["title", "authors", "abstract", "published", "url"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            paper_info = {
                "title": result.title,
                "authors": ', '.join(author.name for author in result.authors),
                "abstract": result.summary.replace('\n', ' '),
                "published": result.published.strftime('%Y-%m-%d'),
                "url": result.entry_id
            }
            writer.writerow(paper_info)
    return results
