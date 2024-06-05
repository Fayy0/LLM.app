import logging
from bs4 import BeautifulSoup
import chromadb
import requests
from sentence_transformers import SentenceTransformer

class TextIndexer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection_name = 'text_collection'
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def scrape_and_index(self, url, base_url="https://u.ae/en/information-and-services"):
        if not url.startswith(base_url):
            logging.warning(f"Skipping URL not starting with base URL: {url}")
            return

        try:
            logging.info(f"Scraping URL: {url}")
            page_content = requests.get(url).content
            soup = BeautifulSoup(page_content, 'html.parser')
            texts = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
            logging.info(f"Scraped texts from {url}: {texts}")

            if texts:
                embeddings = self.model.encode(texts)
                self.collection.upsert(documents=texts, embeddings=embeddings, ids=[f"{url}-{i}" for i in range(len(texts))])
                logging.info(f"Added {len(texts)} texts from {url}")

            for link in soup.find_all('a', href=True):
                sub_url = link['href']
                if sub_url.startswith('/'):
                    sub_url = "https://u.ae" + sub_url
                if sub_url.startswith(base_url):
                    self.scrape_and_index(sub_url, base_url)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to scrape {url}: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error during scraping {url}: {str(e)}")

    def search(self, query, k=5):
        try:
            query_embedding = self.model.encode([query])
            results = self.collection.query(query_embeddings=query_embedding, n_results=k)
            logging.debug(f"Search results: {results}")
            return results
        except Exception as e:
            logging.error(f"Search query failed: {str(e)}")
            return None
