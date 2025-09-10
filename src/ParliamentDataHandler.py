"""
This module provides the `EpTalker` class for interacting with the European Parliament's data API and scraping MEPs' meeting data.

Key Features:
- Retrieve a list of Members of the European Parliament (MEPs).
- Fetch detailed information about individual MEPs.
- Scrape meetings data for a specified date range.
- Parallelize scraping across multiple months for efficiency.
"""

from datetime import datetime, timedelta
import time
import requests
import csv
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.utils import month_range


class ParliamentDataHandler:
    """
    The `EpTalker` class provides methods for interacting with the European Parliament's API and scraping meeting data
    from the European Parliament's website.
    """

    BASE_URL = "https://data.europarl.europa.eu/api/v2"
    WEB_URL = "https://www.europarl.europa.eu"

    def __init__(self) -> None:
        pass

    def request(self, url) -> list[dict] | None:
        try:
            resp = requests.get(url)
            return resp.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return None
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            return None
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            return None
        except requests.exceptions.RequestException as req_err:
            print(f"An error occurred: URL - {url} ERROR: {req_err}")
            return None
        except ValueError as json_err:
            print(f"JSON decode error occurred: {json_err}")
            return None

    def get_meps(self) -> list[dict]:
        url = f"{self.BASE_URL}/meps?format=application%2Fld%2Bjson"
        resp = self.request(url)
        return resp["data"]

    def get_questions(self, limit: int = 1000, offset: int = 0) -> list[dict]:
        url = f"{self.BASE_URL}/parliamentary-questions?format=application%2Fld%2Bjson&offset={offset}&limit={limit}"
        resp = self.request(url)
        if not resp or "data" not in resp:
            return []

        data = resp["data"]
        if len(data) < limit:
            return data
        else:
            return data + self.get_questions(limit=limit, offset=offset + limit)
        
    def get_questions_details(self, question_id) -> list[dict]:
        url = f"{self.BASE_URL}/parliamentary-questions/{question_id}?format=application%2Fld%2Bjson&language=en"
        resp = self.request(url)
        if not resp or "data" not in resp:
            return []
        return resp["data"]
        
    def get_procedures(self, limit: int = 1000, offset: int = 0) -> list[dict]:
        url = f"{self.BASE_URL}/procedures?format=application%2Fld%2Bjson&offset={offset}&limit={limit}"
        resp = self.request(url)
        if not resp or "data" not in resp:
            return []

        data = resp["data"]
        if len(data) < limit:
            return data
        else:
            return data + self.get_procedures(limit=limit, offset=offset + limit)

    def get_procedure_details(self, procedure_id: str) -> list[dict]:
        url = (
            f"{self.BASE_URL}/procedures/{procedure_id}?format=application%2Fld%2Bjson"
        )
        resp = self.request(url)
        if not resp or "data" not in resp:
            return []

        return resp["data"]

    def get_documents(self, limit: int = 1000, offset: int = 0) -> list[dict]:
        url = f"{self.BASE_URL}/documents?format=application%2Fld%2Bjson&offset={offset}&limit={limit}"
        resp = self.request(url)
        if not resp or "data" not in resp:
            return []

        data = resp["data"]
        if len(data) < limit:
            return data
        else:
            return data + self.get_documents(limit=limit, offset=offset + limit)

    def get_mep_details(self, mep_id: int | str) -> dict | None:
        """
        Fetches detailed information about a specific MEP using their ID, with enhanced error handling.

        Args:
            mep_id (int | str): The ID of the MEP for which details are to be fetched.

        Returns:
            dict: A dictionary containing the details of the MEP, or None if an error occurs.
        """
        url = f"{self.BASE_URL}/meps/{mep_id}?format=application%2Fld%2Bjson"
        response = self.request(url)

        if response is None:
            print(f"Failed to retrieve details for MEP ID {mep_id}. URL: {url}")
            return None

        return response.get("data", {})

    def get_mep_details_in_parallel(
        self, mep_ids: None | list[int | str] = None, max_workers: int = 10
    ) -> list[dict]:
        """
        Fetches details for multiple MEPs in parallel using ThreadPoolExecutor.

        Args:
            mep_ids (list[int | str]): A list of MEP IDs for which details are to be fetched.
            max_workers (int, optional): Maximum number of threads to use for parallel processing. Defaults to 10.

        Returns:
            list[dict]: A list of dictionaries containing the details of each MEP.
        """
        if not mep_ids:
            meps = self.get_meps()
            mep_ids = [m["identifier"] for m in meps]

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_mep_id = {
                executor.submit(self.get_mep_details, mep_id): mep_id
                for mep_id in mep_ids
            }

            for future in tqdm(
                as_completed(future_to_mep_id),
                total=len(mep_ids),
                desc="Fetching MEP details",
            ):
                mep_id = future_to_mep_id[future]
                try:
                    data = future.result()
                    results += data
                except Exception as exc:
                    print(f"Failed to fetch details for MEP ID {mep_id}: {exc}")

        return results

    def scrape_meetings(self, init_date: str, end_date: str) -> list[dict]:
        url = f"{self.WEB_URL}/meps/en/search-meetings?textualSearch=&fromDate={init_date}&toDate={end_date}"

        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, "html.parser")

        csv_ = requests.get(soup.find(title="CSV")["href"]).content

        content = csv_.decode()
        file = StringIO(content)
        csv_data = csv.reader(file, delimiter=",")
        data = list(csv_data)

        headers = data[0]

        return [{headers[i]: j for i, j in enumerate(d)} for d in data[1:]]

    def scrape_meetings_in_parallel(self, start_date, end_date, max_workers=10):
        """Parallelize the scraping process over months using ThreadPoolExecutor."""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(self.scrape_meetings, month_start, month_end): (
                    month_start,
                    month_end,
                )
                for month_start, month_end in month_range(start_date, end_date)
            }

            for future in tqdm(as_completed(future_to_date)):
                start, end = future_to_date[future]
                try:
                    data = future.result()
                    results += data
                except Exception as exc:
                    print(f"Scraping failed for {start} to {end}: {exc}")

        return results

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = []
        
    def wait_if_needed(self):
        now = datetime.now()
        # Remove requests older than time window
        self.requests = [t for t in self.requests if now - t < timedelta(seconds=self.time_window)]
        
        if len(self.requests) >= self.max_requests:
            # Wait until oldest request is outside time window
            sleep_time = (self.requests[0] + timedelta(seconds=self.time_window) - now).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.requests = self.requests[1:]
        
        self.requests.append(now)