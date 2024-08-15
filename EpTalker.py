import requests
import csv
from io import StringIO
from bs4 import BeautifulSoup


class EpTalker:
    BASE_URL = "https://data.europarl.europa.eu/api/v2/"
    WEB_URL = "https://www.europarl.europa.eu"

    def __init__(self) -> None:
        pass

    def request(self, url) -> list[dict]:
        try:
            resp = requests.get(url)
            return resp.json()
        except Exception as e:
            raise e

    def get_meps(self) -> list[dict]:
        url = f"{self.BASE_URL}/meps?format=application%2Fld%2Bjson"
        resp = self.request(url)
        return resp["data"]

    def get_mep_details(self, mep_id: int | str) -> dict:
        url = f"{self.BASE_URL}/meps/{mep_id}?format=application%2Fld%2Bjson"
        resp = self.request(url)
        return resp["data"]

    def scrape_meetings(self, init_date: str, end_date:str) -> list[dict]:
        url= f"{self.WEB_URL}/meps/en/search-meetings?textualSearch=&fromDate={init_date}&toDate={end_date}"

        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, "html.parser")

        csv_ = requests.get(soup.find(title="CSV")["href"]).content
        
        content = csv_.decode()
        file = StringIO(content)
        csv_data = csv.reader(file, delimiter=",")
        data = list(csv_data)

        headers = data[0]

        return [{headers[i]: j for i, j in enumerate(d)} for d in data[1:]]
