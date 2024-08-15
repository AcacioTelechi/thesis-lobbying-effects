import requests

class EpApiTalker:
    BASE_URL="https://data.europarl.europa.eu/api/v2/"

    def __init__(self) -> None:
        pass

    def request(self, url) -> list[dict]:
        try:
            resp = requests.get(url)
            return resp.json()
        except Exception as e:
            raise e
        
    def get_meps(self)->list[dict]:
        url = f"{self.BASE_URL}/meps?format=application%2Fld%2Bjson"
        resp = self.request(url)
        return resp['data']

    def get_mep_details(self, mep_id:int|str) -> dict:
        url = f"{self.BASE_URL}/meps/{mep_id}?format=application%2Fld%2Bjson"
        resp = self.request(url)
        return resp['data']