class EpScraper:

    BASE_URL = 'https://www.europarl.europa.eu'
    def __init__(self) -> None:
        pass

    def get_meetings_between_dates(self, init_date:str, end_date:str) -> list[dict]:
        """
        date format: %d/%m/%Y
        """
        url = f'/meps/en/search-meetings?textualSearch=&fromDate={init_date}&toDate={end_date}'