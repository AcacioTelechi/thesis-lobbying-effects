import re
import unicodedata
from datetime import datetime, timedelta
import calendar


def month_range(start_date, end_date):
    """Generate start and end dates for each month between start_date and end_date."""
    start_date = datetime.strptime(start_date, "%d/%m/%Y")
    end_date = datetime.strptime(end_date, "%d/%m/%Y")

    current_date = start_date

    while current_date <= end_date:
        month_start = current_date.replace(day=1)
        _, last_day = calendar.monthrange(current_date.year, current_date.month)

        month_end = current_date.replace(day=last_day)

        if month_end > end_date:
            month_end = end_date

        yield (month_start.strftime("%d/%m/%Y"), month_end.strftime("%d/%m/%Y"))

        current_date = month_end + timedelta(days=1)


def treat_string(string: str):
    # Convert to lowercase
    name = str(string).lower()

    # Remove leading/trailing spaces
    name = name.strip()

    # Replace common abbreviations
    abbreviations = {
        "inc.": "incorporated",
        "ltd.": "limited",
        "eu": "european union",
        "intl.": "international",
        "corp.": "corporation",
        "&": "and",
    }
    for abbr, full in abbreviations.items():
        name = re.sub(r"\b" + re.escape(abbr) + r"\b", full, name)

    # Remove special characters and extra symbols (dots, commas, etc.)
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)

    # Normalize unicode characters (handle accents)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")

    # Collapse multiple spaces into a single space
    name = re.sub(r"\s+", " ", name)

    return name
