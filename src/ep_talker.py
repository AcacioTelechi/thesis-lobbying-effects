import requests
from rdflib import Graph

class EuropeanParliamentTalker:
    def __init__(self):
        """Initialize the EuropeanParliamentTalker class."""
        self.graph = Graph()

    def download_rdf(self, rdf_url: str):
        """Download RDF data and load it into the graph."""
        try:
            response = requests.get(rdf_url)
            response.raise_for_status()

            # Determine the format from the Content-Type header
            content_type = response.headers.get('Content-Type', '').lower()
            if 'rdf+xml' in content_type:
                format_ = "xml"
            elif 'turtle' in content_type:
                format_ = "turtle"
            elif 'json+ld' in content_type:
                format_ = "json-ld"
            else:
                format_ = "xml"  # Default to RDF/XML

            # Parse the RDF data into the graph
            self.graph.parse(data=response.text, format=format_)
            # print(f"RDF data loaded successfully. Total triples: {len(self.graph)}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading RDF data: {e}")
            return
        except Exception as e:
            print(f"Error parsing RDF data: {e}")
            return

    def query(self, sparql_query: str):
        """Run a SPARQL query on the RDF graph."""
        try:
            result = self.graph.query(sparql_query)
            if result:
                return result
            else:
                return []
        except Exception as e:
            print(f"Error executing SPARQL query: {e}")
            return []
