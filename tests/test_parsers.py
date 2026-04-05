import pytest
from src.ingestion.parsers import MultimodalParser
from src.exceptions import IngestionError

@pytest.mark.fast
def test_multimodal_parser_invalid_pdf():
    parser = MultimodalParser()
    with pytest.raises(IngestionError, match="Falla al extraer datos del PDF"):
        parser.parse_pdf("no_existo.pdf")

@pytest.mark.fast
def test_multimodal_parser_invalid_excel():
    parser = MultimodalParser()
    with pytest.raises(IngestionError, match="Falla al extraer datos del Excel"):
        parser.parse_excel("no_existo.xlsx")
