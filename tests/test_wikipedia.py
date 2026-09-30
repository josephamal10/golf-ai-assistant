from app.ingestion.wikipedia import is_disambiguation


def test_disambiguation_pages_are_detected_from_pageprops() -> None:
    # Shape of the MediaWiki response for 'Fairway' with prop=pageprops&ppprop=disambiguation.
    assert is_disambiguation({'title': 'Fairway', 'pageprops': {'disambiguation': ''}})


def test_regular_articles_are_not_disambiguation() -> None:
    assert not is_disambiguation({'title': 'Golf course'})
    assert not is_disambiguation({'title': 'Golf course', 'pageprops': {'wikibase_item': 'Q1048525'}})
