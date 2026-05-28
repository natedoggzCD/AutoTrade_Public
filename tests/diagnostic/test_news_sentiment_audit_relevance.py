from tools.news_sentiment_audit import (
    _count_ticker_mentions,
    _extract_target_snippet,
    assess_article_relevance,
)


def test_ticker_mentions_do_not_match_partial_symbols():
    text = "AMCX rallied while AMC fell. The AMC CEO bought shares."

    assert _count_ticker_mentions(text, "AMC") == 2


def test_multi_stock_roundup_extracts_target_bullet(monkeypatch):
    monkeypatch.setattr(
        "tools.news_sentiment_audit.get_company_aliases",
        lambda ticker: ("AMC Entertainment",) if ticker == "AMC" else (),
    )
    body = """
    -
    Intuit Inc. (INTU) shares fell 4% after plans to cut 17% of its workforce.
    -
    AMC Entertainment Holdings, Inc. (AMC) shares jumped 11.8% after CEO Adam Aron
    disclosed the purchase of 250,000 shares.
    -
    Lowe's Companies, Inc. (LOW) rose after earnings beat estimates.
    """

    snippet = _extract_target_snippet("AMC", "Company News for May 21, 2026", body)
    relevance = assess_article_relevance("AMC", "Company News for May 21, 2026", body)

    assert "AMC Entertainment" in snippet
    assert "Intuit" not in snippet
    assert relevance.label == "relevant"
    assert relevance.body_ticker_mentions == 2
    assert relevance.company_mentions == 1


def test_zero_mention_article_is_irrelevant():
    body = "Federal Reserve officials discussed inflation and rate policy."

    relevance = assess_article_relevance("PFE", "Fed keeps rates unchanged", body)

    assert relevance.label == "irrelevant"
    assert relevance.score == 0.0
    assert relevance.reason == "no_ticker_or_company_mentions"


def test_tangential_spacex_article_is_not_relevant_tsla_news(monkeypatch):
    monkeypatch.setattr(
        "tools.news_sentiment_audit.get_company_aliases",
        lambda ticker: ("Tesla",) if ticker == "TSLA" else (),
    )
    body = """
    SpaceX is reportedly preparing for an IPO discussion with investors.
    Tesla CEO Elon Musk also runs SpaceX, and some Tesla shareholders watch the
    private space company closely.
    """

    relevance = assess_article_relevance("TSLA", "SpaceX IPO speculation grows", body)

    assert relevance.label == "tangential"
    assert relevance.company_mentions == 2
    assert "SpaceX" in relevance.snippet


def test_comparison_article_is_tangential_without_material_target_context(monkeypatch):
    monkeypatch.setattr(
        "tools.news_sentiment_audit.get_company_aliases",
        lambda ticker: ("Advanced Micro Devices",) if ticker == "AMD" else (),
    )
    body = "Nvidia vs AMD vs Intel: which stock is the better AI chip buy this year?"

    relevance = assess_article_relevance("AMD", "Nvidia vs AMD vs Intel", body)

    assert relevance.label == "tangential"
    assert relevance.body_ticker_mentions == 1
