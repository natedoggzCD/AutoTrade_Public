from autotrade.utils.news_relevance import (
    assess_news_relevance,
    count_ticker_mentions,
    extract_target_snippet,
)


def test_ticker_mentions_do_not_match_partial_symbols():
    text = "AMCX rallied while AMC fell. The AMC CEO bought shares."

    assert count_ticker_mentions(text, "AMC") == 2


def test_multi_stock_roundup_extracts_target_bullet(monkeypatch):
    monkeypatch.setattr(
        "autotrade.utils.news_relevance.get_company_aliases",
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

    snippet = extract_target_snippet("AMC", "Company News for May 21, 2026", body)
    relevance = assess_news_relevance("AMC", "Company News for May 21, 2026", body)

    assert "AMC Entertainment" in snippet
    assert "Intuit" not in snippet
    assert relevance.label == "relevant"


def test_zero_mention_article_is_irrelevant():
    relevance = assess_news_relevance(
        "PFE",
        "Fed keeps rates unchanged",
        "Federal Reserve officials discussed inflation and rate policy.",
    )

    assert relevance.label == "irrelevant"
    assert relevance.score == 0.0


def test_comparison_article_is_tangential(monkeypatch):
    monkeypatch.setattr(
        "autotrade.utils.news_relevance.get_company_aliases",
        lambda ticker: ("Advanced Micro Devices",) if ticker == "AMD" else (),
    )
    relevance = assess_news_relevance(
        "AMD",
        "Nvidia vs AMD vs Intel",
        "Nvidia vs AMD vs Intel: which stock is the better AI chip buy this year?",
    )

    assert relevance.label == "tangential"


def test_investors_does_not_trigger_vs_tangential_keyword(monkeypatch):
    monkeypatch.setattr(
        "autotrade.utils.news_relevance.get_company_aliases",
        lambda ticker: ("TotalEnergies",) if ticker == "TTE" else (),
    )
    relevance = assess_news_relevance(
        "TTE",
        "Is It Time To Reassess TotalEnergies (TTE)?",
        "Investors are reassessing TotalEnergies after recent share price moves.",
    )

    assert relevance.label == "relevant"


def test_repeated_company_mentions_override_sector_context(monkeypatch):
    monkeypatch.setattr(
        "autotrade.utils.news_relevance.get_company_aliases",
        lambda ticker: ("Rayonier",) if ticker == "RYN" else (),
    )
    relevance = assess_news_relevance(
        "RYN",
        "Is Rayonier (RYN) Pricing Fair?",
        "Rayonier is being reassessed in the REIT sector. Rayonier shares rose.",
    )

    assert relevance.label == "relevant"
