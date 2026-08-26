from scripts.audit_reference_table_groups import _coverage, _numbers, _words


def test_audit_normalizes_financial_numbers() -> None:
    values = _numbers("Revenue was $8,649 and margin was (28)% in 2022.")

    assert values == {"8649", "28%", "2022"}


def test_audit_reads_fiscal_year_without_partial_number() -> None:
    assert _numbers("FY2022 and FY2023") == {"2022", "2023"}


def test_audit_does_not_treat_date_comma_as_thousands_separator() -> None:
    assert _numbers("December 31,2022") == {"31", "2022"}
    assert _numbers("Revenue was 1,001,425") == {"1001425"}


def test_audit_coverage_is_fractional() -> None:
    assert _coverage({"a", "b"}, {"b", "c"}) == 0.5
    assert _coverage(set(), {"b"}) == 1.0


def test_audit_words_ignore_short_noise() -> None:
    assert _words("The FY net income") == {"the", "net", "income"}
