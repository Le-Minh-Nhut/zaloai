"""ITN tests: timings, money, cardinals, dates, percents, guards. Stdlib only.

Run with: python3 tests/run_tests.py test_itn.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lifemate.asr.normalization import int_to_vietnamese_words, to_llm_text  # noqa: E402
from lifemate.text.itn import (  # noqa: E402
    apply_itn_pipeline,
    itn_rules,
    normalize_itn,
    parse_vi_cardinal,
)

def _itn(text: str) -> str:
    return normalize_itn(text, enabled=True).text

# ------------------------------------------------------------------- time

def test_time_table_period_markers():
    """Every marker resolves its own 12h window; the table is the spec."""
    cases = {
        "bảy giờ ba mươi sáng": "07:30",
        "mười giờ sáng": "10:00",
        "mười một giờ ba mươi trưa": "11:30",
        "mười hai giờ trưa": "12:00",
        "ba giờ chiều": "15:00",
        "năm giờ mười lăm chiều": "17:15",
        "bảy giờ ba mươi tối": "19:30",
        "mười giờ đêm": "22:00",
        "hai mươi ba giờ đêm": "23:00",
        "bảy rưỡi tối": "19:30",
        "bảy giờ kém mười lăm sáng": "06:45",
        "mười chín giờ ba mươi": "19:30",
        "hai mươi ba giờ": "23:00",
        "19h30": "19:30",
    }
    for spoken, expected in cases.items():
        assert _itn(spoken) == expected, f"{spoken!r} -> {_itn(spoken)!r}, expected {expected!r}"

def test_time_without_marker_is_not_guessed():
    """7:30 with no period marker could be AM or PM: the digits stay spoken."""
    result = normalize_itn("đặt báo thức bảy giờ ba mươi", enabled=True)
    assert result.text == "đặt báo thức bảy giờ ba mươi"
    assert result.changed is False
    assert [c.rule for c in result.changes] == ["skipped_ambiguous"], result.changes
    assert result.changes[0].replacement == result.changes[0].original

def test_time_digits_are_only_converted_when_unambiguous():
    assert _itn("nhắc 7h30") == "nhắc 7h30"
    assert _itn("nhắc 19h30") == "nhắc 19:30"
    assert _itn("nhắc 7 giờ 30") == "nhắc 7 giờ 30"

def test_time_outside_the_marker_window_is_skipped():
    """A spoken hour outside its marker's 24h window (trưa 11..13, chiều 12..17)
    is never guessed: ``bảy giờ chiều`` stays words."""
    for text in ("bảy giờ ba mươi trưa", "bảy giờ ba mươi chiều"):
        result = normalize_itn(text, enabled=True)
        assert result.text == text, result.text
        assert [c.rule for c in result.changes] == ["skipped_out_of_window"], result.changes


def test_time_out_of_window_is_skipped_not_guessed():
    """``mười chín giờ sáng`` contradicts itself: record and leave it alone."""
    result = normalize_itn("mười chín giờ sáng", enabled=True)
    assert result.text == "mười chín giờ sáng"
    assert [c.rule for c in result.changes] == ["skipped_out_of_window"], result.changes

# ------------------------------------------------------------------ money

def test_money_each_marker():
    cases = {
        "chuyển năm trăm nghìn đồng": "chuyển 500.000đ",
        "chuyển năm trăm nghìn vnđ": "chuyển 500.000đ",
        "chuyển năm trăm nghìn đ": "chuyển 500.000đ",
        "chuyển 500 nghìn": "chuyển 500.000đ",
        "chuyển 500k": "chuyển 500.000đ",
        "chuyển 500.000đ": "chuyển 500.000đ",
        "chuyển 2 triệu": "chuyển 2.000.000đ",
        "giá hai mươi lăm nghìn": "giá 25.000đ",
    }
    for spoken, expected in cases.items():
        assert _itn(spoken) == expected, f"{spoken!r} -> {_itn(spoken)!r}"

def test_bare_number_never_gains_a_currency():
    """No marker, no scale word, or a noun right after: not a price."""
    for text in ("hai mươi tuổi", "hai nghìn người", "hai trăm cái"):
        assert _itn(text) == text, f"{text!r} must not become money"
    # A clause-final compound is a plain number, never a price.
    assert _itn("hai mươi") == "20"
    assert "đ" not in _itn("có hai mươi")

# --------------------------------------------------------------- cardinals

def test_cardinal_parser_values():
    cases = {
        "không": 0,
        "một": 1,
        "mười": 10,
        "mười một": 11,
        "mười lăm": 15,
        "hai mươi": 20,
        "hai mươi mốt": 21,
        "hai mươi lăm": 25,
        "một trăm": 100,
        "một trăm linh năm": 105,
        "hai trăm": 200,
        "năm trăm nghìn": 500_000,
        "một triệu": 1_000_000,
        "một tỷ": 1_000_000_000,
        "một tỷ hai trăm ba mươi bốn triệu năm trăm sáu mươi bảy nghìn tám trăm chín mươi": 1_234_567_890,
        "bảy rưỡi": 7.5,
    }
    for words, expected in cases.items():
        assert parse_vi_cardinal(words) == expected, f"{words!r} -> {parse_vi_cardinal(words)!r}"
    assert parse_vi_cardinal("mươi") is None, "a bare 'mươi' is not a number"
    assert parse_vi_cardinal("bảy bảy") is None, "two digits in a row is not a number"
    assert parse_vi_cardinal("rưỡi") is None, "'rưỡi' alone is not a number"

def test_cardinal_round_trip_against_the_inverse_helper():
    """``int_to_vietnamese_words`` and the ITN parser must agree both ways."""
    values = [
        0, 1, 5, 10, 11, 15, 20, 21, 24, 25, 100, 101, 105, 110, 115, 121,
        500, 999, 1_000, 1_001, 2_010, 500_000, 123_456, 1_000_000,
        1_000_001, 123_456_789, 1_000_000_000,
    ]
    mismatched = [
        (value, int_to_vietnamese_words(value), parse_vi_cardinal(int_to_vietnamese_words(value)))
        for value in values
        if parse_vi_cardinal(int_to_vietnamese_words(value)) != value
    ]
    assert mismatched == [], mismatched

def test_cardinal_standalone_conversion_needs_a_clause_boundary():
    assert _itn("tổng cộng hai mươi") == "tổng cộng 20"
    assert _itn("có một trăm hai mươi ba cái") == "có một trăm hai mươi ba cái"
    assert _itn("có hai mươi") == "có 20", "a clause-final compound is a plain number"
    assert _itn("hai mươi tuổi") == "hai mươi tuổi"

# ----------------------------------------------------- dates and percents

def test_dates_need_the_ngay_thang_frame():
    assert _itn("ngày hai mươi tháng mười một") == "20/11"
    assert _itn("ngày mười tháng một") == "10/1"
    assert _itn("hẹn hai mươi tháng mười một") == "hẹn hai mươi tháng mười một"
    assert _itn("ngày hai mươi tháng mười một 2026") == "20/11 2026"
    assert _itn("ngày ba mươi hai tháng mười ba") == "ngày ba mươi hai tháng mười ba"

def test_percent():
    assert _itn("hai mươi phần trăm") == "20%"
    assert _itn("ba mươi hai phần trăm") == "32%"
    assert _itn("phần trăm") == "phần trăm"

# ----------------------------------------------------------- hard guards

def test_phone_and_id_clauses_are_untouched():
    for text in (
        "gọi +84900000000",
        "số điện thoại 0900000000 nhé",
        "mã 123456 bảy giờ ba mươi tối",
        "gọi 090 000 0000 hai mươi phần trăm",
    ):
        assert _itn(text) == to_llm_text(text), f"{text!r} must not be normalized"

def test_address_email_and_handle_clauses_are_untouched():
    for text in (
        "địa chỉ hai mươi đường Lê Lợi",
        "mail toi@example.com hai mươi phần trăm",
        "@handle hai mươi phần trăm",
    ):
        assert _itn(text) == to_llm_text(text), f"{text!r} must not be normalized"

def test_name_following_a_candidate_blocks_the_rewrite():
    assert _itn("bảy giờ ba mươi Nam") == "bảy giờ ba mươi Nam"
    assert _itn("ngày hai mươi tháng mười một Hà Nội") == "ngày hai mươi tháng mười một Hà Nội"

def test_wrong_rule_negative_controls():
    """Expressions no rule may touch, each for a different reason."""
    untouched = (
        "không có gì",              # 'không' is a digit word but never a price
        "một hai ba",               # digit words without a scale: not a compound
        "giờ ba mươi",              # no hour at all
        "tháng mười một",           # no 'ngày' frame
        "phần trăm",                # no number
        "mươi",                     # not a number
        "hai mươi phần",            # not 'phần trăm'
    )
    for text in untouched:
        assert _itn(text) == text, f"{text!r} must stay unchanged"

# ------------------------------------------------ baseline and invariants

def test_disabled_path_is_the_baseline():
    for text in (
        "  đặt   báo thức bảy rưỡi tối ",
        "chuyển năm trăm nghìn đồng",
        "ngày hai mươi tháng mười một",
        "gọi +84900000000",
    ):
        result = normalize_itn(text)
        assert result.enabled is False
        assert result.text == to_llm_text(text)
        assert result.changes == ()
        assert result.changed is False
        assert apply_itn_pipeline(text) == to_llm_text(text)

def test_enabled_pipeline_starts_from_to_llm_text():
    messy = "  đặt   báo thức bảy rưỡi tối "
    assert apply_itn_pipeline(messy, enabled=True) == "đặt báo thức 19:30"
    assert "  " not in apply_itn_pipeline(messy, enabled=True)

def test_idempotence():
    for text in (
        "đặt báo thức bảy rưỡi tối",
        "chuyển năm trăm nghìn đồng",
        "ngày hai mươi tháng mười một",
        "hai mươi phần trăm",
        "tổng cộng hai mươi",
        "bảy giờ ba mươi",
        "mười chín giờ ba mươi",
        "500k",
        "19h30",
    ):
        once = normalize_itn(text, enabled=True).text
        twice = normalize_itn(once, enabled=True).text
        assert once == twice, f"{text!r}: {once!r} is not idempotent ({twice!r})"

def test_no_empty_replacement_for_a_non_empty_span():
    for text in ("bảy rưỡi tối", "hai mươi phần trăm", "ngày mười tháng một", "500k"):
        for change in normalize_itn(text, enabled=True).changes:
            assert change.replacement != "" or change.original == "", change
            assert change.end > change.start

def test_change_spans_point_at_the_original_text():
    result = normalize_itn("đặt báo thức bảy rưỡi tối", enabled=True)
    assert len(result.changes) == 1, result.changes
    change = result.changes[0]
    assert change.rule == "time_words"
    assert change.original == "bảy rưỡi tối"
    assert change.replacement == "19:30"
    base = to_llm_text("đặt báo thức bảy rưỡi tối")
    assert base[change.start : change.end] == "bảy rưỡi tối"

def test_rule_table_is_introspectable_and_named():
    names = [rule.name for rule in itn_rules()]
    assert names == [
        "date",
        "time_words",
        "time_digits",
        "percent",
        "money_words",
        "money_digits",
        "cardinal",
    ], names
    for rule in itn_rules():
        assert hasattr(rule.ipattern, "finditer"), rule.name
        assert rule.ipattern.search("không có gì") is None or rule.name == "money_words"
