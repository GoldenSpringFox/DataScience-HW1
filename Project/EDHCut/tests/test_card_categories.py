import numpy as np

from edhcut.analysis.card_categories import same_category_mask


def test_same_category_mask_matches_query_land_status() -> None:
    is_land = np.array([True, False, True, False])
    mask = same_category_mask(is_land, True, include_cross_category=False)
    assert list(mask) == [True, False, True, False]


def test_same_category_mask_matches_nonland_query() -> None:
    is_land = np.array([True, False, True, False])
    mask = same_category_mask(is_land, False, include_cross_category=False)
    assert list(mask) == [False, True, False, True]


def test_same_category_mask_include_cross_category_is_a_no_op() -> None:
    is_land = np.array([True, False, True])
    mask = same_category_mask(is_land, True, include_cross_category=True)
    assert list(mask) == [True, True, True]
