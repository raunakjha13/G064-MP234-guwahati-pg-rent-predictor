from rentpred.clean import clean_data


def test_leaking_column_is_dropped(small_df):
    out = clean_data(small_df)
    assert "rent_in_signed_agreement" not in out.columns


def test_sector_east_area_converted_to_sqft(small_df):
    out = clean_data(small_df)
    east_area = out.loc[out["locality"] == "Sector East", "room_area_sqft"].iloc[0]
    assert abs(east_area - 20.0 * 10.7639) < 0.01


def test_other_localities_area_unchanged(small_df):
    out = clean_data(small_df)
    assert out.loc[0, "room_area_sqft"] == 200.0
    assert out.loc[2, "room_area_sqft"] == 220.0


def test_sentinel_wifi_replaced_by_median(small_df):
    out = clean_data(small_df)
    assert (out["wifi_mbps"] >= 0).all()
    assert out.loc[1, "wifi_mbps"] == 60.0   # median of 50 and 70


def test_furnishing_has_three_clean_levels(small_df):
    out = clean_data(small_df)
    assert list(out["furnishing"]) == ["full", "unfurnished", "semi"]


def test_clean_data_does_not_change_the_input(small_df):
    before = small_df.copy()
    clean_data(small_df)
    assert small_df.equals(before)