from tram_odometry.geo import LocalEnu

# Широта и долгота порядка московских
LAT0, LON0, ALT0 = 55.75, 37.62, 150.0


def test_origin_maps_to_zero():
    enu = LocalEnu(LAT0, LON0, ALT0)
    east, north, up = enu.to_enu(LAT0, LON0, ALT0)
    assert abs(east) < 1e-6 and abs(north) < 1e-6 and abs(up) < 1e-6


def test_latitude_step_goes_north():
    # 0,001° широты на 55,75° ≈ 111,34 м (меридиональный радиус кривизны WGS84)
    east, north, up = LocalEnu(LAT0, LON0, ALT0).to_enu(LAT0 + 0.001, LON0, ALT0)
    assert abs(east) < 0.01
    assert abs(north - 111.34) < 0.1
    assert abs(up) < 0.01


def test_longitude_step_goes_east():
    # 0,001° долготы на 55,75° ≈ 62,79 м (радиус первого вертикала × cos широты)
    east, north, up = LocalEnu(LAT0, LON0, ALT0).to_enu(LAT0, LON0 + 0.001, ALT0)
    assert abs(east - 62.79) < 0.1
    assert abs(north) < 0.01
    assert abs(up) < 0.01


def test_altitude_goes_up():
    east, north, up = LocalEnu(LAT0, LON0, ALT0).to_enu(LAT0, LON0, ALT0 + 10.0)
    assert abs(up - 10.0) < 1e-6
    assert abs(east) < 1e-6 and abs(north) < 1e-6
