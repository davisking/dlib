from dlib import rgb_pixel


def test_rgb_pixel():
    p = rgb_pixel(0, 50, 100)
    assert p.red == 0
    assert p.green == 50
    assert p.blue == 100
    assert str(p) == "red: 0, green: 50, blue: 100"
    assert repr(p) == "rgb_pixel(0,50,100)"

    p = rgb_pixel(blue=0, red=50, green=100)
    assert p.red == 50
    assert p.green == 100
    assert p.blue == 0
    assert str(p) == "red: 50, green: 100, blue: 0"
    assert repr(p) == "rgb_pixel(50,100,0)"

    p.red = 100
    p.green = 0
    p.blue = 50
    assert p.red == 100
    assert p.green == 0
    assert p.blue == 50
    assert str(p) == "red: 100, green: 0, blue: 50"
    assert repr(p) == "rgb_pixel(100,0,50)"
