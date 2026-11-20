"""Автотести методів і пайплайнів на малих зображеннях (очікування пораховано вручну)."""
import json
import math
import os
import tempfile

import numpy as np
from PIL import Image

import stem

with open("config.json", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)


def rgb(values):
    """Робить з 2D масиву RGB-зображення з трьома однаковими каналами."""
    gray = np.float64(values)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_bilateral():
    """3x3, центр 10, решта 0, radius 1, sigma_s 1, sigma_r = 10 (для такого зображення).

    Оцінка шуму: відгук маски в єдиній точці 4 * 10 = 40,
    sigma = sqrt(pi / 2) * 40 / 6, sigma_r = factor * sigma = 10 -> factor = 1.5 / sqrt(pi / 2).
    Квадрат відстані між кольорами центру і сусіда 3 * 10^2 = 300: sqrt(3) * 10, тому ваги:
    прямі сусіди exp(-1/2) * exp(-300/200) = e^-2, діагональні exp(-1) * e^-1.5 = e^-2.5.
    Новий центр = 10 / (1 + 4e^-2 + 4e^-2.5).
    """
    image = rgb([[0, 0, 0], [0, 10, 0], [0, 0, 0]])
    factor = 1.5 / math.sqrt(math.pi / 2)
    result = stem.bilateral(image, 1, 1.0, factor)
    expected = 10 / (1 + 4 * math.exp(-2) + 4 * math.exp(-2.5))
    assert np.allclose(result[1, 1], expected)


def test_clahe():
    """2x2 [[10, 10], [10, 20]], 1 блок, обрізання не спрацьовує (clip 1000):
    накопичена гістограма 3 для рівня 10 і 4 для рівня 20,
    тому 10 стає 3/4 * 255 = 191.25, а 20 стає 255."""
    gray = np.float64([[10, 10], [10, 20]])
    expected = [[191.25, 191.25], [191.25, 255]]
    assert np.allclose(stem.clahe_gray(gray, 1, 1000), expected)


def test_clahe_clip():
    """3x3 зі значенням 128, clip 1.5: ліміт 1, надлишок 8 ділиться на 256 рівнів (+1/32).
    Нове значення = (128 / 32 + 1) / (9 - 1/32) * 255."""
    gray = np.full((3, 3), 128.0)
    expected = (128 / 32 + 1) / (9 - 1 / 32) * 255
    assert np.allclose(stem.clahe_gray(gray, 1, 1.5), expected)


def test_unsharp():
    """Рядок [0, 100, 0], sigma 0.3: ядро [e, 1, e] / (1 + 2e), e = exp(-1/0.18).
    Розмитий центр 100 / (1 + 2e), результат 100 + (100 - 100 / (1 + 2e))."""
    edge = math.exp(-1 / 0.18)
    image = rgb([[0, 100, 0]])
    result = stem.unsharp(image, 0.3, 1.0, 0)
    assert np.allclose(result[0, 1], 100 + (100 - 100 / (1 + 2 * edge)))


def test_unsharp_threshold():
    """Деталь висотою 1 менша за поріг 5, тому зображення не змінюється."""
    image = rgb([[0, 1, 0]])
    assert np.allclose(stem.unsharp(image, 0.3, 1.0, 5), image)


def test_gray_world():
    """Пікселі (100, 50, 50), norm 1, strength 1: середні 100, 50, 50, загальне 66.67 ->
    усі канали стають 66.67."""
    image = np.float64([[[100, 50, 50], [100, 50, 50]]])
    assert np.allclose(stem.gray_world(image, 1.0, 1.0), 200 / 3)


def test_pyramid():
    """Нульові пороги: піраміда відновлює зображення точно; м'яке обрізання стале зображення
    не змінює (усі рівні деталей нульові)."""
    image = np.random.default_rng(1).uniform(0, 255, (16, 12, 3))
    assert np.allclose(stem.pyramid(image, [0, 0, 0]), image)
    assert np.allclose(stem.pyramid(np.full((8, 8, 3), 70.0), [5, 5]), 70)


def test_butterworth():
    """H(0, 0) = 1, тому стале зображення (лише постійна складова) не змінюється."""
    image = np.full((5, 6, 3), 90.0)
    assert np.allclose(stem.butterworth(image, 0.3, 2), 90)


def test_ycbcr():
    """Білий колір: Y = 255, Cb = Cr = 128; перетворення туди й назад повертає те саме."""
    white = np.full((1, 1, 3), 255.0)
    luma, cb, cr = stem.rgb_to_ycbcr(white)
    assert np.allclose([luma[0, 0], cb[0, 0], cr[0, 0]], [255, 128, 128])
    image = np.random.default_rng(0).uniform(0, 255, (4, 4, 3))
    assert np.allclose(stem.ycbcr_to_rgb(*stem.rgb_to_ycbcr(image)), image)


def test_pipeline_1():
    """3x3 сірий (128, 128, 128), параметри p1 з config.json.

    Білатеральний фільтр і нерізке маскування стале зображення не змінюють.
    CLAHE: tiles 4 обмежується до 3, у кожному блоці 1 піксель, ліміт 1, надлишку немає ->
    рівень 128 стає 255. Змішування 0.2 * 128 + 0.8 * 255 = 229.6, колір сірий, тому R = G = B.
    """
    image = np.full((3, 3, 3), 128.0)
    result = stem.run_pipeline(image, stem.PIPELINES["p1"], CONFIG["p1"])
    assert len(result) == 3
    assert np.allclose(result[-1][1], 0.2 * 128 + 0.8 * 255)


def test_pipeline_2():
    """1x2 (200, 100, 100), параметри p2: gray_world strength 0.5 дає множники
    1 + 0.5 * (133.33 / 200 - 1) = 0.8333 і 1 + 0.5 * (133.33 / 100 - 1) = 1.1667
    -> (166.67, 116.67, 116.67). Піраміда і фільтр Баттерворта стале зображення не змінюють."""
    image = np.float64([[[200, 100, 100], [200, 100, 100]]])
    result = stem.run_pipeline(image, stem.PIPELINES["p2"], CONFIG["p2"])
    assert np.allclose(result[-1][1], [[[500 / 3, 350 / 3, 350 / 3]] * 2])


def test_metrics():
    """Різниця 2 у кожному пікселі: MSE = 4, PSNR = 10 * lg(255^2 / 4).
    Сталі зображення 100 і 110: SSIM = (2 * 100 * 110 + C1) / (100^2 + 110^2 + C1)."""
    assert stem.mse(np.zeros((2, 2)), np.full((2, 2), 2.0)) == 4
    assert np.isclose(stem.psnr(np.zeros((2, 2)), np.full((2, 2), 2.0)),
                      10 * math.log10(255 ** 2 / 4))
    c1 = (0.01 * 255) ** 2
    expected = (2 * 100 * 110 + c1) / (100 ** 2 + 110 ** 2 + c1)
    assert np.isclose(stem.ssim(np.full((5, 5), 100.0), np.full((5, 5), 110.0)), expected)


def test_edge_cases():
    """Порожній і пошкоджений файли пропускаються; сіре й прозоре зображення читаються як RGB;
    зображення 1x1, 2x2, 3x7 і повністю чорне проходять обидва пайплайни без помилок."""
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, "empty.png"), "wb"):
            pass
        with open(os.path.join(folder, "broken.jpg"), "wb") as file:
            file.write(b"not an image")
        Image.fromarray(np.full((4, 5), 70, np.uint8)).save(os.path.join(folder, "gray.png"))
        Image.fromarray(np.zeros((2, 2, 4), np.uint8)).save(os.path.join(folder, "alpha.png"))
        images = dict(stem.read_images(folder))
        assert sorted(images) == ["alpha.png", "gray.png"]
        assert images["gray.png"].shape == (4, 5, 3)
        assert np.all(images["alpha.png"] == 255)
    for shape in [(1, 1, 3), (2, 2, 3), (3, 7, 3)]:
        image = np.random.default_rng(2).uniform(0, 255, shape)
        for name, stages in stem.PIPELINES.items():
            result = stem.run_pipeline(image, stages, CONFIG[name])[-1][1]
            assert result.shape == shape and np.all(np.isfinite(result))
    for name, stages in stem.PIPELINES.items():
        black = stem.run_pipeline(np.zeros((8, 8, 3)), stages, CONFIG[name])[-1][1]
        assert np.all(np.isfinite(black))


if __name__ == "__main__":
    test_bilateral()
    test_clahe()
    test_clahe_clip()
    test_unsharp()
    test_unsharp_threshold()
    test_gray_world()
    test_pyramid()
    test_butterworth()
    test_ycbcr()
    test_pipeline_1()
    test_pipeline_2()
    test_metrics()
    test_edge_cases()
    print("Усі тести пройдено")
