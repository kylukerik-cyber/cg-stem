"""Відновлення фотографій облич класичними методами: два пайплайни по 3 стадії.

Пайплайн 1: білатеральний фільтр -> CLAHE -> нерізке маскування.
Пайплайн 2: баланс білого Gray World -> шумозаглушення в піраміді -> фільтр Баттерворта.
Усі методи реалізовані через NumPy. Pillow потрібен лише щоб читати і зберігати файли.
"""
import csv
import itertools
import json
import os
import time
import zlib

import numpy as np
from PIL import Image

# стадії пайплайнів у порядку виконання і категорія кожної стадії
PIPELINES = {
    "p1": ["bilateral", "clahe", "unsharp"],
    "p2": ["gray_world", "pyramid", "butterworth"],
}
CATEGORIES = {
    "bilateral": "фільтрація",
    "clahe": "гістограмна корекція",
    "unsharp": "ретушування",
    "gray_world": "колірна корекція",
    "pyramid": "масштабно-інваріантні методи",
    "butterworth": "перетворення Фур'є",
}


# введення / виведення
def load_image(path):
    """Завантажує зображення як RGB масив float64 (0..255).

    Сірі зображення стають трьома однаковими каналами, прозорі накладаються на білий фон.
    Порожній або пошкоджений файл дає ValueError.
    """
    if os.path.getsize(path) == 0:
        raise ValueError("порожній файл")
    with Image.open(path) as img:
        img.load()
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            data = np.float64(np.array(img))
            gray = data / 65535 * 255 if data.max() > 255 else data
            return np.repeat(gray[:, :, None], 3, axis=2)
        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
            rgba = img.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white, rgba)
        return np.float64(np.array(img.convert("RGB")))


def save_image(arr, path):
    """Зберігає масив як PNG, створюючи папку."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.uint8(np.clip(np.rint(arr), 0, 255))).save(path)


# допоміжні операції (не є окремими стадіями)
def gaussian_kernel(sigma):
    """Одновимірне ядро Гауса довжиною 2 * ceil(3 * sigma) + 1, сума = 1."""
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur(image, sigma):
    """Розмиття Гаусом: згортка по рядках, потім по стовпцях (краї повторюються)."""
    kernel = gaussian_kernel(sigma)
    radius = len(kernel) // 2
    result = np.float64(image)
    for axis in (0, 1):
        pad = [(0, 0)] * result.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(result, pad, mode="edge")
        size = result.shape[axis]
        blurred = np.zeros_like(result)
        for i, weight in enumerate(kernel):
            blurred += weight * np.take(padded, np.arange(i, i + size), axis=axis)
        result = blurred
    return result


def resize(image, height, width):
    """Зміна розміру білінійною інтерполяцією (центри пікселів вирівнюються)."""
    in_h, in_w = image.shape[:2]
    ys = np.clip((np.arange(height) + 0.5) * in_h / height - 0.5, 0, in_h - 1)
    xs = np.clip((np.arange(width) + 0.5) * in_w / width - 0.5, 0, in_w - 1)
    y0 = np.int64(ys)
    x0 = np.int64(xs)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0).reshape(-1, 1, 1)
    wx = (xs - x0).reshape(1, -1, 1)
    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def rgb_to_ycbcr(img):
    """RGB -> YCbCr (BT.601, як у JPEG): Y - яскравість, Cb і Cr - колір."""
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 128 + 0.564 * (blue - luma)
    cr = 128 + 0.713 * (red - luma)
    return luma, cb, cr


def ycbcr_to_rgb(luma, cb, cr):
    """YCbCr -> RGB (обернені формули до rgb_to_ycbcr)."""
    red = luma + (cr - 128) / 0.713
    blue = luma + (cb - 128) / 0.564
    green = (luma - 0.299 * red - 0.114 * blue) / 0.587
    return np.stack([red, green, blue], axis=2)


def estimate_noise(img):
    """Оцінка sigma гаусового шуму (метод Immerkaer, 1996).

    Маска [[1,-2,1],[-2,4,-2],[1,-2,1]] майже гасить плавні ділянки й лишає шум:
    sigma = sqrt(pi / 2) * sum|I * N| / (6 * (W - 2) * (H - 2)).
    """
    gray, _, _ = rgb_to_ycbcr(img)
    height, width = gray.shape
    if height < 3 or width < 3:
        return 0.0
    mask = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    response = np.zeros((height - 2, width - 2))
    for dy in range(3):
        for dx in range(3):
            response += mask[dy][dx] * gray[dy:dy + height - 2, dx:dx + width - 2]
    return np.sqrt(np.pi / 2) * np.abs(response).sum() / (6 * (width - 2) * (height - 2))


# стадія: фільтрація
def bilateral(img, radius, sigma_s, sigma_r_factor):
    """Білатеральний фільтр: згладжує шум, але зберігає межі.

    Вага сусіда q для пікселя p: exp(-|p-q|^2 / 2 sigma_s^2) * exp(-||I(p)-I(q)||^2 / 2 sigma_r^2),
    ||I(p)-I(q)|| - відстань між кольорами в RGB. sigma_r = sigma_r_factor * оцінка шуму,
    тому на чистому зображенні фільтр майже нічого не змінює, а на шумному згладжує сильніше.
    """
    sigma_r = max(1.0, sigma_r_factor * estimate_noise(img))
    height, width = img.shape[:2]
    padded = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    total = np.zeros_like(img)
    weights = np.zeros((height, width, 1))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # зсунута копія: у кожному пікселі лежить значення сусіда (dy, dx)
            neighbour = padded[radius + dy:radius + dy + height, radius + dx:radius + dx + width]
            distance = np.sum((neighbour - img) ** 2, axis=2, keepdims=True)
            weight = np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma_s ** 2)) \
                * np.exp(-distance / (2 * sigma_r ** 2))
            total += weight * neighbour
            weights += weight
    return total / weights


# стадія: гістограмна корекція
def tile_table(tile, clip_limit):
    """Таблиця нових яскравостей (256 значень) для одного блоку CLAHE.

    Стовпці гістограми обрізаються на рівні clip_limit * (пікселів / 256), надлишок
    порівну ділиться між усіма рівнями, далі - як звичайне вирівнювання гістограми.
    """
    hist = np.float64(np.bincount(np.int64(tile.ravel()), minlength=256))
    limit = max(1.0, clip_limit * tile.size / 256)
    excess = np.sum(np.maximum(hist - limit, 0))
    hist = np.minimum(hist, limit) + excess / 256
    cdf = np.cumsum(hist)
    return (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-12) * 255


def clahe_gray(gray, tiles, clip_limit):
    """CLAHE для одного каналу (адаптивне вирівнювання гістограми з обмеженням контрасту).

    Зображення ділиться на tiles x tiles блоків, кожен має свою таблицю. Щоб на межах
    блоків не було сходинок, значення пікселя - білінійна інтерполяція таблиць
    чотирьох найближчих центрів блоків.
    """
    height, width = gray.shape
    rows, cols = min(tiles, height), min(tiles, width)
    y_edges = np.linspace(0, height, rows + 1).astype(int)
    x_edges = np.linspace(0, width, cols + 1).astype(int)
    levels = np.int64(np.clip(np.rint(gray), 0, 255))
    tables = np.zeros((rows, cols, 256))
    for i in range(rows):
        for j in range(cols):
            tile = levels[y_edges[i]:y_edges[i + 1], x_edges[j]:x_edges[j + 1]]
            tables[i, j] = tile_table(tile, clip_limit)
    # положення пікселя між центрами блоків
    y_pos = np.interp(np.arange(height), (y_edges[:-1] + y_edges[1:] - 1) / 2, np.arange(rows))
    x_pos = np.interp(np.arange(width), (x_edges[:-1] + x_edges[1:] - 1) / 2, np.arange(cols))
    y0, x0 = np.int64(y_pos), np.int64(x_pos)
    y1, x1 = np.minimum(y0 + 1, rows - 1), np.minimum(x0 + 1, cols - 1)
    wy = (y_pos - y0)[:, None]
    wx = (x_pos - x0)[None, :]
    y0, y1, x0, x1 = y0[:, None], y1[:, None], x0[None, :], x1[None, :]
    top = tables[y0, x0, levels] * (1 - wx) + tables[y0, x1, levels] * wx
    bottom = tables[y1, x0, levels] * (1 - wx) + tables[y1, x1, levels] * wx
    return top * (1 - wy) + bottom * wy


def clahe(img, tiles, clip_limit, blend):
    """CLAHE лише для яскравості Y, колір (Cb, Cr) не змінюється.

    blend змішує результат з початковою яскравістю: Y' = (1 - blend) * Y + blend * CLAHE(Y).
    """
    luma, cb, cr = rgb_to_ycbcr(img)
    luma = (1 - blend) * luma + blend * clahe_gray(luma, tiles, clip_limit)
    return ycbcr_to_rgb(luma, cb, cr)


# стадія: ретушування
def unsharp(img, sigma, amount, threshold):
    """Нерізке маскування: I' = I + amount * M, де M = I - розмите(I).

    Якщо |M| < threshold, це вважається залишком шуму і не підсилюється.
    """
    mask = img - gaussian_blur(img, sigma)
    mask[np.abs(mask) < threshold] = 0
    return img + amount * mask


# стадія: колірна корекція
def gray_world(img, strength, norm):
    """Баланс білого Gray World (узагальнення Shades of Gray).

    Оцінка освітлення в каналі c: e_c = (mean((I_c / 255)^p))^(1/p), p = norm
    (p = 1 - звичайне середнє). Канал множиться на 1 + strength * (mean(e) / e_c - 1):
    strength = 0 нічого не змінює, strength = 1 - повна корекція.
    """
    estimate = np.mean((img / 255) ** norm, axis=(0, 1)) ** (1 / norm)
    estimate = np.maximum(estimate, 1e-6)  # чорне зображення: ділення на 0
    gains = 1 + strength * (estimate.mean() / estimate - 1)
    return img * gains


# стадія: масштабно-інваріантні методи
def build_pyramid(img, levels):
    """Лапласіанна піраміда [L0, L1, ..., G_n]: L_i = G_i - збільшене(G_(i+1)).

    G_(i+1) - зменшена вдвічі розмита копія G_i. L_i містить деталі масштабу i.
    """
    layers = []
    current = img
    for _ in range(levels):
        height, width = current.shape[:2]
        if min(height, width) < 4:
            break
        smaller = gaussian_blur(current, 1.0)[::2, ::2]
        layers.append(current - resize(smaller, height, width))
        current = smaller
    layers.append(current)
    return layers


def collapse_pyramid(layers):
    """Відновлення з піраміди: G_i = L_i + збільшене(G_(i+1))."""
    current = layers[-1]
    for detail in reversed(layers[:-1]):
        current = detail + resize(current, detail.shape[0], detail.shape[1])
    return current


def pyramid(img, thresholds):
    """Шумозаглушення в піраміді: коефіцієнти L_i обрізаються з порогом thresholds[i].

    М'яке обрізання sign(x) * max(|x| - t, 0) прибирає малі коливання (шум)
    і лише трохи зменшує великі (контури).
    """
    levels = build_pyramid(img, len(thresholds))
    for i in range(len(levels) - 1):
        levels[i] = np.sign(levels[i]) * np.maximum(np.abs(levels[i]) - thresholds[i], 0)
    return collapse_pyramid(levels)


# стадія: перетворення Фур'є
def butterworth(img, cutoff, order):
    """Низькочастотний фільтр Баттерворта в частотній області.

    H(u, v) = 1 / (1 + (D / D0)^(2n)), D - відстань до центру спектра, D0 = cutoff * min(H, W) / 2.
    Краї дзеркально доповнюються на чверть розміру, щоб зменшити «дзвін».
    numpy.fft - загальна математична функція ДПФ, сам фільтр зроблено вручну.
    """
    height, width = img.shape[:2]
    pad_y, pad_x = height // 4, width // 4
    mode = "reflect" if min(height, width) > 1 else "edge"
    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode=mode)
    full_h, full_w = padded.shape[:2]
    v = np.arange(full_h) - full_h // 2
    u = np.arange(full_w) - full_w // 2
    distance = np.sqrt(v[:, None] ** 2 + u[None, :] ** 2)
    d0 = cutoff * min(full_h, full_w) / 2
    transfer = 1 / (1 + (distance / d0) ** (2 * order))
    result = np.zeros_like(padded)
    for channel in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(padded[:, :, channel]))
        result[:, :, channel] = np.fft.ifft2(np.fft.ifftshift(spectrum * transfer)).real
    return result[pad_y:pad_y + height, pad_x:pad_x + width]


METHODS = {"bilateral": bilateral, "clahe": clahe, "unsharp": unsharp,
           "gray_world": gray_world, "pyramid": pyramid, "butterworth": butterworth}


def run_pipeline(img, stages, params):
    """Виконує стадії по черзі; повертає список (назва, результат, час у с).

    Після кожної стадії значення обмежуються 0..255.
    """
    results = []
    current = img
    for stage in stages:
        start = time.perf_counter()
        current = np.clip(METHODS[stage](current, **params[stage]), 0, 255)
        results.append((stage, current, time.perf_counter() - start))
    return results


# шум, метрики
def scenario_seed(seed, filename, scenario):
    """Seed шуму залежить від назви файлу і сценарію, тому шум завжди однаковий."""
    return (seed + zlib.crc32((filename + ":" + scenario).encode("utf-8"))) % 2 ** 32


def add_noise(img, scenario, seed):
    """Спотворює зображення за сценарієм з config.json."""
    rng = np.random.default_rng(seed)
    kind = scenario["kind"]
    result = img.copy()
    if kind == "salt_pepper":
        chance = rng.random(img.shape[:2])
        result[chance < scenario["amount"] / 2] = 0
        result[chance > 1 - scenario["amount"] / 2] = 255
        return result
    if kind == "low_light":
        # темніше освітлення: I' = gain * 255 * (I / 255)^gamma
        result = scenario["gain"] * 255 * (img / 255) ** scenario["gamma"]
    elif kind == "color_cast":
        # неправильний баланс білого: канали R, G, B множаться на різні коефіцієнти
        result = img * np.array(scenario["gains"])
    return np.clip(result + rng.normal(0, scenario["sigma"], img.shape), 0, 255)


def mse(image1, image2):
    """Середньоквадратична помилка."""
    diff = np.float64(image1) - np.float64(image2)
    return np.sum(diff ** 2) / diff.size


def psnr(image1, image2):
    """Пікове відношення сигнал/шум у dB: 10 * log10(255^2 / MSE)."""
    error = mse(image1, image2)
    if error == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / error)


def ssim(image1, image2):
    """SSIM з гаусовим вікном (sigma = 1.5, 11x11), середнє по всіх пікселях і каналах.

    SSIM = ((2 mu_x mu_y + C1)(2 sigma_xy + C2))
           / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)),
    C1 = (0.01 * 255)^2, C2 = (0.03 * 255)^2; середні, дисперсії й коваріація - у вікні.
    """
    x, y = np.float64(image1), np.float64(image2)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mean_x, mean_y = gaussian_blur(x, 1.5), gaussian_blur(y, 1.5)
    var_x = gaussian_blur(x * x, 1.5) - mean_x ** 2
    var_y = gaussian_blur(y * y, 1.5) - mean_y ** 2
    cov = gaussian_blur(x * y, 1.5) - mean_x * mean_y
    top = (2 * mean_x * mean_y + c1) * (2 * cov + c2)
    bottom = (mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2)
    return np.mean(top / bottom)


# запуск
def read_images(folder):
    """Читає всі зображення з папки; пошкоджені пропускаються з повідомленням."""
    images = []
    for filename in sorted(os.listdir(folder)):
        try:
            images.append((filename, load_image(os.path.join(folder, filename))))
        except (OSError, ValueError) as error:
            print("Пропущено:", filename, "-", error)
    return images


def process_images(runs, scenarios, config, folder="images"):
    """Обробляє всі зображення: зберігає результати кожної стадії і metrics.csv.

    runs - словник {назва запуску: список стадій}, наприклад {"p1": PIPELINES["p1"]}.
    Повертає рядки метрик (для зведеної таблиці).
    """
    all_rows = []
    images = read_images(folder)
    for run_name, stages in runs.items():
        params = config[run_name] if run_name in PIPELINES else \
            {stages[0]: config[pipeline_of(stages[0])][stages[0]]}
        for scenario in scenarios:
            out_folder = "results/" + run_name + "_" + scenario
            rows = []
            for filename, original in images:
                noisy = add_noise(original, config["scenarios"][scenario],
                                  scenario_seed(config["seed"], filename, scenario))
                name = os.path.splitext(filename)[0]
                save_image(noisy, out_folder + "/" + name + "_0_input.png")
                for number, (stage, result, seconds) in \
                        enumerate(run_pipeline(noisy, stages, params), start=1):
                    save_image(result, out_folder + "/" + name + "_" + str(number) + "_"
                               + stage + ".png")
                    rows.append({"image": filename, "stage": number, "method": stage,
                                 "mse_in": mse(original, noisy), "psnr_in": psnr(original, noisy),
                                 "ssim_in": ssim(original, noisy), "mse": mse(original, result),
                                 "psnr": psnr(original, result), "ssim": ssim(original, result),
                                 "seconds": seconds})
            with open(out_folder + "/metrics.csv", "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for row in rows:
                all_rows.append({"run": run_name, "scenario": scenario, **row})
            print("Готово:", out_folder, "-", len(images), "зображень")
    return all_rows


def pipeline_of(method):
    """Назва пайплайна, у якому використовується метод."""
    for name, stages in PIPELINES.items():
        if method in stages:
            return name
    raise KeyError(method)


def summary(all_rows, config):
    """Зведена таблиця по останній стадії, тест-кейси і порівняння p1 з p2 (results/summary.csv)."""
    last = {}
    for row in all_rows:
        if row["stage"] == len(PIPELINES.get(row["run"], [row["method"]])):
            last[(row["run"], row["scenario"], row["image"])] = row
    table = []
    print(f"{'запуск':8}{'сценарій':14}{'PSNR вх':>9}{'PSNR':>8}{'SSIM вх':>9}{'SSIM':>8}"
          f"{'тест-кейс':>11}")
    for run_name, scenario in sorted({(k[0], k[1]) for k in last}):
        rows = [r for k, r in last.items() if k[0] == run_name and k[1] == scenario]
        line = {"run": run_name, "scenario": scenario, "n": len(rows)}
        for key in ("mse_in", "mse", "psnr_in", "psnr", "ssim_in", "ssim", "seconds"):
            line[key] = np.mean([r[key] for r in rows])
        line["psnr_std"] = np.std([r["psnr"] for r in rows])
        line["ssim_std"] = np.std([r["ssim"] for r in rows])
        line["test_passed"] = ""
        if scenario in config["test_cases"].get(run_name, []):
            # тест-кейс пройдено, якщо PSNR зріс на min_psnr_gain дБ і SSIM не впав
            good = [r for r in rows if r["psnr"] - r["psnr_in"] >= config["min_psnr_gain"]
                    and r["ssim"] >= r["ssim_in"]]
            line["test_passed"] = f"{len(good)}/{len(rows)}"
        table.append(line)
        print(f"{run_name:8}{scenario:14}{line['psnr_in']:9.2f}{line['psnr']:8.2f}"
              f"{line['ssim_in']:9.3f}{line['ssim']:8.3f}{line['test_passed']:>11}")
    comparison = compare(last)
    with open("results/summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    if comparison:
        with open("results/comparison.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)


def compare(last):
    """Попарне порівняння p1 і p2 на тих самих зображеннях і сценаріях."""
    comparison = []
    for scenario in sorted({k[1] for k in last}):
        pairs = [(r, last[("p2", scenario, k[2])]) for k, r in last.items()
                 if k[0] == "p1" and k[1] == scenario and ("p2", scenario, k[2]) in last]
        for metric in ("psnr", "ssim"):
            diffs = [p1[metric] - p2[metric] for p1, p2 in pairs]
            if diffs:
                line = {"scenario": scenario, "metric": metric,
                        "p1_better": sum(d > 0 for d in diffs),
                        "p2_better": sum(d < 0 for d in diffs),
                        "mean_p1_minus_p2": np.mean(diffs)}
                comparison.append(line)
                print(f"{scenario:14}{metric}: p1 кращий на {line['p1_better']}, p2 кращий на "
                      f"{line['p2_better']} зображеннях, середня різниця {np.mean(diffs):+.3f}")
    return comparison


def tune(pipeline_name, config):
    """Підбір параметрів на tune_images: для кожної стадії по черзі перебирається сітка.

    Критерій - середній SSIM на сценаріях тест-кейсів пайплайна. Два проходи, бо
    найкращий параметр однієї стадії залежить від інших. Журнал: results/tuning_<p>.csv.
    """
    params = json.loads(json.dumps(config[pipeline_name]))  # копія
    samples = []
    for filename, original in read_images("tune_images"):
        for scenario in config["test_cases"][pipeline_name]:
            seed = scenario_seed(config["seed"] + 500000, filename, scenario)
            samples.append((original, add_noise(original, config["scenarios"][scenario], seed)))

    def score(trial):
        outputs = [run_pipeline(noisy, PIPELINES[pipeline_name], trial)[-1][1]
                   for _, noisy in samples]
        return (np.mean([ssim(o, r) for (o, _), r in zip(samples, outputs)]),
                np.mean([psnr(o, r) for (o, _), r in zip(samples, outputs)]))

    best = score(params)
    log = []
    for pass_number in (1, 2):
        for stage in PIPELINES[pipeline_name]:
            grid = config["tuning_grid"][stage]
            for values in itertools.product(*grid.values()):
                trial = json.loads(json.dumps(params))
                trial[stage].update(dict(zip(grid, values)))
                result = score(trial)
                log.append([pass_number, stage, json.dumps(dict(zip(grid, values))),
                            result[0], result[1]])
                if result[0] > best[0]:
                    best = result
                    params = trial
            print(f"Прохід {pass_number}, {stage}: {params[stage]}, SSIM {best[0]:.4f}, "
                  f"PSNR {best[1]:.2f}")
    os.makedirs("results", exist_ok=True)
    with open("results/tuning_" + pipeline_name + ".csv", "w", newline="",
              encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["pass", "stage", "params", "ssim", "psnr"])
        writer.writerows(log)
    print("Найкращі параметри (можна перенести в config.json):")
    print(json.dumps(params, ensure_ascii=False))
    return params


def choose_runs():
    """Меню: що запускати. None - вихід, порожній словник - невірний номер."""
    print("Вибір запуску:")
    print("1 - Пайплайн 1 (bilateral -> clahe -> unsharp)")
    print("2 - Пайплайн 2 (gray_world -> pyramid -> butterworth)")
    print("3 - Обидва пайплайни + зведена таблиця")
    print("4 - Один метод")
    print("5 - Підбір параметрів")
    print("0 - Вихід")
    choice = input("Введіть номер: ")
    if choice == "0":
        return None
    if choice in ("1", "2"):
        return {"p" + choice: PIPELINES["p" + choice]}
    if choice == "3":
        return dict(PIPELINES)
    if choice == "4":
        names = list(METHODS)
        for number, name in enumerate(names, start=1):
            print(number, "-", name, "(" + CATEGORIES[name] + ")")
        number = input("Номер методу: ")
        if number.isdigit() and 1 <= int(number) <= len(names):
            return {names[int(number) - 1]: [names[int(number) - 1]]}
    if choice == "5":
        return {"tune": []}
    print("Невірний номер")
    return {}


def choose_scenarios(config):
    """Меню сценаріїв спотворення; None при невірному номері."""
    names = list(config["scenarios"])
    print("Вибір сценарію:")
    for number, name in enumerate(names, start=1):
        print(number, "-", name)
    print(len(names) + 1, "- Усі сценарії")
    choice = input("Введіть номер: ")
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        return [names[int(choice) - 1]]
    if choice == str(len(names) + 1):
        return names
    print("Невірний номер")
    return None


def main():
    """Меню запуску пайплайнів і методів на зображеннях з папки images."""
    with open("config.json", encoding="utf-8") as file:
        config = json.load(file)
    if not os.path.isdir("images"):
        print("Папку images не знайдено")
        return
    while True:
        runs = choose_runs()
        if runs is None:
            break
        if not runs:
            continue
        if "tune" in runs:
            name = input("Пайплайн (p1 або p2): ").strip()
            if name in PIPELINES:
                tune(name, config)
            else:
                print("Невірна назва пайплайна")
            continue
        scenarios = choose_scenarios(config)
        if scenarios is None:
            continue
        rows = process_images(runs, scenarios, config)
        if len(runs) > 1:
            summary(rows, config)
        print("Обробку завершено")


if __name__ == "__main__":
    main()
