"""Generates a synthetic product catalog (data/csv/products.csv) for NovaTech Electronics."""
import csv
import random
from pathlib import Path

random.seed(42)

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "csv" / "products.csv"

CATEGORIES = {
    "Laptops": {
        "brands": ["NovaTech", "Zenbyte", "Corevia"],
        "models": ["NovaBook Air 13", "NovaBook Pro 15", "NovaBook Pro 17", "Zenbyte Slim 14",
                   "Zenbyte Ultra 16", "Corevia Work 15", "Corevia Studio 16"],
        "price": (699, 2499),
    },
    "Smartphones": {
        "brands": ["NovaTech", "Pulsar", "Corevia"],
        "models": ["Pulsar X2", "Pulsar X2 Pro", "Pulsar Lite", "NovaPhone 12", "NovaPhone 12 Mini",
                   "Corevia Edge"],
        "price": (299, 1299),
    },
    "Headphones": {
        "brands": ["Pulse", "NovaTech", "Aurora"],
        "models": ["Pulse Earbuds X2", "Pulse Earbuds X2 Pro", "Pulse Over-Ear ANC", "NovaSound Buds",
                   "Aurora Studio Cans"],
        "price": (39, 349),
    },
    "Monitors": {
        "brands": ["NovaTech", "Corevia", "Zenbyte"],
        "models": ["NovaView 24 FHD", "NovaView 27 QHD", "NovaView 32 4K", "Corevia UltraWide 34",
                   "Zenbyte Studio 27"],
        "price": (149, 899),
    },
    "Smartwatches": {
        "brands": ["Aurora", "NovaTech", "Pulsar"],
        "models": ["Aurora Smartwatch", "Aurora Smartwatch Pro", "NovaFit Watch", "Pulsar Watch Lite"],
        "price": (79, 449),
    },
    "Tablets": {
        "brands": ["NovaTech", "Corevia"],
        "models": ["NovaTab 10", "NovaTab 10 Pro", "Corevia Slate 11"],
        "price": (199, 899),
    },
    "Cameras": {
        "brands": ["Zenbyte", "NovaTech"],
        "models": ["Zenbyte ClickPro X", "Zenbyte ClickPro X2", "NovaCam Mirrorless"],
        "price": (349, 1899),
    },
    "Speakers": {
        "brands": ["Pulse", "Aurora"],
        "models": ["Pulse Boom Mini", "Pulse Boom XL", "Aurora Home Speaker"],
        "price": (29, 299),
    },
    "Accessories": {
        "brands": ["NovaTech", "Corevia", "Pulse"],
        "models": ["USB-C 100W Charger", "Wireless Mouse M2", "Mechanical Keyboard K5",
                   "Laptop Sleeve 15in", "Fast Wireless Charger Pad"],
        "price": (9, 89),
    },
    "Gaming Consoles": {
        "brands": ["Corevia", "NovaTech"],
        "models": ["Corevia PlayDeck", "Corevia PlayDeck Pro", "NovaPlay Handheld"],
        "price": (199, 599),
    },
}

SPEC_TEMPLATES = {
    "Laptops": "{ram}GB RAM, {storage}GB SSD, {cpu} CPU, {screen}in display, {battery}h battery",
    "Smartphones": "{storage}GB storage, {screen}in display, {battery}mAh battery, {camera}MP camera",
    "Headphones": "{battery}h battery life, {anc}, Bluetooth {bt}",
    "Monitors": "{screen}in, {res}, {refresh}Hz refresh rate",
    "Smartwatches": "{battery}h battery, {waterproof}, heart-rate + SpO2 sensors",
    "Tablets": "{storage}GB storage, {screen}in display, {battery}h battery",
    "Cameras": "{mp}MP sensor, 4K video, {zoom}x optical zoom",
    "Speakers": "{battery}h battery, {waterproof}, {watt}W output",
    "Accessories": "Universal compatibility, 1-year limited warranty",
    "Gaming Consoles": "{storage}GB storage, {battery}h battery, handheld + docked mode",
}


def rand_specs(category):
    t = SPEC_TEMPLATES[category]
    return t.format(
        ram=random.choice([8, 16, 32]),
        storage=random.choice([128, 256, 512, 1024]),
        cpu=random.choice(["NovaCore i5", "NovaCore i7", "NovaCore i9", "Zen A2"]),
        screen=random.choice([13, 14, 15, 16, 17, 24, 27, 32, 10, 11]),
        battery=random.choice([6, 8, 10, 12, 18, 24, 30, 40]),
        camera=random.choice([12, 48, 50, 108]),
        anc=random.choice(["Active Noise Cancelling", "Passive isolation"]),
        bt=random.choice(["5.2", "5.3"]),
        res=random.choice(["1920x1080", "2560x1440", "3840x2160"]),
        refresh=random.choice([60, 75, 144, 165]),
        waterproof=random.choice(["IP67 water resistant", "IPX4 splash resistant"]),
        mp=random.choice([20, 24, 32]),
        zoom=random.choice([3, 5, 10]),
        watt=random.choice([10, 20, 40, 80]),
    )


ADJECTIVES = ["reliable", "compact", "high-performance", "everyday", "premium", "entry-level", "flagship"]


def build_rows():
    rows = []
    pid = 1001
    for category, cfg in CATEGORIES.items():
        for model in cfg["models"]:
            brand = random.choice(cfg["brands"])
            lo, hi = cfg["price"]
            price = round(random.uniform(lo, hi), 2)
            stock = random.randint(0, 250)
            rating = round(random.uniform(3.2, 4.9), 1)
            year = random.choice([2022, 2023, 2024, 2025])
            adj = random.choice(ADJECTIVES)
            desc = f"{brand} {model} is a {adj} {category.lower()[:-1] if category.endswith('s') else category.lower()} released in {year}."
            rows.append({
                "product_id": f"P{pid}",
                "name": model,
                "category": category,
                "brand": brand,
                "price_usd": price,
                "stock": stock,
                "rating": rating,
                "release_year": year,
                "specs_summary": rand_specs(category),
                "description": desc,
            })
            pid += 1
    return rows


def main():
    rows = build_rows()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} products to {OUT_PATH}")


if __name__ == "__main__":
    main()
