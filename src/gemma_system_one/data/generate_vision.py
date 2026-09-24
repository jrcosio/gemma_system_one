"""Generador visual ``support-vision-v1`` (fase 4, spec §5.1, §7.C; decisión 0007).

Cada caso (= ``group_id``) es un panel de errores por servicio dibujado con Pillow a partir de
valores conocidos: 3–5 barras con nombre de servicio, eje 0–100 y, a veces, una línea de umbral.
El estado textual sólo dice que se adjunta el panel: **la respuesta está únicamente en la imagen**.
Varias preguntas sobre la misma imagen con respuestas distintas obligan a leer la pregunta.

Etiquetas verificables con márgenes (ninguna barra a menos de 5 unidades de un límite de la
rúbrica, 8 del umbral; máximo/mínimo únicos con 10 de margen; pares comparados con 12). El
título no revela la respuesta y el fichero se nombra por su sha256.

Estilos: en cada lista de variantes (estilos visuales, frases de pregunta y de estado) la última
sólo se usa con ``variant="transfer"``: mide estilos visuales y redacciones no vistos, no tareas
nuevas. ``audit.jsonl`` guarda los hechos de cada caso para verificar etiquetas; el modelo no lo lee.
"""

from __future__ import annotations

import io
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..contracts import Example
from ..images import content_address
from .dataset import EXAMPLES_FILE, MANIFEST_FILE, sha256_bytes

GENERATOR_VERSIONS = {
    "v1": "support-vision-v1",
    # v2 (revisión de fase 4, spec §5.1): el split por grupos se planifica antes de muestrear y cada
    # partición tiene su propio repertorio de estilos (disjuntos) y su propio flujo aleatorio.
    "v2": "support-vision-v2",
}
GENERATOR_VERSION = GENERATOR_VERSIONS["v1"]
VisionVersion = Literal["v1", "v2"]
Variant = Literal["main", "transfer"]
SERVICES = ["API", "Auth", "Billing", "Search", "Email", "Storage", "Sync"]
SIZE = (480, 360)
LEVEL_BOUNDS = (25, 50, 75)
LEVEL_MARGIN, THRESHOLD_MARGIN, EXTREME_MARGIN, PAIR_MARGIN = 5, 8, 10, 12

# Última variante de cada lista: reservada para transferencia.
STYLES = [
    {"bg": "#ffffff", "fg": "#222222", "bars": ["#3b6fb6"], "grid": True, "font": 14, "orient": "v", "thr": "#d62728"},
    {"bg": "#f7f4ea", "fg": "#333333", "bars": ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"], "grid": False,
     "font": 15, "orient": "v", "thr": "#d62728"},
    {"bg": "#ffffff", "fg": "#111111", "bars": ["#ff7f0e"], "grid": True, "font": 13, "orient": "h", "thr": "#1f77b4"},
    {"bg": "#eef3f8", "fg": "#1b2a3a", "bars": ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"], "grid": True,
     "font": 14, "orient": "h", "thr": "#b279a2"},
    {"bg": "#1e1e24", "fg": "#e8e8e8", "bars": ["#f2c14e", "#5ab1bb", "#f78154", "#a5c882", "#b8a1d9"], "grid": True,
     "font": 16, "orient": "v", "thr": "#ff5c8a"},
    # --- v2 (índices 5–12). Claves opcionales: width (ancho relativo de barra), dash (trazo del umbral).
    {"bg": "#fbfbfb", "fg": "#202020", "bars": ["#1b9e77"], "grid": True, "font": 14, "orient": "v", "thr": "#e7298a",
     "width": 0.45},
    {"bg": "#fff8f0", "fg": "#3a2a1a", "bars": ["#d95f02", "#7570b3", "#66a61e", "#e6ab02", "#a6761d"], "grid": False,
     "font": 13, "orient": "h", "thr": "#1f78b4", "width": 0.6},
    {"bg": "#f2f2f2", "fg": "#101010", "bars": ["#6a3d9a"], "grid": True, "font": 15, "orient": "v", "thr": "#33a02c",
     "width": 0.35, "dash": (4, 4)},
    {"bg": "#e8f4ec", "fg": "#0f2e1a", "bars": ["#b15928", "#1f78b4", "#fb9a99", "#cab2d6", "#ffff99"], "grid": True,
     "font": 14, "orient": "h", "thr": "#e31a1c", "width": 0.5, "dash": (12, 4)},
    {"bg": "#ffffff", "fg": "#333333", "bars": ["#8dd3c7", "#fb8072", "#80b1d3", "#fdb462", "#b3de69"], "grid": False,
     "font": 16, "orient": "v", "thr": "#000000", "width": 0.7},
    {"bg": "#f5f0ff", "fg": "#241a3a", "bars": ["#e41a1c"], "grid": True, "font": 13, "orient": "h", "thr": "#377eb8",
     "width": 0.4, "dash": (6, 6)},
    {"bg": "#fffde7", "fg": "#2b2b00", "bars": ["#984ea3", "#4daf4a", "#ff7f00", "#377eb8", "#a65628"], "grid": True,
     "font": 15, "orient": "v", "thr": "#e41a1c", "width": 0.5, "dash": (10, 5)},
    {"bg": "#eef0f2", "fg": "#15191d", "bars": ["#00798c"], "grid": False, "font": 14, "orient": "h", "thr": "#d1495b",
     "width": 0.65, "dash": (3, 6)},
]  # fmt: skip
V1_STYLE_COUNT = 5  # v1 sólo conoce los 5 primeros estilos (mismas elecciones aleatorias que antes)
# v2: repertorios disjuntos por partición; transferencia conserva el estilo oscuro reservado.
STYLE_POOLS_V2 = {
    "train": [0, 1, 2, 3, 5, 6],
    "validation": [7, 8],
    "calibration": [9, 10],
    "test": [11, 12],
    "transfer": [4],
}
# La fuente incrustada de Pillow (determinista entre máquinas) no dibuja vocales acentuadas.
TITLES = {"es": "Errores por servicio (hora anterior)", "en": "Errors per service (last hour)"}
THRESHOLD_WORD = {"es": "umbral", "en": "threshold"}
STATE = {
    "es": ["Adjunto la captura del panel de errores de la última hora.", "Os paso el panel de errores de monitorización.",
           "Captura del panel de errores por servicio adjunta.", "Incluyo la imagen del panel de errores de hoy."],
    "en": ["Attached is the screenshot of the last hour's error panel.", "Sharing the monitoring error panel.",
           "Error panel per service attached.", "Including today's error panel image."],
}  # fmt: skip
CLOSERS = {"es": ["", "¿Podéis revisarlo?", "Gracias."], "en": ["", "Can you check it?", "Thanks."]}
Q = {
    "any_over": {
        "es": ["¿Algún servicio supera la línea de umbral del panel?", "¿Hay alguna barra por encima del umbral?",
               "¿Rebasa algún servicio el umbral marcado?"],
        "en": ["Does any service exceed the panel's threshold line?", "Is any bar above the threshold?",
               "Does some service go past the marked threshold?"]},
    "service_over": {
        "es": ["¿Supera {a} la línea de umbral del panel?", "¿Está la barra de {a} por encima del umbral?",
               "¿Rebasa {a} el umbral marcado?"],
        "en": ["Does {a} exceed the panel's threshold line?", "Is the {a} bar above the threshold?",
               "Does {a} go past the marked threshold?"]},
    "compare": {
        "es": ["¿Tiene {a} más errores que {b} en el panel?", "¿Es la barra de {a} más alta que la de {b}?",
               "¿Registra {a} más errores que {b}?"],
        "en": ["Does {a} have more errors than {b} in the panel?", "Is the {a} bar higher than the {b} bar?",
               "Does {a} log more errors than {b}?"]},
    "top": {
        "es": ["¿Qué servicio muestra más errores en el panel?", "Indica el servicio con la barra más alta.",
               "¿Cuál es el servicio con más errores?"],
        "en": ["Which service shows the most errors in the panel?", "Name the service with the highest bar.",
               "Which service has the most errors?"]},
    "bottom": {
        "es": ["¿Qué servicio muestra menos errores en el panel?", "Indica el servicio con la barra más baja.",
               "¿Cuál es el servicio con menos errores?"],
        "en": ["Which service shows the fewest errors in the panel?", "Name the service with the lowest bar.",
               "Which service has the fewest errors?"]},
    "level": {
        "es": ["Evalúa el nivel de errores de {a} según el eje del panel.", "¿En qué tramo del eje está la barra de {a}?",
               "Clasifica los errores de {a} según la escala del panel."],
        "en": ["Rate the error level of {a} using the panel's axis.", "Which axis band is the {a} bar in?",
               "Classify {a}'s errors on the panel's scale."]},
}  # fmt: skip
RUBRIC = {
    "es": ["Menos de 25 errores", "Entre 25 y 50 errores", "Entre 50 y 75 errores", "Más de 75 errores"],
    "en": ["Fewer than 25 errors", "Between 25 and 50 errors", "Between 50 and 75 errors", "More than 75 errors"],
}
KINDS = {"noul": ["any_over", "service_over", "compare"], "choice": ["top", "bottom"], "score": ["level"]}
TYPE_WEIGHTS = {"noul": 0.4, "choice": 0.3, "score": 0.3}


def _pool(values: list, variant: Variant) -> list:
    if len(values) < 2:
        raise ValueError("Cada lista necesita al menos una variante main y una transfer")
    return values[:-1] if variant == "main" else values[-1:]


def level_of(v: int) -> int:
    return sum(v > b for b in LEVEL_BOUNDS)


def _ok_values(vals: list[int]) -> bool:
    s = sorted(vals)
    return (
        all(abs(v - b) >= LEVEL_MARGIN for v in vals for b in LEVEL_BOUNDS)
        and s[-1] - s[-2] >= EXTREME_MARGIN
        and s[1] - s[0] >= EXTREME_MARGIN
    )


def sample_facts(rng: random.Random, variant: Variant, style_pool: list[int] | None = None) -> dict[str, Any]:
    k = rng.choice([3, 4, 5])
    services = rng.sample(SERVICES, k)
    while True:
        values = [rng.randint(3, 97) for _ in range(k)]
        if _ok_values(values):
            break
    threshold = None
    if rng.random() < 0.7:
        top = max(values)
        # Mitad de los umbrales por encima de todas las barras (ninguna lo supera), si cabe.
        none_over = rng.random() < 0.5 and top + THRESHOLD_MARGIN <= 92
        lo, hi = (top + THRESHOLD_MARGIN, 92) if none_over else (12, max(12, top - THRESHOLD_MARGIN))
        for _ in range(200):
            t = rng.randint(lo, hi)
            if all(abs(v - t) >= THRESHOLD_MARGIN for v in values):
                threshold = t
                break
    if style_pool is not None:
        style = rng.choice(style_pool)
    else:  # v1: exactamente las mismas llamadas al RNG que en el piloto histórico
        style = rng.randrange(V1_STYLE_COUNT - 1) if variant == "main" else V1_STYLE_COUNT - 1
    return {"services": services, "values": values, "threshold": threshold, "style": style}


def render_chart(facts: dict[str, Any], lang: str) -> bytes:
    """PNG determinista del panel (mismos hechos → mismos bytes)."""
    from PIL import Image, ImageDraw, ImageFont

    st = STYLES[facts["style"]]
    font = ImageFont.load_default(size=st["font"])
    small = ImageFont.load_default(size=max(10, st["font"] - 3))
    im = Image.new("RGB", SIZE, st["bg"])
    d = ImageDraw.Draw(im)
    d.text((SIZE[0] // 2, 14), TITLES[lang], fill=st["fg"], font=font, anchor="mm")
    names, values = facts["services"], facts["values"]
    k = len(names)
    if st["orient"] == "v":
        x0, x1, y0, y1 = 56, 464, 34, 312  # área de trazado; y1 = valor 0
        def ypos(v):  # noqa: E306
            return y1 - (y1 - y0) * v / 100
        for tick in (0, 25, 50, 75, 100):
            y = ypos(tick)
            if st["grid"] and tick:
                d.line([(x0, y), (x1, y)], fill="#999999" if st["bg"] != "#1e1e24" else "#555555", width=1)
            d.text((x0 - 6, y), str(tick), fill=st["fg"], font=small, anchor="rm")
        d.line([(x0, y0), (x0, y1), (x1, y1)], fill=st["fg"], width=2)
        slot = (x1 - x0) / k
        for i, (name, v) in enumerate(zip(names, values, strict=True)):
            cx = x0 + slot * (i + 0.5)
            w = slot * st.get("width", 0.55)
            d.rectangle([cx - w / 2, ypos(v), cx + w / 2, y1], fill=st["bars"][i % len(st["bars"])])
            d.text((cx, y1 + 16), name, fill=st["fg"], font=font, anchor="mm")
        if facts["threshold"] is not None:
            y = ypos(facts["threshold"])
            on, off = st.get("dash", (8, 6))
            for xs in range(x0, x1, on + off):
                d.line([(xs, y), (min(xs + on, x1), y)], fill=st["thr"], width=3)
            d.text((x1, y - 10), THRESHOLD_WORD[lang], fill=st["thr"], font=small, anchor="rm")
    else:
        x0, x1, y0, y1 = 96, 460, 36, 318  # x0 = valor 0
        def xpos(v):  # noqa: E306
            return x0 + (x1 - x0) * v / 100
        for tick in (0, 25, 50, 75, 100):
            x = xpos(tick)
            if st["grid"] and tick:
                d.line([(x, y0), (x, y1)], fill="#999999", width=1)
            d.text((x, y1 + 12), str(tick), fill=st["fg"], font=small, anchor="mm")
        d.line([(x0, y0), (x0, y1), (x1, y1)], fill=st["fg"], width=2)
        slot = (y1 - y0) / k
        for i, (name, v) in enumerate(zip(names, values, strict=True)):
            cy = y0 + slot * (i + 0.5)
            h = slot * st.get("width", 0.55)
            d.rectangle([x0, cy - h / 2, xpos(v), cy + h / 2], fill=st["bars"][i % len(st["bars"])])
            d.text((x0 - 8, cy), name, fill=st["fg"], font=font, anchor="rm")
        if facts["threshold"] is not None:
            x = xpos(facts["threshold"])
            on, off = st.get("dash", (8, 6))
            for ys in range(y0, y1, on + off):
                d.line([(x, ys), (x, min(ys + on, y1))], fill=st["thr"], width=3)
            d.text((x + 4, y0 - 4), THRESHOLD_WORD[lang], fill=st["thr"], font=small, anchor="lb")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def _opaque_ids(rng: random.Random, n: int) -> list[str]:
    ids: list[str] = []
    while len(ids) < n:
        c = "o" + "".join(rng.choice("0123456789abcdef") for _ in range(6))
        if c not in ids:
            ids.append(c)
    return ids


def _build(kind: str, rng: random.Random, lang: str, f: dict[str, Any], variant: Variant, used: set):
    names, values, thr = f["services"], f["values"], f["threshold"]
    val = dict(zip(names, values, strict=True))
    phr = rng.choice(_pool(Q[kind][lang], variant))
    if kind == "any_over":
        return {"type": "noul", "instructions": phr}, {"label": int(max(values) > thr)}, "any_bar_above_threshold"
    if kind == "service_over":
        a = rng.choice([n for n in names if ("over", n) not in used])
        used.add(("over", a))
        return {"type": "noul", "instructions": phr.format(a=a)}, {"label": int(val[a] > thr)}, "bar_above_threshold"
    if kind == "compare":
        pairs = [(a, b) for a in names for b in names if a != b and abs(val[a] - val[b]) >= PAIR_MARGIN]
        pairs = [p for p in pairs if frozenset(p) not in used]
        a, b = rng.choice(pairs)
        used.add(frozenset((a, b)))
        return {"type": "noul", "instructions": phr.format(a=a, b=b)}, {"label": int(val[a] > val[b])}, "bar_compare"
    if kind in ("top", "bottom"):
        order = rng.sample(names, len(names))  # orden del mapa aleatorio; el serializador lo canoniza
        ids = _opaque_ids(rng, len(order))
        target = max(names, key=val.get) if kind == "top" else min(names, key=val.get)
        criteria = dict(zip(ids, order, strict=True))
        label = ids[order.index(target)]
        return {"type": "choice", "instructions": phr, "criteria": criteria}, {"class_id": label}, f"bar_{kind}"
    if kind == "level":
        a = rng.choice([n for n in names if ("level", n) not in used])
        used.add(("level", a))
        criteria, level = list(RUBRIC[lang]), level_of(val[a])
        method = "bar_level_axis_m4"
        if rng.random() < 0.5:  # rúbrica descendente: mismo significado por nivel, índice invertido
            criteria, level, method = criteria[::-1], len(criteria) - 1 - level, method + "_descending"
        return {"type": "score", "instructions": phr.format(a=a), "criteria": criteria}, {"level_index": level}, method
    raise ValueError(kind)


def _available(kind: str, f: dict[str, Any], used: set) -> bool:
    names, values = f["services"], f["values"]
    val = dict(zip(names, values, strict=True))
    if kind in ("any_over", "service_over") and f["threshold"] is None:
        return False
    if kind == "any_over":
        return "any_over" not in used
    if kind == "service_over":
        return any(("over", n) not in used for n in names)
    if kind == "compare":
        return any(
            frozenset((a, b)) not in used
            for a in names
            for b in names
            if a != b and abs(val[a] - val[b]) >= PAIR_MARGIN
        )
    if kind == "level":
        return any(("level", n) not in used for n in names)
    return kind not in used


def planned_split(group_ids: list[str], split_seed: int) -> dict[str, str]:
    """El mismo reparto que hará ``gso split --seed split_seed`` (misma función, mismos grupos)."""
    from .split import assign_groups

    return assign_groups(group_ids, split_seed)


def split_plan_sha256(assignment: dict[str, str]) -> str:
    return sha256_bytes("\n".join(f"{g}:{assignment[g]}" for g in sorted(assignment)).encode())


def generate_vision(
    n_cases: int,
    seed: int,
    variant: Variant = "main",
    questions_per_case: int = 3,
    version: VisionVersion = "v1",
    split_seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    """Ejemplos (dict validables), imágenes por referencia de contenido y auditoría por caso.

    v2 principal: cada caso sabe su partición antes de muestrearse; estilo y valores salen del
    repertorio y del flujo aleatorio de esa partición. ``split_seed`` debe ser la semilla de ``gso split``.
    """
    if n_cases < 1 or not 1 <= questions_per_case <= 6:
        raise ValueError("cases debe ser positivo y questions_per_case estar entre 1 y 6")
    gen_version = GENERATOR_VERSIONS[version]
    rng = random.Random(f"{gen_version}:{variant}:{seed}")
    prefix = {"v1": "vis", "v2": "vis2"}[version] + ("" if variant == "main" else "T")
    group_ids = [f"{prefix}-s{seed}-{case:05d}" for case in range(n_cases)]
    plan = planned_split(group_ids, split_seed) if version == "v2" and variant == "main" else None
    rngs = {}
    raws, images, audit = [], {}, []
    for case in range(n_cases):
        lang = "es" if case % 2 == 0 else "en"
        if version == "v2":
            part = plan[group_ids[case]] if plan is not None else "transfer"
            rng = rngs.setdefault(part, random.Random(f"{gen_version}:{variant}:{seed}:{part}"))
            pool = STYLE_POOLS_V2[part]
        else:
            part, pool = None, None
        for _ in range(100):
            f = sample_facts(rng, variant, pool)
            png = render_chart(f, lang)
            rel = content_address(png, "png")
            if rel not in images:
                break
        else:
            raise RuntimeError("No se encontró una imagen única")
        images[rel] = png
        state = " ".join(
            p for p in (rng.choice(_pool(STATE[lang], variant)), rng.choice(_pool(CLOSERS[lang], variant))) if p
        )
        group_id = group_ids[case]
        used: set = set()
        kinds: list[str] = []
        attempts = 0
        while len(kinds) < questions_per_case:
            attempts += 1
            if attempts > 200:
                raise RuntimeError("No hay suficientes preguntas disponibles para el caso")
            t = rng.choices(list(TYPE_WEIGHTS), weights=list(TYPE_WEIGHTS.values()))[0]
            options = [k for k in KINDS[t] if _available(k, f, used)]
            if not options:
                continue
            kind = rng.choice(options)
            question, target, method = _build(kind, rng, lang, f, variant, used)
            if kind in ("any_over", "top", "bottom"):
                used.add(kind)
            n = sum(k.startswith(kind) for k in kinds)
            kinds.append(kind if n == 0 else f"{kind}{n + 1}")
            raws.append(
                {
                    "schema_version": 1,
                    "id": f"{group_id}-{kinds[-1]}",
                    "group_id": group_id,
                    "task_family": f"panel_{kind}",
                    "language": lang,
                    "state": state,
                    "image_path": rel,
                    "question": question,
                    "target": target,
                    "provenance": {
                        "source": "synthetic_rule",
                        "generator_version": gen_version,
                        "label_method": method,
                        "template_id": f"style{f['style']}-{lang}-{variant}",
                        "seed": seed,
                    },
                }
            )
        entry = {"group_id": group_id, "image": rel, "lang": lang, **f, "questions": kinds}
        if part is not None:
            entry["planned_split"] = part
        audit.append(entry)
    return raws, images, audit


def write_vision_dataset(
    out_dir: Path,
    n_cases: int,
    seed: int,
    variant: Variant = "main",
    questions_per_case: int = 3,
    version: VisionVersion = "v1",
    split_seed: int = 0,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    raws, images, audit = generate_vision(n_cases, seed, variant, questions_per_case, version, split_seed)
    examples = [Example.model_validate(r) for r in raws]
    lines = [json.dumps(e.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for e in examples]
    data = ("\n".join(lines) + "\n").encode()
    target = out_dir / EXAMPLES_FILE
    if target.exists() and target.read_bytes() != data:
        raise FileExistsError(f"{target} existe con otro contenido; usa otro directorio")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for rel, png in images.items():
        path = out_dir / rel
        if not path.exists():
            path.write_bytes(png)
    target.write_bytes(data)
    (out_dir / "audit.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False, sort_keys=True) + "\n" for a in audit), encoding="utf-8"
    )
    manifest = {
        "generator_version": GENERATOR_VERSIONS[version],
        "variant": variant,
        "seed": seed,
        "n_cases": n_cases,
        "questions_per_case": questions_per_case,
        "examples": len(examples),
        "examples_sha256": sha256_bytes(data),
        "images": len(images),
        "images_sha256": sha256_bytes("\n".join(sorted(images)).encode()),
        "image_size": list(SIZE),
        "created_utc": datetime.now(UTC).isoformat(),
        "note": "Sintético por reglas; la respuesta sólo está en la imagen. 'transfer' usa estilos reservados.",
    }
    if version == "v2":
        pools = STYLE_POOLS_V2 if variant == "main" else {"transfer": STYLE_POOLS_V2["transfer"]}
        manifest["style_pools"] = pools
        if variant == "main":
            plan = {a["group_id"]: a["planned_split"] for a in audit}
            manifest["planned_split"] = {"seed": split_seed, "assignment_sha256": split_plan_sha256(plan)}
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
