"""Generador sintético de incidencias de soporte con etiquetas por regla (spec §5.2).

Cada caso (= ``group_id``) combina hechos controlados; cada pregunta Noul tiene una
regla de etiquetado comprobable sobre esos hechos. Se incluyen negaciones y
distractores que comparten vocabulario con el hecho positivo («no quiero una
devolución», «el año pasado me devolvieron…»), y varias preguntas por estado con
respuestas distintas. Es un banco de pruebas de humo: las frases proceden de un
repertorio finito y un buen resultado no demuestra comprensión general.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..contracts import Example
from .dataset import EXAMPLES_FILE, MANIFEST_FILE, sha256_bytes

GENERATOR_VERSION = "support-noul-v1"

# Valores posibles de cada hecho y probabilidad de muestreo.
FACTS: dict[str, list[tuple[str, float]]] = {
    "refund": [("requested", 0.4), ("declined", 0.2), ("past", 0.2), ("absent", 0.2)],
    "charge": [("double", 0.45), ("denied", 0.2), ("absent", 0.35)],
    "fault": [("active", 0.45), ("resolved", 0.25), ("absent", 0.3)],
    "access": [("locked", 0.4), ("ok", 0.2), ("absent", 0.4)],
}

SENTENCES: dict[str, dict[str, dict[str, list[str]]]] = {
    "es": {
        "refund": {
            "requested": [
                "Solicito que me devuelvan el importe.",
                "Quiero el reembolso de lo que pagué.",
                "Por favor, devuélvanme el dinero.",
            ],
            "declined": [
                "No quiero una devolución, solo que lo arreglen.",
                "No hace falta que me reembolsen nada.",
                "No pido que me devuelvan el dinero.",
            ],
            "past": [
                "El año pasado ya me hicieron una devolución por otro pedido.",
                "Hace meses me reembolsaron una compra distinta sin problema.",
            ],
        },
        "charge": {
            "double": [
                "Me han cobrado dos veces {item}.",
                "Veo un cargo duplicado por {item} en mi tarjeta.",
                "Aparecen dos cobros idénticos por {item}.",
            ],
            "denied": [
                "El cobro por {item} es correcto, solo aparece una vez.",
                "No hay ningún cargo repetido en mi cuenta.",
            ],
        },
        "fault": {
            "active": [
                "La aplicación se cierra cada vez que abro {feature}.",
                "Desde ayer {feature} no carga y muestra un error.",
                "Sigue fallando {feature} con un error 500.",
            ],
            "resolved": [
                "El error de {feature} de ayer ya se solucionó solo.",
                "Falló {feature} por la mañana, pero ahora funciona bien.",
            ],
        },
        "access": {
            "locked": [
                "No puedo iniciar sesión en mi cuenta.",
                "Mi cuenta está bloqueada y no me deja entrar.",
                "Me rechaza la contraseña y no consigo acceder.",
            ],
            "ok": [
                "Puedo entrar en mi cuenta sin problemas.",
                "El acceso a la cuenta funciona con normalidad.",
            ],
        },
    },
    "en": {
        "refund": {
            "requested": [
                "I would like the amount refunded.",
                "Please give me my money back.",
                "I am asking for a refund of what I paid.",
            ],
            "declined": [
                "I do not want a refund, just fix it.",
                "There is no need to refund me anything.",
                "I am not asking for my money back.",
            ],
            "past": [
                "Last year you already refunded me for a different order.",
                "A few months ago I got a refund for another purchase without issues.",
            ],
        },
        "charge": {
            "double": [
                "I was charged twice for {item}.",
                "I see a duplicate charge for {item} on my card.",
                "There are two identical charges for {item}.",
            ],
            "denied": [
                "The charge for {item} is correct and appears only once.",
                "There is no repeated charge on my account.",
            ],
        },
        "fault": {
            "active": [
                "The app crashes every time I open {feature}.",
                "Since yesterday {feature} does not load and shows an error.",
                "{Feature} keeps failing with a 500 error.",
            ],
            "resolved": [
                "Yesterday's error in {feature} has already fixed itself.",
                "{Feature} failed this morning, but it works fine now.",
            ],
        },
        "access": {
            "locked": [
                "I cannot log in to my account.",
                "My account is locked and will not let me in.",
                "It rejects my password and I cannot get in.",
            ],
            "ok": [
                "I can access my account without problems.",
                "Logging in to the account works normally.",
            ],
        },
    },
}

ITEMS = {
    "es": ["el pedido 4821", "la suscripción mensual", "el plan anual", "la factura de marzo"],
    "en": ["order 4821", "the monthly subscription", "the annual plan", "the March invoice"],
}
FEATURES = {
    "es": ["la pantalla de pagos", "la exportación de informes", "la sección de facturas", "la página de perfil"],
    "en": ["the payments screen", "the report export", "the invoices section", "the profile page"],
}
OPENERS = {"es": ["Hola,", "Buenos días.", "Escribo por lo siguiente:", ""], "en": ["Hi,", "Good morning.", "Writing about the following:", ""]}
CLOSERS = {"es": ["Gracias.", "Un saludo.", "Quedo a la espera.", ""], "en": ["Thanks.", "Best regards.", "Looking forward to your reply.", ""]}


@dataclass(frozen=True)
class QuestionKind:
    kind: str
    task_family: str
    label_method: str
    phrasings: dict[str, list[str]]

    def label(self, facts: dict[str, str]) -> int:
        return int(RULES[self.kind](facts))


RULES = {
    "refund": lambda f: f["refund"] == "requested",
    "double": lambda f: f["charge"] == "double",
    "fault": lambda f: f["fault"] == "active",
    "locked": lambda f: f["access"] == "locked",
    "escalate": lambda f: f["charge"] == "double" and f["refund"] == "requested",
}

QUESTIONS = [
    QuestionKind(
        "refund",
        "refund",
        "explicit_refund_request",
        {
            "es": ["¿El cliente solicita una devolución del dinero?", "¿Pide el cliente un reembolso en este mensaje?"],
            "en": ["Does the customer ask for a refund?", "Is the customer requesting their money back in this message?"],
        },
    ),
    QuestionKind(
        "double",
        "duplicate_charge",
        "explicit_duplicate_charge",
        {
            "es": ["¿Se informa de un cobro duplicado?", "¿Indica el cliente que se le ha cobrado dos veces?"],
            "en": ["Does the message report a duplicate charge?", "Does the customer say they were charged twice?"],
        },
    ),
    QuestionKind(
        "fault",
        "service_fault",
        "active_fault_statement",
        {
            "es": ["¿Se describe un fallo técnico que sigue activo?", "¿Hay un error del servicio que todavía no está resuelto?"],
            "en": ["Is an ongoing technical failure described?", "Is there a service error that is still unresolved?"],
        },
    ),
    QuestionKind(
        "locked",
        "account_access",
        "explicit_access_block",
        {
            "es": ["¿El cliente no puede acceder a su cuenta?", "¿Indica el mensaje que el acceso a la cuenta está bloqueado?"],
            "en": ["Is the customer unable to access their account?", "Does the message say that account access is blocked?"],
        },
    ),
    QuestionKind(
        "escalate",
        "billing_escalation",
        "policy_and(duplicate_charge,refund_request)",
        {
            "es": [
                "Política: se escala a facturación solo si hay un cobro duplicado y además el cliente pide la "
                "devolución. ¿Debe escalarse a facturación?"
            ],
            "en": [
                "Policy: escalate to billing only if there is a duplicate charge and the customer also asks for a "
                "refund. Should this be escalated to billing?"
            ],
        },
    ),
]


def _pick(rng: random.Random, options: list[tuple[str, float]]) -> str:
    values, weights = zip(*options, strict=True)
    return rng.choices(values, weights=weights, k=1)[0]


def sample_facts(rng: random.Random) -> dict[str, str]:
    while True:
        facts = {name: _pick(rng, opts) for name, opts in FACTS.items()}
        if any(v != "absent" for v in facts.values()):
            return facts


def render_state(rng: random.Random, lang: str, facts: dict[str, str]) -> tuple[Any, str]:
    item, feature = rng.choice(ITEMS[lang]), rng.choice(FEATURES[lang])
    sentences = []
    for name, value in facts.items():
        if value == "absent":
            continue
        s = rng.choice(SENTENCES[lang][name][value])
        s = s.format(item=item, feature=feature, Feature=feature[:1].upper() + feature[1:])
        sentences.append(s)
    rng.shuffle(sentences)
    parts = [p for p in (rng.choice(OPENERS[lang]), *sentences, rng.choice(CLOSERS[lang])) if p]
    message = " ".join(parts)
    if rng.random() < 0.25:
        state = {
            "channel": rng.choice(["email", "chat", "web_form"]),
            "customer_tier": rng.choice(["standard", "premium"]),
            "message": message,
        }
        return state, f"{lang}-json"
    return message, f"{lang}-text"


def generate(n_cases: int, seed: int, questions_per_case: int = 3) -> tuple[list[Example], list[dict]]:
    """Devuelve los ejemplos y, aparte, los hechos de cada caso (sólo para auditoría/tests)."""
    rng = random.Random(seed)
    examples: list[Example] = []
    audit: list[dict] = []
    for case in range(n_cases):
        lang = "es" if case % 2 == 0 else "en"
        facts = sample_facts(rng)
        state, template_id = render_state(rng, lang, facts)
        group_id = f"case-s{seed}-{case:04d}"
        kinds = rng.sample(QUESTIONS, questions_per_case)
        audit.append({"group_id": group_id, "facts": facts})
        for qk in kinds:
            raw = {
                "schema_version": 1,
                "id": f"n-s{seed}-{case:04d}-{qk.kind}",
                "group_id": group_id,
                "task_family": qk.task_family,
                "language": lang,
                "state": state,
                "image_path": None,
                "question": {"type": "noul", "instructions": rng.choice(qk.phrasings[lang])},
                "target": {"label": qk.label(facts)},
                "provenance": {
                    "source": "synthetic_rule",
                    "generator_version": GENERATOR_VERSION,
                    "label_method": qk.label_method,
                    "template_id": template_id,
                    "seed": seed,
                },
            }
            examples.append(Example.model_validate(raw))
    return examples, audit


def write_dataset(out_dir: Path, n_cases: int, seed: int, questions_per_case: int = 3) -> dict[str, Any]:
    out_dir = Path(out_dir)
    target = out_dir / EXAMPLES_FILE
    examples, _ = generate(n_cases, seed, questions_per_case)
    lines = [json.dumps(e.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for e in examples]
    data = ("\n".join(lines) + "\n").encode()
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"{target} existe con otro contenido; usa otro directorio")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "n_cases": n_cases,
        "questions_per_case": questions_per_case,
        "examples": len(examples),
        "examples_sha256": sha256_bytes(data),
        "created_utc": datetime.now(UTC).isoformat(),
        "note": "Sintético por reglas; banco de humo, no evidencia de generalización.",
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
