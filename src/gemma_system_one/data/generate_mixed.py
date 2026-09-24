"""Generador mixto Noul/Choice/Score de incidencias de soporte (``support-mixed-v1``).

Cada caso (= ``group_id``) combina hechos controlados; cada pregunta tiene una regla de
etiquetado comprobable. Choice usa IDs opacos aleatorios por pregunta (el significado
está en la descripción) y conjuntos de opciones variables (K=3–6 o taxonomía de 4),
con ``other``/``none`` cuando el conjunto no es exhaustivo. Score usa rúbricas ordenadas
con M=3–5.

Plantillas reservadas: en cada lista de variantes (frases, preguntas, opciones,
rúbricas, saludos) la **última** variante sólo se usa con ``variant="transfer"``;
``variant="main"`` usa las demás. Así el conjunto de transferencia mide plantillas de
superficie no vistas. Las familias de tarea y las reglas son las mismas: no mide
transferencia a tareas nuevas.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..contracts import Example
from .dataset import EXAMPLES_FILE, MANIFEST_FILE, sha256_bytes

GENERATOR_VERSIONS = {
    # v1: rúbricas ascendentes; los artefactos históricos no son regenerables (P2-2).
    "v1": "support-mixed-v1",
    # v2: rúbricas ascendentes o descendentes al 50 % (la etiqueta sigue a la descripción)
    #     y minúscula tras un saludo terminado en coma. Corrige el atajo posicional del piloto v1.
    "v2": "support-mixed-v2",
    # v3: v2 + cláusula de política en las preguntas de fallo técnico (Noul, Choice y Score):
    #     los problemas de acceso a la cuenta no cuentan como fallo técnico. Es la regla con
    #     la que v2 ya etiquetaba, pero no la decía (ambigüedad observada en el piloto v2).
    "v3": "support-mixed-v3",
    # v4: v3 sin la pista de K en fault_type (revisión de la fase 6b). En v3, los casos «other»
    #     tenían como mucho 5 opciones (sólo 4 categorías no verdaderas disponibles), así que K = 6
    #     revelaba que la respuesta no era «other». v4 añade tres categorías distractoras que ningún
    #     estado describe y sortea «other» con la misma probabilidad para cualquier K. Nueva secuencia
    #     aleatoria: no es emparejable con v3.
    "v4": "support-mixed-v4",
    # v5: v4 con definiciones de fault_type excluyentes («aplicación» no incluye lentitud ni pérdida
    #     de datos, más una cláusula de precedencia en la instrucción) y la composición de las opciones
    #     sorteada sin mirar los hechos: siempre «none» y «other» más K − 2 categorías; la etiqueta se
    #     deduce después (revisión de las fases 6d/6e). Nueva secuencia aleatoria.
    "v5": "support-mixed-v5",
}
GENERATOR_VERSION = GENERATOR_VERSIONS["v3"]
Variant = Literal["main", "transfer"]
Version = Literal["v1", "v2", "v3", "v4", "v5"]
# v3 reutiliza la secuencia aleatoria de v2: mismos estados, etiquetas, grupos e IDs; sólo
# cambia el texto de las instrucciones de fallo técnico. Así la comparación v2/v3 es emparejada.
_RNG_STREAM = {
    "v1": "support-mixed-v1",
    "v2": "support-mixed-v2",
    "v3": "support-mixed-v2",
    "v4": "support-mixed-v4",
    "v5": "support-mixed-v5",
}
FAULT_FAMILIES = ("service_fault", "fault_type", "fault_severity")
ACCESS_POLICY_CLAUSE = {
    "es": "Los problemas de acceso a la cuenta (inicio de sesión, contraseña o bloqueo) no cuentan como fallo técnico.",
    "en": "Account access problems (login, password or lockout) do not count as technical failures.",
}

FACTS = {
    "refund": [("requested", 0.35), ("declined", 0.2), ("past", 0.15), ("absent", 0.3)],
    "charge": [("double", 0.35), ("denied", 0.15), ("absent", 0.5)],
    "fault": [("active", 0.45), ("resolved", 0.25), ("absent", 0.3)],
    "fault_kind": [("network", 0.25), ("performance", 0.25), ("data", 0.25), ("application", 0.25)],
    "access": [("locked", 0.3), ("ok", 0.15), ("absent", 0.55)],
}

S: dict[str, dict[str, dict[str, list[str]]]] = {
    "es": {
        "refund.requested": {"v": ["Solicito que me devuelvan el importe.", "Quiero el reembolso de lo que pagué.", "Por favor, devuélvanme el dinero.", "Reclamo que me reintegren lo cobrado."]},
        "refund.declined": {"v": ["No quiero una devolución, solo que lo arreglen.", "No hace falta que me reembolsen nada.", "No pido que me devuelvan el dinero.", "El reintegro no me interesa; prefiero que lo solucionen."]},
        "refund.past": {"v": ["El año pasado ya me hicieron una devolución por otro pedido.", "Hace meses me reembolsaron una compra distinta sin problema.", "En otra ocasión anterior me reintegraron un cargo sin incidencias."]},
        "charge.double": {"v": ["Me han cobrado dos veces {item}.", "Veo un cargo duplicado por {item} en mi tarjeta.", "Aparecen dos cobros idénticos por {item}.", "El banco muestra {item} cargado por partida doble."]},
        "charge.denied": {"v": ["El cobro por {item} es correcto, solo aparece una vez.", "No hay ningún cargo repetido en mi cuenta.", "He revisado el extracto y {item} figura una única vez."]},
        "fault.active.network": {"v": ["No resuelve el DNS del servicio desde esta mañana.", "La VPN se desconecta cada pocos minutos.", "No hay conexión con el servidor desde la oficina.", "Los equipos pierden la conectividad con la plataforma continuamente."]},
        "fault.active.performance": {"v": ["La aplicación va extremadamente lenta al abrir {feature}.", "Cada acción tarda más de un minuto en responder.", "El sistema responde con muchísimo retraso.", "Todo funciona, pero con una lentitud desesperante."]},
        "fault.active.data": {"v": ["Han desaparecido registros de {feature}.", "Los datos del historial aparecen corruptos.", "Se han perdido los cambios guardados ayer.", "Faltan entradas que antes estaban almacenadas."]},
        "fault.active.application": {"v": ["La aplicación se cierra cada vez que abro {feature}.", "Desde ayer {feature} no carga y muestra un error.", "Sigue fallando {feature} con un error 500.", "Al pulsar en {feature} aparece una pantalla en blanco."]},
        "fault.resolved.network": {"v": ["Ayer se cayó la VPN, pero ya funciona con normalidad.", "Hubo un problema de DNS esta mañana que ya está resuelto.", "La conexión falló un rato, aunque se restableció enseguida."]},
        "fault.resolved.performance": {"v": ["La lentitud de ayer ya se ha solucionado.", "Esta mañana el sistema iba muy lento, pero ya responde bien.", "El retraso en las respuestas desapareció hace horas."]},
        "fault.resolved.data": {"v": ["Faltaban registros de {feature}, pero ya se han recuperado.", "Los datos corruptos de ayer ya están restaurados.", "La información perdida volvió a aparecer tras la copia de seguridad."]},
        "fault.resolved.application": {"v": ["El error de {feature} de ayer ya se solucionó solo.", "Falló {feature} por la mañana, pero ahora funciona bien.", "La pantalla en blanco de {feature} ya no aparece."]},
        "access.locked": {"v": ["No puedo iniciar sesión en mi cuenta.", "Mi cuenta está bloqueada y no me deja entrar.", "Me rechaza la contraseña y no consigo acceder.", "El acceso a mi perfil está denegado."]},
        "access.ok": {"v": ["Puedo entrar en mi cuenta sin problemas.", "El acceso a la cuenta funciona con normalidad.", "Inicio sesión sin ninguna dificultad."]},
        "impact": {"v": ["Afecta a {x} de los {y} usuarios de la empresa.", "Somos {y} usuarios y {x} tienen el problema.", "El problema lo sufren {x} personas de un total de {y}."]},
    },
    "en": {
        "refund.requested": {"v": ["I would like the amount refunded.", "Please give me my money back.", "I am asking for a refund of what I paid.", "I demand reimbursement of the charge."]},
        "refund.declined": {"v": ["I do not want a refund, just fix it.", "There is no need to refund me anything.", "I am not asking for my money back.", "Reimbursement does not interest me; I prefer a fix."]},
        "refund.past": {"v": ["Last year you already refunded me for a different order.", "A few months ago I got a refund for another purchase without issues.", "On an earlier occasion you reimbursed a charge without problems."]},
        "charge.double": {"v": ["I was charged twice for {item}.", "I see a duplicate charge for {item} on my card.", "There are two identical charges for {item}.", "My bank shows {item} billed two times."]},
        "charge.denied": {"v": ["The charge for {item} is correct and appears only once.", "There is no repeated charge on my account.", "I checked the statement and {item} appears a single time."]},
        "fault.active.network": {"v": ["The service DNS has not resolved since this morning.", "The VPN disconnects every few minutes.", "There is no connection to the server from the office.", "Our machines keep losing connectivity to the platform."]},
        "fault.active.performance": {"v": ["The app is extremely slow when opening {feature}.", "Every action takes more than a minute to respond.", "The system responds with a huge delay.", "Everything works, but painfully slowly."]},
        "fault.active.data": {"v": ["Records have disappeared from {feature}.", "The history data looks corrupted.", "The changes saved yesterday have been lost.", "Entries that were stored before are missing."]},
        "fault.active.application": {"v": ["The app crashes every time I open {feature}.", "Since yesterday {feature} does not load and shows an error.", "{Feature} keeps failing with a 500 error.", "Clicking on {feature} shows a blank screen."]},
        "fault.resolved.network": {"v": ["The VPN went down yesterday, but it works normally now.", "There was a DNS problem this morning that is already fixed.", "The connection dropped for a while, but it came back quickly."]},
        "fault.resolved.performance": {"v": ["Yesterday's slowness has already been fixed.", "This morning the system was very slow, but it responds well now.", "The response delay went away hours ago."]},
        "fault.resolved.data": {"v": ["Records were missing from {feature}, but they have been recovered.", "Yesterday's corrupted data has already been restored.", "The lost information reappeared after the backup."]},
        "fault.resolved.application": {"v": ["Yesterday's error in {feature} has already fixed itself.", "{Feature} failed this morning, but it works fine now.", "The blank screen in {feature} no longer appears."]},
        "access.locked": {"v": ["I cannot log in to my account.", "My account is locked and will not let me in.", "It rejects my password and I cannot get in.", "Access to my profile is denied."]},
        "access.ok": {"v": ["I can access my account without problems.", "Logging in to the account works normally.", "I sign in without any difficulty."]},
        "impact": {"v": ["It affects {x} of the company's {y} users.", "We are {y} users and {x} have the problem.", "The problem hits {x} people out of a total of {y}."]},
    },
}

ITEMS = {
    "es": ["el pedido 4821", "la suscripción mensual", "el plan anual", "la factura de marzo", "la renovación de mayo"],
    "en": ["order 4821", "the monthly subscription", "the annual plan", "the March invoice", "the May renewal"],
}
FEATURES = {
    "es": ["la pantalla de pagos", "la exportación de informes", "la sección de facturas", "la página de perfil", "la agenda compartida"],
    "en": ["the payments screen", "the report export", "the invoices section", "the profile page", "the shared calendar"],
}
OPENERS = {"es": ["Hola,", "Buenos días.", "Escribo por lo siguiente:", "", "Estimado equipo:"], "en": ["Hi,", "Good morning.", "Writing about the following:", "", "Dear team,"]}
CLOSERS = {"es": ["Gracias.", "Un saludo.", "Quedo a la espera.", "", "Atentamente."], "en": ["Thanks.", "Best regards.", "Looking forward to your reply.", "", "Sincerely."]}

NOUL_Q = {
    "refund": ("refund", "explicit_refund_request", {
        "es": ["¿El cliente solicita una devolución del dinero?", "¿Pide el cliente un reembolso en este mensaje?", "¿Reclama el remitente que se le reintegre lo pagado?"],
        "en": ["Does the customer ask for a refund?", "Is the customer requesting their money back in this message?", "Is the sender claiming reimbursement of the payment?"]}),
    "double": ("duplicate_charge", "explicit_duplicate_charge", {
        "es": ["¿Se informa de un cobro duplicado?", "¿Indica el cliente que se le ha cobrado dos veces?", "¿Menciona el mensaje un cargo repetido?"],
        "en": ["Does the message report a duplicate charge?", "Does the customer say they were charged twice?", "Is a repeated charge mentioned?"]}),
    "fault": ("service_fault", "active_fault_statement", {
        "es": ["¿Se describe un fallo técnico que sigue activo?", "¿Hay un error del servicio que todavía no está resuelto?", "¿Persiste algún problema técnico en este momento?"],
        "en": ["Is an ongoing technical failure described?", "Is there a service error that is still unresolved?", "Does some technical problem persist right now?"]}),
    "locked": ("account_access", "explicit_access_block", {
        "es": ["¿El cliente no puede acceder a su cuenta?", "¿Indica el mensaje que el acceso a la cuenta está bloqueado?", "¿Tiene el usuario impedida la entrada a su perfil?"],
        "en": ["Is the customer unable to access their account?", "Does the message say that account access is blocked?", "Is the user prevented from entering their profile?"]}),
    "escalate": ("billing_escalation", "policy_and(duplicate_charge,refund_request)", {
        "es": ["Política: se escala a facturación solo si hay un cobro duplicado y además el cliente pide la devolución. ¿Debe escalarse a facturación?",
               "Según la norma interna, facturación solo interviene cuando concurren un cargo duplicado y una petición de reembolso. ¿Debe intervenir?",
               "Regla: pasar a pagos únicamente si existe un cobro repetido y también se reclama el dinero. ¿Hay que pasarlo a pagos?"],
        "en": ["Policy: escalate to billing only if there is a duplicate charge and the customer also asks for a refund. Should this be escalated to billing?",
               "Under the internal rule, billing only steps in when a duplicate charge and a refund request both occur. Should billing step in?",
               "Rule: hand over to payments only if a repeated charge exists and the money is also claimed. Should it be handed over?"]}),
}
NOUL_RULES = {
    "refund": lambda f: f["refund"] == "requested",
    "double": lambda f: f["charge"] == "double",
    "fault": lambda f: f["fault"] == "active",
    "locked": lambda f: f["access"] == "locked",
    "escalate": lambda f: f["charge"] == "double" and f["refund"] == "requested",
}

FAULT_KIND_INSTR = {
    "es": ["¿Qué tipo de fallo técnico describe el mensaje?", "Clasifica el fallo técnico descrito.", "Indica la categoría del problema técnico mencionado."],
    "en": ["What kind of technical failure does the message describe?", "Classify the described technical failure.", "State the category of the technical problem mentioned."],
}
FAULT_KIND_OPTIONS = {
    "es": {
        "network": ["Red o conectividad (DNS, VPN, conexión)", "Problema de conexión o de red", "Fallo de conectividad"],
        "performance": ["Lentitud o rendimiento", "El sistema responde con mucho retraso", "Problema de velocidad"],
        "data": ["Pérdida o corrupción de datos", "Datos desaparecidos o dañados", "Información perdida o estropeada"],
        "application": ["Error en una pantalla o función de la aplicación", "Una función de la aplicación falla o no carga", "Fallo de una sección de la app"],
        "none": ["No se describe ningún fallo técnico", "El mensaje no menciona fallos técnicos", "No hay ningún problema técnico"],
        "other": ["Otro tipo de fallo técnico", "Un fallo técnico distinto de los anteriores", "Algún otro fallo técnico"],
    },
    "en": {
        "network": ["Network or connectivity (DNS, VPN, connection)", "Connection or network problem", "Connectivity failure"],
        "performance": ["Slowness or performance", "The system responds with a long delay", "Speed problem"],
        "data": ["Data loss or corruption", "Missing or damaged data", "Lost or broken information"],
        "application": ["Error in an app screen or feature", "An app feature fails or does not load", "A section of the app breaks"],
        "none": ["No technical failure is described", "The message mentions no technical failures", "There is no technical problem at all"],
        "other": ["Another type of technical failure", "A technical failure different from the above", "Some other technical failure"],
    },
}
# v4: categorías de fallo que ningún estado del generador describe (comprobado: ninguna palabra clave
# aparece en los estados de pilot_v3 ni de final8). Nunca son la respuesta; la última redacción es
# de transfer. «hardware» e «install» usan los mismos textos que data/derive.NEVER_TRUE_FAULT_OPTIONS.
FAULT_DISTRACTOR_OPTIONS = {
    "es": {
        "hardware": [
            "Avería de un dispositivo físico (impresora, lector o terminal)",
            "Un equipo físico está estropeado",
            "Fallo de hardware",
        ],
        "install": [
            "Fallo al instalar o actualizar la aplicación",
            "La instalación o la actualización no se completa",
            "Problema de instalación",
        ],
        "notifications": [
            "Las notificaciones o avisos no llegan",
            "No se reciben las notificaciones de la aplicación",
            "Fallo de notificaciones",
        ],
    },
    "en": {
        "hardware": [
            "A physical device is broken (printer, reader or terminal)",
            "Some physical equipment is damaged",
            "Hardware failure",
        ],
        "install": [
            "The app fails to install or update",
            "Installation or update does not complete",
            "Installation problem",
        ],
        "notifications": [
            "Notifications or alerts do not arrive",
            "App notifications are not received",
            "Notification failure",
        ],
    },
}
# v5: definiciones mutuamente excluyentes. Los estados de rendimiento y de datos mencionan funciones
# de la app («al abrir {feature}»); en v3/v4 «Error en una pantalla o función de la aplicación» los
# abarcaba (fase 6e). La última redacción de cada lista es de transfer.
FAULT_KIND_OPTIONS_V5 = {
    "es": {
        **FAULT_KIND_OPTIONS["es"],
        "performance": [
            "Lentitud o rendimiento (funciona, pero tarda mucho)",
            "El sistema responde con mucho retraso",
            "Problema de velocidad",
        ],
        "application": [
            "Error, cierre o pantalla que no carga en una función (no incluye lentitud ni pérdida de datos)",
            "Una función de la aplicación da error o no carga, sin ir lenta ni perder datos",
            "Fallo de una sección de la app (ni lentitud ni datos perdidos)",
        ],
    },
    "en": {
        **FAULT_KIND_OPTIONS["en"],
        "performance": [
            "Slowness or performance (it works, but takes very long)",
            "The system responds with a long delay",
            "Speed problem",
        ],
        "application": [
            "An error, crash or screen that does not load in a feature (not slowness or data loss)",
            "An app feature errors out or does not load, without being slow or losing data",
            "A section of the app breaks (neither slowness nor lost data)",
        ],
    },
}
FAULT_TYPE_CLAUSE_V5 = {
    "es": "Clasifica por la naturaleza del fallo: la lentitud es rendimiento y los datos perdidos o dañados son datos, aunque ocurran en una función de la aplicación.",
    "en": "Classify by the nature of the failure: slowness is performance and lost or damaged data is data, even if it happens in an app feature.",
}
FAULT_KIND_K = (3, 6)  # K de entrenamiento en fault_type (igual que v3)
OTHER_RATE = 0.2  # con fallo activo o resuelto: proporción de conjuntos sin la categoría real

TEAM_NAMES = {
    # main: nombres 0-1; transfer: 2. La política usa los mismos nombres que las opciones.
    "es": {"billing": ["Facturación", "Equipo de facturación", "Departamento de pagos"],
           "accounts": ["Cuentas", "Equipo de cuentas", "Gestión de usuarios"],
           "technical": ["Soporte técnico", "Equipo de soporte técnico", "Ingeniería de incidencias"],
           "general": ["Atención general", "Equipo de atención general", "Atención al cliente básica"]},
    "en": {"billing": ["Billing", "Billing team", "Payments department"],
           "accounts": ["Accounts", "Accounts team", "User management"],
           "technical": ["Technical support", "Technical support team", "Incident engineering"],
           "general": ["General care", "General care team", "Basic customer service"]},
}
TEAM_POLICY = {
    "es": "Asigna el equipo según esta política: {billing} si hay un cobro duplicado o se pide una devolución; si no, {accounts} si el cliente no puede acceder a su cuenta; si no, {technical} si hay un fallo técnico activo; en cualquier otro caso, {general}. ¿Qué equipo corresponde?",
    "en": "Assign the team with this policy: {billing} if there is a duplicate charge or a refund is requested; otherwise {accounts} if the customer cannot access their account; otherwise {technical} if there is an active technical failure; in any other case, {general}. Which team applies?",
}
TEAM_RULE = lambda f: (  # noqa: E731
    "billing" if f["charge"] == "double" or f["refund"] == "requested"
    else "accounts" if f["access"] == "locked"
    else "technical" if f["fault"] == "active"
    else "general"
)

IMPACT_INSTR = {
    "es": ["Evalúa el alcance de la incidencia según los usuarios afectados.", "Valora cuántos usuarios se ven afectados por la incidencia.", "¿Qué parte de los usuarios sufre el problema?"],
    "en": ["Rate the scope of the incident by the affected users.", "Assess how many users are affected by the incident.", "What share of the users suffers the problem?"],
}
IMPACT_RUBRICS = {
    # m -> variantes (main: 0; transfer: 1)
    "es": {
        3: [["Ningún usuario afectado", "Algunos usuarios, pero no todos", "Todos los usuarios"],
            ["Nadie", "Una parte de los usuarios", "La totalidad de los usuarios"]],
        4: [["Ningún usuario", "Menos de la mitad de los usuarios", "La mitad o más, pero no todos", "Todos los usuarios"],
            ["Nadie", "Una minoría (menos del 50 %)", "Una mayoría incompleta (50 % o más)", "La totalidad"]],
        5: [["Ninguno (0 %)", "Hasta un 25 %", "Más del 25 % y hasta el 50 %", "Más del 50 %, pero no todos", "Todos (100 %)"],
            ["0 %", "Un cuarto o menos", "Entre un cuarto y la mitad", "Más de la mitad sin llegar al total", "El total"]],
    },
    "en": {
        3: [["No users affected", "Some users, but not all", "All users"],
            ["Nobody", "A portion of the users", "The entirety of the users"]],
        4: [["No users", "Less than half of the users", "Half or more, but not all", "All users"],
            ["Nobody", "A minority (under 50%)", "An incomplete majority (50% or more)", "Everyone"]],
        5: [["None (0%)", "Up to 25%", "More than 25% and up to 50%", "More than 50%, but not all", "All (100%)"],
            ["0%", "A quarter or less", "Between a quarter and half", "Over half but short of everyone", "The total"]],
    },
}
SEVERITY_INSTR = {
    "es": ["Evalúa la situación técnica descrita.", "Valora el estado del problema técnico.", "¿En qué estado está la incidencia técnica?"],
    "en": ["Rate the technical situation described.", "Assess the state of the technical problem.", "What state is the technical incident in?"],
}
SEVERITY_RUBRICS = {
    "es": [["No se describe ningún fallo técnico", "Hubo un fallo técnico, pero ya está resuelto", "Hay un fallo técnico activo"],
           ["Sin incidencia técnica", "Incidencia ya solucionada", "Incidencia técnica en curso"]],
    "en": [["No technical failure is described", "There was a technical failure, but it is resolved", "There is an active technical failure"],
           ["No technical incident", "Incident already fixed", "Technical incident in progress"]],
}


def _pool(values: list, variant: Variant) -> list:
    if len(values) < 2:
        raise ValueError("Cada lista necesita al menos una variante main y una transfer")
    return values[:-1] if variant == "main" else values[-1:]


def _pick(rng: random.Random, options: list[tuple[str, float]]) -> str:
    values, weights = zip(*options, strict=True)
    return rng.choices(values, weights=weights, k=1)[0]


def impact_level(x: int, y: int, m: int) -> int:
    r = x / y
    if x == 0:
        return 0
    if x == y:
        return m - 1
    if m == 3:
        return 1
    if m == 4:
        return 1 if r < 0.5 else 2
    if m == 5:
        return 1 if r <= 0.25 else 2 if r <= 0.5 else 3
    raise ValueError(m)


@dataclass(frozen=True)
class Case:
    facts: dict[str, Any]
    lang: str


def sample_facts(rng: random.Random) -> dict[str, Any]:
    while True:
        f: dict[str, Any] = {k: _pick(rng, v) for k, v in FACTS.items()}
        if f["fault"] == "absent":
            f["fault_kind"] = None
        f["impact"] = None
        if f["fault"] != "absent" and rng.random() < 0.7:
            y = rng.randint(4, 200)
            kind = rng.choices(["zero", "all", "partial"], weights=[0.2, 0.2, 0.6])[0]
            x = 0 if kind == "zero" else y if kind == "all" else rng.randint(1, y - 1)
            f["impact"] = (x, y)
        if any(f[k] != "absent" for k in ("refund", "charge", "fault", "access")):
            return f


def render_state(
    rng: random.Random, lang: str, f: dict[str, Any], variant: Variant, version: Version = "v2"
) -> tuple[Any, str]:
    item, feature = rng.choice(ITEMS[lang]), rng.choice(FEATURES[lang])
    fmt_kw = {"item": item, "feature": feature, "Feature": feature[:1].upper() + feature[1:]}
    use_json = rng.random() < 0.3
    sentences = []
    for key in ("refund", "charge", "access"):
        if f[key] != "absent":
            sentences.append(rng.choice(_pool(S[lang][f"{key}.{f[key]}"]["v"], variant)).format(**fmt_kw))
    if f["fault"] != "absent":
        sentences.append(
            rng.choice(_pool(S[lang][f"fault.{f['fault']}.{f['fault_kind']}"]["v"], variant)).format(**fmt_kw)
        )
    if f["impact"] is not None and not use_json:
        x, y = f["impact"]
        sentences.append(rng.choice(_pool(S[lang]["impact"]["v"], variant)).format(x=x, y=y))
    rng.shuffle(sentences)
    opener = rng.choice(_pool(OPENERS[lang], variant))
    closer = rng.choice(_pool(CLOSERS[lang], variant))
    if version != "v1" and opener.endswith(",") and sentences:
        sentences[0] = sentences[0][:1].lower() + sentences[0][1:]
    message = " ".join(p for p in (opener, *sentences, closer) if p)
    if use_json:
        state: dict[str, Any] = {
            "channel": rng.choice(["email", "chat", "web_form"]),
            "customer_tier": rng.choice(["standard", "premium"]),
            "message": message,
        }
        if f["impact"] is not None:
            state["affected_users"], state["total_users"] = f["impact"]
        return state, f"{lang}-json-{variant}"
    return message, f"{lang}-text-{variant}"


def _opaque_ids(rng: random.Random, n: int) -> list[str]:
    # Lista en orden de generación: iterar un set de str depende de PYTHONHASHSEED y hacía
    # la asignación ID→opción no reproducible entre procesos (fallo P2-2).
    ids: list[str] = []
    while len(ids) < n:
        candidate = "o" + "".join(rng.choice("0123456789abcdef") for _ in range(6))
        if candidate not in ids:
            ids.append(candidate)
    return ids


def _choice_fault_kind(rng, lang, f, variant):
    true = "none" if f["fault"] == "absent" else f["fault_kind"]
    k = rng.randint(3, 6)
    kinds = ["network", "performance", "data", "application"]
    if true != "none" and rng.random() < 0.2:
        # Conjunto no exhaustivo: la categoría real no está y la respuesta correcta es "other".
        label = "other"
        pool = [c for c in [*kinds, "none"] if c != true]
        chosen = ["other", *rng.sample(pool, min(k, len(pool) + 1) - 1)]
    else:
        label = true
        pool = [c for c in [*kinds, "none", "other"] if c != true]
        chosen = [true, *rng.sample(pool, k - 1)]
    ids = _opaque_ids(rng, len(chosen))
    criteria = {i: rng.choice(_pool(FAULT_KIND_OPTIONS[lang][c], variant)) for i, c in zip(ids, chosen, strict=True)}
    label_id = ids[chosen.index(label)]
    instr = rng.choice(_pool(FAULT_KIND_INSTR[lang], variant))
    return {"type": "choice", "instructions": instr, "criteria": criteria}, {"class_id": label_id}, (
        "fault_type", "fault_kind_or_none_or_other"
    )


def fault_categories(lang: str, version: str = "v4") -> dict[str, list[str]]:
    """Categorías de fault_type en v4/v5: las del generador y las distractoras."""
    base = FAULT_KIND_OPTIONS_V5 if version == "v5" else FAULT_KIND_OPTIONS
    return {**base[lang], **FAULT_DISTRACTOR_OPTIONS[lang]}


def true_fault_category(f: dict[str, Any]) -> str:
    return "none" if f["fault"] == "absent" else f["fault_kind"]


def _choice_fault_kind_v4(rng, lang, f, variant, k_range: tuple[int, int] = FAULT_KIND_K):
    """Como v3, pero la etiqueta «other» no depende de K: se sortea antes y con la misma
    probabilidad para cualquier K, y siempre hay categorías no verdaderas suficientes."""
    true = true_fault_category(f)
    k = rng.randint(*k_range)
    wrong = [c for c in [*FAULT_KIND_OPTIONS[lang], *FAULT_DISTRACTOR_OPTIONS[lang]] if c not in (true, "other")]
    if true != "none" and rng.random() < OTHER_RATE:
        label, chosen = "other", ["other", *rng.sample(wrong, k - 1)]
    else:
        label = true
        chosen = [true, *rng.sample([*wrong, "other"], k - 1)]
    cats = fault_categories(lang)
    ids = _opaque_ids(rng, len(chosen))
    criteria = {i: rng.choice(_pool(cats[c], variant)) for i, c in zip(ids, chosen, strict=True)}
    label_id = ids[chosen.index(label)]
    instr = rng.choice(_pool(FAULT_KIND_INSTR[lang], variant))
    return {"type": "choice", "instructions": instr, "criteria": criteria}, {"class_id": label_id}, (
        "fault_type", "fault_kind_or_none_or_other_v4"
    )


def _choice_fault_kind_v5(rng, lang, f, variant, k_range: tuple[int, int] = FAULT_KIND_K):
    """Composición independiente de los hechos: ``none`` + ``other`` + K − 2 categorías sorteadas entre
    las cuatro reales y las tres distractoras, sin mirar el estado. La etiqueta se deduce después
    (categoría real si está listada; si no, ``other``; sin fallo, ``none``). Así, P(opciones | hechos)
    no depende de los hechos y las opciones sólo aportan la información semántica de la tarea."""
    k = rng.randint(*k_range)
    pool = [c for c in [*FAULT_KIND_OPTIONS_V5[lang], *FAULT_DISTRACTOR_OPTIONS[lang]] if c not in ("none", "other")]
    chosen = ["none", "other", *rng.sample(pool, k - 2)]
    rng.shuffle(chosen)
    true = true_fault_category(f)
    label = true if true in chosen else "other"  # sin fallo, true == "none", que siempre está
    cats = fault_categories(lang, "v5")
    ids = _opaque_ids(rng, len(chosen))
    criteria = {i: rng.choice(_pool(cats[c], variant)) for i, c in zip(ids, chosen, strict=True)}
    label_id = ids[chosen.index(label)]
    instr = f"{rng.choice(_pool(FAULT_KIND_INSTR[lang], variant))} {FAULT_TYPE_CLAUSE_V5[lang]}"
    return {"type": "choice", "instructions": instr, "criteria": criteria}, {"class_id": label_id}, (
        "fault_type", "fault_kind_v5_facts_independent_options"
    )


def _choice_team(rng, lang, f, variant):
    teams = ["billing", "accounts", "technical", "general"]
    names = {t: rng.choice(_pool(TEAM_NAMES[lang][t], variant)) for t in teams}
    # La política cambia de redacción con la variante; el orden de las cláusulas es la prioridad.
    instr = TEAM_POLICY[lang].format(**names)
    if variant == "transfer":
        instr = instr.replace("Asigna el equipo según esta política", "Elige el equipo aplicando esta norma").replace(
            "Assign the team with this policy", "Pick the team by applying this rule"
        )
    ids = _opaque_ids(rng, 4)
    criteria = {i: names[t] for i, t in zip(ids, teams, strict=True)}
    label_id = ids[teams.index(TEAM_RULE(f))]
    return {"type": "choice", "instructions": instr, "criteria": criteria}, {"class_id": label_id}, (
        "routing", "priority_policy(billing>accounts>technical>general)"
    )


def _score_impact(rng, lang, f, variant):
    m = rng.choice([3, 4, 5])
    rubric = _pool(IMPACT_RUBRICS[lang][m], variant)[0]
    x, y = f["impact"]
    instr = rng.choice(_pool(IMPACT_INSTR[lang], variant))
    return {"type": "score", "instructions": instr, "criteria": list(rubric)}, {
        "level_index": impact_level(x, y, m)
    }, ("service_impact", f"affected_ratio_m{m}")


def _score_severity(rng, lang, f, variant):
    rubric = _pool(SEVERITY_RUBRICS[lang], variant)[0]
    level = {"absent": 0, "resolved": 1, "active": 2}[f["fault"]]
    instr = rng.choice(_pool(SEVERITY_INSTR[lang], variant))
    return {"type": "score", "instructions": instr, "criteria": list(rubric)}, {"level_index": level}, (
        "fault_severity", "fault_state_ordinal"
    )


def _noul(kind):
    family, method, phrasings = NOUL_Q[kind]

    def build(rng, lang, f, variant):
        instr = rng.choice(_pool(phrasings[lang], variant))
        return {"type": "noul", "instructions": instr}, {"label": int(NOUL_RULES[kind](f))}, (family, method)

    return build


BUILDERS = {
    **{f"noul.{k}": _noul(k) for k in NOUL_Q},
    "choice.fault_kind": _choice_fault_kind,
    "choice.team": _choice_team,
    "score.impact": _score_impact,
    "score.severity": _score_severity,
}
TYPE_WEIGHTS = {"noul": 0.4, "choice": 0.3, "score": 0.3}


def generate_mixed(
    n_cases: int, seed: int, variant: Variant = "main", questions_per_case: int = 3, version: Version = "v3"
) -> tuple[list[Example], list[dict[str, Any]]]:
    if n_cases < 1 or not 1 <= questions_per_case <= len(BUILDERS):
        raise ValueError("cases debe ser positivo y questions_per_case estar entre 1 y el número de reglas")
    gen_version = GENERATOR_VERSIONS[version]
    rng = random.Random(f"{_RNG_STREAM[version]}:{variant}:{seed}")
    prefix = "mix" if variant == "main" else "mixT"
    examples, audit = [], []
    seen_states: set[str] = set()
    for case in range(n_cases):
        lang = "es" if case % 2 == 0 else "en"
        # Estados únicos: con un repertorio finito dos casos podrían coincidir literalmente,
        # y una entrada idéntica en grupos distintos sería una fuga entre particiones.
        for _ in range(1000):
            f = sample_facts(rng)
            state, template_id = render_state(rng, lang, f, variant, version)
            key = json.dumps(state, ensure_ascii=False, sort_keys=True)
            if key not in seen_states:
                seen_states.add(key)
                break
        else:
            raise RuntimeError("No se encontró un estado único; reduce n_cases o amplía el repertorio")
        group_id = f"{prefix}-s{seed}-{case:05d}"
        available = [k for k in BUILDERS if not (k == "score.impact" and f["impact"] is None)]
        if questions_per_case > len(available):
            raise ValueError("questions_per_case supera las preguntas disponibles para este caso")
        chosen: list[str] = []
        while len(chosen) < questions_per_case:
            t = rng.choices(list(TYPE_WEIGHTS), weights=list(TYPE_WEIGHTS.values()))[0]
            options = [k for k in available if k.startswith(t + ".") and k not in chosen]
            if options:
                chosen.append(rng.choice(options))
        audit.append({"group_id": group_id, "facts": f, "questions": chosen})
        for kind in chosen:
            builder = BUILDERS[kind]
            if kind == "choice.fault_kind" and version in ("v4", "v5"):
                builder = _choice_fault_kind_v4 if version == "v4" else _choice_fault_kind_v5
            question, target, (family, method) = builder(rng, lang, f, variant)
            if version != "v1" and question["type"] == "score" and rng.random() < 0.5:
                # Rúbrica descendente: mismo significado por nivel, índice invertido.
                question["criteria"] = list(reversed(question["criteria"]))
                target = {"level_index": len(question["criteria"]) - 1 - target["level_index"]}
                method = f"{method}_descending"
            if version in ("v3", "v4", "v5") and family in FAULT_FAMILIES:
                # No consume aleatoriedad: la secuencia de v2 se conserva.
                question["instructions"] = f"{question['instructions']} {ACCESS_POLICY_CLAUSE[lang]}"
            raw = {
                "schema_version": 1,
                "id": f"{prefix}-s{seed}-{case:05d}-{kind.replace('.', '-')}",
                "group_id": group_id,
                "task_family": family,
                "language": lang,
                "state": state,
                "image_path": None,
                "question": question,
                "target": target,
                "provenance": {
                    "source": "synthetic_rule",
                    "generator_version": gen_version,
                    "label_method": method,
                    "template_id": template_id,
                    "seed": seed,
                },
            }
            examples.append(Example.model_validate(raw))
    return examples, audit


def write_mixed_dataset(
    out_dir: Path,
    n_cases: int,
    seed: int,
    variant: Variant = "main",
    questions_per_case: int = 3,
    version: Version = "v3",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    target = out_dir / EXAMPLES_FILE
    examples, _ = generate_mixed(n_cases, seed, variant, questions_per_case, version)
    lines = [json.dumps(e.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for e in examples]
    data = ("\n".join(lines) + "\n").encode()
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"{target} existe con otro contenido; usa otro directorio")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = {
        "generator_version": GENERATOR_VERSIONS[version],
        "variant": variant,
        "seed": seed,
        "n_cases": n_cases,
        "questions_per_case": questions_per_case,
        "examples": len(examples),
        "examples_sha256": sha256_bytes(data),
        "created_utc": datetime.now(UTC).isoformat(),
        "note": "Sintético por reglas. 'transfer' usa sólo plantillas de superficie reservadas.",
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
