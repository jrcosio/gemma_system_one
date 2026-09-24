import pytest
from pydantic import ValidationError

from conftest import make_example
from gemma_system_one.contracts import (
    ChoiceQuestion,
    Example,
    NoulTarget,
    ScoreQuestion,
    parse_json_strict,
)

SPEC_EXAMPLES = [
    # Los tres ejemplos literales de la especificación §5.1.
    '{"schema_version":1,"id":"n-001","group_id":"case-001","task_family":"refund","language":"es","state":"Solicito que me devuelvan el cobro duplicado.","image_path":null,"question":{"type":"noul","instructions":"¿El cliente solicita una devolución?"},"target":{"label":1},"provenance":{"source":"synthetic_rule","generator_version":"v1","label_method":"explicit_request"}}',  # noqa: E501
    '{"schema_version":1,"id":"c-001","group_id":"case-002","task_family":"ticket_routing","language":"es","state":"No resuelve el DNS del servicio.","image_path":null,"question":{"type":"choice","instructions":"Selecciona el subsistema que presenta el fallo descrito.","criteria":{"network":"Resolución DNS o conectividad","auth":"Credenciales o permisos","other":"Ninguna de las anteriores"}},"target":{"class_id":"network"},"provenance":{"source":"synthetic_rule","generator_version":"v1","label_method":"explicit_fault"}}',  # noqa: E501
    '{"schema_version":1,"id":"s-001","group_id":"case-003","task_family":"service_impact","language":"es","state":{"affected_users":12,"total_users":12},"image_path":null,"question":{"type":"score","instructions":"Evalúa alcance de la incidencia según usuarios afectados.","criteria":["Ningún usuario","Algunos usuarios, pero no todos","Todos los usuarios"]},"target":{"level_index":2},"provenance":{"source":"synthetic_rule","generator_version":"v1","label_method":"affected_ratio"}}',  # noqa: E501
]


@pytest.mark.parametrize("line", SPEC_EXAMPLES)
def test_spec_examples_validate(line):
    Example.model_validate(parse_json_strict(line))


def _choice(n: int, **kw):
    return {
        "type": "choice",
        "instructions": "x",
        "criteria": {f"o{i}": f"opción {i}" for i in range(n)},
        **kw,
    }


@pytest.mark.parametrize("n,ok", [(1, False), (2, True), (8, True), (9, False)])
def test_choice_cardinality(n, ok):
    if ok:
        ChoiceQuestion.model_validate(_choice(n))
    else:
        with pytest.raises(ValidationError):
            ChoiceQuestion.model_validate(_choice(n))


@pytest.mark.parametrize("m,ok", [(1, False), (2, True), (5, True), (6, False)])
def test_score_cardinality(m, ok):
    q = {"type": "score", "instructions": "x", "criteria": [f"nivel {i}" for i in range(m)]}
    if ok:
        ScoreQuestion.model_validate(q)
    else:
        with pytest.raises(ValidationError):
            ScoreQuestion.model_validate(q)


def test_choice_rejects_duplicate_descriptions_after_normalization():
    with pytest.raises(ValidationError, match="duplicadas"):
        ChoiceQuestion.model_validate(
            {"type": "choice", "instructions": "x", "criteria": {"a": "Red  DNS", "b": "red dns"}}
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"question": {"type": "noul", "instructions": "   "}},
        {"question": {"type": "noul", "instructions": "x", "extra": 1}},
        {"question": {"type": "binary", "instructions": "x"}},
        {"target": {"class_id": "a"}},
        {"target": {"label": 2}},
        {"target": {"label": True}},
        {"target": {"label": 0.9}},  # etiquetas blandas no usan el campo duro
        {"state": ""},
        {"state": {}},
        {"state": []},
        {"state": {"x": float("nan")}},
        {"state": {"x": float("inf")}},
        {"state": {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": 1}}}}}}}}}}},
        {"schema_version": 2},
        {"language": "fr"},
        {"id": "con espacio"},
        {"unexpected": "field"},
        {"image_path": "/etc/passwd"},
        {"image_path": "../fuera.png"},
        {"image_path": "img/../../fuera.png"},
        {"image_path": "C:\\img.png"},
    ],
)
def test_invalid_examples_rejected(overrides):
    with pytest.raises(ValidationError):
        make_example(**overrides)


def test_target_must_match_question_criteria():
    choice = _choice(3)
    with pytest.raises(ValidationError, match="class_id"):
        make_example(question=choice, target={"class_id": "zz"})
    make_example(question=choice, target={"class_id": "o1"})
    score = {"type": "score", "instructions": "x", "criteria": ["a", "b"]}
    with pytest.raises(ValidationError, match="fuera de rango"):
        make_example(question=score, target={"level_index": 2})
    with pytest.raises(ValidationError):
        make_example(question=score, target={"label": 1})


def test_json_state_and_relative_image_path_accepted():
    e = make_example(state={"affected": 3, "total": 10, "tags": ["a"]}, image_path="imgs/chart_01.png")
    assert e.image_path == "imgs/chart_01.png"
    assert isinstance(e.target, NoulTarget)


def test_strict_json_parser_rejects_nan():
    with pytest.raises(ValueError):
        parse_json_strict('{"state": NaN}')
    with pytest.raises(ValueError):
        parse_json_strict('{"state": Infinity}')
