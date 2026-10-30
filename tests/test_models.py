from __future__ import annotations

import torch

from goalexit.models.goalexit import (
    CausalGazeGRU,
    GlobalStatsMLP,
    PrivilegedTeacher,
    make_deployment_model,
    parameter_counts,
)
from goalexit.train.engine import bernoulli_distillation_loss
from goalexit.train.fixed_refit import smooth_binary_targets


def test_authorized_model_shapes_and_student_capacity_identity() -> None:
    b0 = GlobalStatsMLP(22)
    b1 = make_deployment_model("b1", 10)
    d = make_deployment_model("d", 10)
    a = make_deployment_model("a", 10)
    assert tuple(b0(torch.zeros(3, 22)).shape) == (3,)
    lengths = torch.tensor([8, 6, 3])
    assert tuple(b1(torch.zeros(3, 8, 10), lengths).shape) == (3,)
    assert isinstance(b1, CausalGazeGRU)
    assert sum(p.numel() for p in b1.parameters()) == sum(p.numel() for p in d.parameters())
    assert sum(p.numel() for p in d.parameters()) == sum(p.numel() for p in a.parameters())
    assert not any("reader" in name or "identity" in name for name, _ in b1.named_parameters())
    assert b1.gru.bidirectional is False and b1.gru.num_layers == 2


def test_student_architecture_can_be_tuned_without_changing_interface() -> None:
    model = make_deployment_model(
        "b1",
        10,
        student_architecture={
            "gru_layers": 1,
            "hidden_size": 128,
            "head_size": 128,
            "bidirectional": False,
            "recurrent_dropout": 0.0,
        },
    )
    output = model(torch.randn(3, 7, 10), torch.tensor([7, 5, 4]))
    default = make_deployment_model("b1", 10)
    assert output.shape == (3,)
    assert parameter_counts(model)["trainable"] > parameter_counts(default)["trainable"]


def test_teacher_shapes_and_text_only_ignores_gaze_values() -> None:
    teacher = PrivilegedTeacher(maximum_words=20, text_only=False, dropout=0.0).eval()
    text_only = PrivilegedTeacher(maximum_words=20, text_only=True, dropout=0.0).eval()
    text = torch.randn(2, 12, 768)
    gaze = torch.randn(2, 12, 8)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    assert tuple(teacher(text, gaze, mask).shape) == (2,)
    with torch.no_grad():
        first = text_only(text, gaze, mask)
        second = text_only(text, gaze + 1000, mask)
    torch.testing.assert_close(first, second)


def test_gaze_only_teacher_ignores_text_values() -> None:
    gaze_only = PrivilegedTeacher(
        maximum_words=20, gaze_only=True, dropout=0.0
    ).eval()
    text = torch.randn(2, 12, 768)
    gaze = torch.randn(2, 12, 8)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    with torch.no_grad():
        first = gaze_only(text, gaze, mask)
        second = gaze_only(text + 1000, gaze, mask)
    torch.testing.assert_close(first, second)


def test_teacher_rejects_empty_dual_modality() -> None:
    try:
        PrivilegedTeacher(maximum_words=20, text_only=True, gaze_only=True)
    except ValueError:
        return
    raise AssertionError("Teacher accepted simultaneous text-only and gaze-only modes")


def test_distillation_reduces_to_hard_bce_at_zero_alpha() -> None:
    student = torch.tensor([-0.5, 0.7])
    teacher = torch.tensor([2.0, -1.0])
    target = torch.tensor([0.0, 1.0])
    observed = bernoulli_distillation_loss(
        student, teacher, target, alpha=0.0, temperature=2.0
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(student, target)
    torch.testing.assert_close(observed, expected)


def test_binary_label_smoothing_moves_targets_toward_half() -> None:
    target = torch.tensor([0.0, 1.0])
    torch.testing.assert_close(
        smooth_binary_targets(target, 0.1), torch.tensor([0.05, 0.95])
    )
