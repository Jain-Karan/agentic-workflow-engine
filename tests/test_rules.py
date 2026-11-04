from claimflow.domain import Action, Category
from claimflow.rules import RuleConfig, decide
from claimflow.synthetic import _policy_dict
from tests.conftest import make_policy


def p(**kw):
    return _policy_dict(make_policy(**kw))


def test_low_confidence_goes_to_a_human():
    d = decide(Category.AUTO_GLASS, 0.4, 500, p())
    assert d.action == Action.MANUAL_TRIAGE and d.queue == "intake-manual"


def test_service_requests_are_not_claims():
    assert decide(Category.BILLING, 0.9, 120, None).action == Action.SERVICE_REPLY


def test_missing_policy_needs_triage():
    assert decide(Category.THEFT, 0.9, 500, None).action == Action.MANUAL_TRIAGE


def test_lapsed_policy_denial_needs_approval():
    d = decide(Category.THEFT, 0.9, 500, p(status="lapsed"))
    assert d.action == Action.DENY_NO_COVERAGE and d.needs_approval


def test_missing_coverage_is_denied_with_approval():
    d = decide(Category.AUTO_GLASS, 0.9, 500, p(dwelling={"limit": 1.0, "deductible": 0.0}))
    assert d.action == Action.DENY_NO_COVERAGE and d.needs_approval


def test_fraud_and_frequency_go_to_siu():
    assert decide(Category.THEFT, 0.9, 500, p(fraud=True)).action == Action.REFER_SIU
    assert decide(Category.THEFT, 0.9, 500, p(prior=3)).action == Action.REFER_SIU


def test_fast_track_payout_net_of_deductible():
    d = decide(Category.AUTO_GLASS, 0.9, 640, p())
    assert d.action == Action.AUTO_SETTLE and d.payout == 540.0 and not d.needs_approval


def test_payout_above_threshold_needs_approval():
    d = decide(Category.PROPERTY_WATER, 0.9, 2300, p())
    assert d.action == Action.AUTO_SETTLE and d.payout == 1300.0 and d.needs_approval


def test_large_losses_go_to_adjuster():
    d = decide(Category.PROPERTY_WATER, 0.9, 12000, p())
    assert d.action == Action.ASSIGN_ADJUSTER and d.queue == "adjuster-property"


def test_collision_is_never_fast_tracked():
    assert decide(Category.AUTO_COLLISION, 0.9, 900, p()).action == Action.ASSIGN_ADJUSTER


def test_custom_thresholds():
    cfg = RuleConfig(approval_over=5000)
    assert not decide(Category.PROPERTY_WATER, 0.9, 2300, p(), cfg).needs_approval
