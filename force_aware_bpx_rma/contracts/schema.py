from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

class ContractError(ValueError):
    pass

@dataclass(frozen=True)
class Contract:
    estimator_frame_dim: int
    history_steps: int
    policy_dim: int
    history_dim: int
    action_dim: int
    sample_period_s: float
    policy_rate_hz: int
    pd_rate_hz: int
    force_frame: str
    force_scale_N: float

def _parse_scalar(value: str):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value.strip("'\"")

def load_contract(path: Optional[Union[str, Path]] = None) -> Contract:
    path = Path(path) if path else Path(__file__).with_name("force_policy_contract.yaml")
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _parse_scalar(value)
    required = {"estimator_frame_dim", "history_steps", "policy_dim", "history_dim", "action_dim", "sample_period_s", "policy_rate_hz", "pd_rate_hz", "force_frame", "force_scale_N"}
    missing = required - values.keys()
    if missing:
        raise ContractError(f"missing fields: {sorted(missing)}")
    if values["history_dim"] != values["history_steps"] * values["policy_dim"]:
        raise ContractError("history_dim must equal history_steps * policy_dim")
    if values["sample_period_s"] != 1 / values["policy_rate_hz"]:
        raise ContractError("sample_period_s must equal policy period")
    if values["force_frame"] != "body" or values["force_scale_N"] <= 0:
        raise ContractError("force frame/scale invalid")
    return Contract(**{k: values[k] for k in required})
