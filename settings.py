"""Dataset field declarations and the paper's training hyperparameters."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field


CAUSAL_EVENT_TIME_FIELDS = (
    "seconds_since_last",
    "seconds_since_start",
    "timesincelastevent",
    "timesincecasestart",
)
_TEMPLATE_FIELD = re.compile(r"{{\s*([^{}]+?)\s*}}")
_PROHIBITED = {
    "case",
    "caseid",
    "caseidentifier",
    "caseconceptname",
    "traceid",
    "sourceorder",
    "caselength",
    "tracelength",
    "totalcaselength",
    "progress",
    "caseprogress",
    "remainingtime",
    "remainingtimedays",
    "remainingtimez",
    "remainingdays",
    "computedremainingdays",
    "nextactivity",
    "nexttime",
    "nexttimedays",
    "nexttimestamp",
    "caseend",
    "caseendtime",
    "caseendtimestamp",
    "endtime",
    "endtimestamp",
    "target",
    "label",
}


def template_aliases(name: str) -> set[str]:
    return {
        name,
        name.replace(":", "_").replace(" ", "_"),
        re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_"),
    }


def _prohibited(name: str) -> bool:
    key = re.sub(r"[^a-z0-9]", "", name.casefold())
    return (
        name.startswith("_")
        or key in _PROHIBITED
        or any(
            key.startswith(prefix) and key[len(prefix) :] in _PROHIBITED
            for prefix in ("case", "trace", "event")
        )
    )


@dataclass
class DatasetSpec:
    """Declare inputs available at prediction time; targets are computed separately.

    ``column_mapping`` maps source names to canonical names. Known target and
    case identity fields are prohibited as model inputs. For custom fields,
    their availability at prediction time remains a dataset author decision.
    """

    name: str
    event_categorical_columns: tuple[str, ...] = ("activity", "resource")
    event_text_columns: tuple[str, ...] = ()
    static_columns: tuple[str, ...] = ()
    event_template: str | None = None
    trace_template: str | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    expected_statistics: dict[str, int] | None = None
    csv_encoding: str = "utf-8-sig"

    def __post_init__(self) -> None:
        for name in (
            "event_categorical_columns",
            "event_text_columns",
            "static_columns",
        ):
            setattr(self, name, tuple(getattr(self, name)))
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Dataset name must not be empty")
        for group in (
            self.event_categorical_columns,
            self.event_text_columns,
            self.static_columns,
        ):
            if len(set(group)) != len(group) or any(
                not isinstance(c, str) or not c.strip() for c in group
            ):
                raise ValueError("Input column names must be nonempty and unique")
        if "activity" not in self.event_categorical_columns:
            raise ValueError("event_categorical_columns must include activity")
        if not isinstance(self.column_mapping, dict) or any(
            not isinstance(k, str)
            or not k.strip()
            or not isinstance(v, str)
            or not v.strip()
            for k, v in self.column_mapping.items()
        ):
            raise ValueError(
                "column_mapping must map nonempty source names to canonical names"
            )
        if len(set(self.column_mapping.values())) != len(self.column_mapping):
            raise ValueError("column_mapping has conflicting destination columns")
        if any(
            _prohibited(old)
            for old, new in self.column_mapping.items()
            if new == "timestamp"
        ):
            raise ValueError(
                "Timestamp cannot be derived from an identity or future-target field"
            )
        columns = set(
            self.event_categorical_columns
            + self.event_text_columns
            + self.static_columns
        )
        for column in columns:
            source_names = [column] + [
                k for k, v in self.column_mapping.items() if v == column
            ]
            if any(_prohibited(name) for name in source_names):
                raise ValueError("Prohibited identity or future input: " + column)
        for template, available in (
            (
                self.event_template,
                self.event_categorical_columns
                + self.event_text_columns
                + ("timestamp",)
                + CAUSAL_EVENT_TIME_FIELDS,
            ),
            (self.trace_template, self.static_columns),
        ):
            if template is None:
                continue
            if not isinstance(template, str):
                raise ValueError("Templates must be strings")
            allowed = {
                alias for column in available for alias in template_aliases(column)
            }
            unknown = {
                m.group(1).strip() for m in _TEMPLATE_FIELD.finditer(template)
            } - allowed
            if unknown:
                raise ValueError(
                    "Template contains undeclared or future input: "
                    + ", ".join(sorted(unknown))
                )
            if "{{" in _TEMPLATE_FIELD.sub("", template) or "}}" in _TEMPLATE_FIELD.sub(
                "", template
            ):
                raise ValueError("Malformed template placeholder")
        if self.expected_statistics is not None and (
            set(self.expected_statistics) != {"cases", "events", "activities"}
            or any(
                type(v) is not int or v < 1 for v in self.expected_statistics.values()
            )
        ):
            raise ValueError(
                "expected_statistics requires positive cases/events/activities"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_epochs: int = 50
    early_stopping_patience: int = 8
    bert_learning_rate: float = 1e-5
    head_learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    smooth_l1_beta: float = 0.5
    classification_loss_weight: float = 1.0
    regression_loss_weight: float = 1.0
    use_amp: bool = True

    def validate(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_epochs",
            "early_stopping_patience",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("bert_learning_rate", "head_learning_rate", "smooth_l1_beta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in (
            "weight_decay",
            "gradient_clip_norm",
            "classification_loss_weight",
            "regression_loss_weight",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        for name in ("warmup_ratio", "label_smoothing"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(name + " must be in [0, 1)")
        if self.classification_loss_weight + self.regression_loss_weight <= 0:
            raise ValueError("At least one task loss weight must be positive")
        if type(self.use_amp) is not bool:
            raise ValueError("use_amp must be a boolean")

    def to_dict(self) -> dict:
        return asdict(self)


# Field declarations from the eight registered paper datasets.
DATASETS = {
    "BPIC_2013_C": DatasetSpec(
        **{
            "name": "BPIC_2013_C",
            "event_categorical_columns": [
                "activity",
                "resource",
                "lifecycletransition",
            ],
            "event_text_columns": ["resourcecountry"],
            "static_columns": ["product", "organizationcountry", "impact"],
            "event_template": "{{resource}} performed {{activity}} with lifecycle {{lifecycletransition}} in "
            "{{resourcecountry}}.",
            "trace_template": "The case concerns product {{product}}, organization country "
            "{{organizationcountry}}, and impact {{impact}}.",
            "column_mapping": {},
            "expected_statistics": {"cases": 1487, "events": 6660, "activities": 4},
            "csv_encoding": "utf-8-sig",
        }
    ),
    "BPIC_2017_O": DatasetSpec(
        **{
            "name": "BPIC_2017_O",
            "event_categorical_columns": ["activity", "resource", "action"],
            "static_columns": [
                "MonthlyCost",
                "CreditScore",
                "FirstWithdrawalAmount",
                "OfferedAmount",
                "NumberOfTerms",
            ],
            "event_template": "{{resource}} performed {{activity}} with status {{action}}.",
            "trace_template": "Monthly cost {{MonthlyCost}}, credit score {{CreditScore}}, first withdrawal "
            "{{FirstWithdrawalAmount}}, offered amount {{OfferedAmount}}, and "
            "{{NumberOfTerms}} terms.",
            "column_mapping": {},
            "expected_statistics": {"cases": 42995, "events": 193849, "activities": 8},
            "csv_encoding": "utf-8-sig",
        }
    ),
    "BPIC_2020_Pr": DatasetSpec(
        **{
            "name": "BPIC_2020_Pr",
            "event_categorical_columns": ["activity", "resource", "role"],
            "static_columns": ["case:Task", "case:Permit BudgetNumber", "case:Project"],
            "event_template": "{{resource}} with role {{role}} performed {{activity}}.",
            "trace_template": "Project {{case_Project}}, task {{case_Task}}, permit budget "
            "{{case_Permit_BudgetNumber}}.",
            "column_mapping": {},
            "expected_statistics": {"cases": 2099, "events": 18246, "activities": 29},
            "csv_encoding": "utf-8-sig",
        }
    ),
    "BPIC_2020_Re": DatasetSpec(
        **{
            "name": "BPIC_2020_Re",
            "event_categorical_columns": ["activity", "resource", "role"],
            "static_columns": [
                "case:Project",
                "case:RequestedAmount",
                "case:OrganizationalEntity",
            ],
            "event_template": "{{resource}} with role {{role}} performed {{activity}}.",
            "trace_template": "Project {{case_Project}}, amount {{case_RequestedAmount}}, organization "
            "{{case_OrganizationalEntity}}.",
            "column_mapping": {
                "case:concept:name": "case",
                "concept:name": "activity",
                "time:timestamp": "timestamp",
                "org:resource": "resource",
                "org:role": "role",
            },
            "expected_statistics": {"cases": 6886, "events": 36796, "activities": 19},
            "csv_encoding": "utf-8-sig",
        }
    ),
    "Helpdesk": DatasetSpec(
        **{
            "name": "Helpdesk",
            "event_categorical_columns": [
                "activity",
                "resource",
                "servicelevel",
                "servicetype",
                "workgroup",
                "product",
                "customer",
            ],
            "static_columns": ["supportsection", "responsiblesection"],
            "event_template": "{{resource}} performed {{activity}}. Workgroup {{workgroup}} handled product "
            "{{product}} for customer {{customer}} with service {{servicetype}} at level "
            "{{servicelevel}}.",
            "trace_template": "Support section {{supportsection}} is led by {{responsiblesection}}.",
            "column_mapping": {
                "concept:name": "activity",
                "time:timestamp": "timestamp",
            },
            "expected_statistics": {"cases": 4580, "events": 21348, "activities": 14},
            "csv_encoding": "utf-8-sig",
        }
    ),
    "MIP": DatasetSpec(
        **{
            "name": "MIP",
            "event_categorical_columns": ["activity", "resource"],
            "static_columns": [],
            "event_template": "{{resource}} performed {{activity}}.",
            "trace_template": "",
            "column_mapping": {},
            "expected_statistics": {"cases": 1000, "events": 49604, "activities": 36},
            "csv_encoding": "utf-8-sig",
        }
    ),
    "Production": DatasetSpec(
        **{
            "name": "Production",
            "event_categorical_columns": [
                "activity",
                "resource",
                "ReportType",
                "PartDesc",
            ],
            "static_columns": [],
            "event_template": "{{resource}} performed {{activity}} on {{PartDesc}} with report "
            "{{ReportType}}.",
            "trace_template": "",
            "column_mapping": {
                "case:concept:name": "case",
                "concept:name": "activity",
                "Complete Timestamp": "timestamp",
                "Resource": "resource",
                "Report Type": "ReportType",
                "Part Desc.": "PartDesc",
                "lifecycle:transition": "lifecycletransition",
            },
            "expected_statistics": {"cases": 225, "events": 4543, "activities": 55},
            "csv_encoding": "utf-8-sig",
        }
    ),
    "Receipt": DatasetSpec(
        **{
            "name": "Receipt",
            "event_categorical_columns": ["activity", "resource", "group"],
            "static_columns": ["channel", "department", "responsible"],
            "event_template": "{{resource}} performed {{activity}} within group {{group}}.",
            "trace_template": "The request arrived through {{channel}}, is handled by {{department}}, and has "
            "responsible party {{responsible}}.",
            "column_mapping": {},
            "expected_statistics": {"cases": 1434, "events": 8577, "activities": 27},
            "csv_encoding": "utf-8-sig",
        }
    ),
    "custom": DatasetSpec(name="custom"),
}
