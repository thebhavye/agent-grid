"""Optional grid settlement; examples use synthetic, currency-free tariffs."""

from dataclasses import dataclass, field

from simulation.baseline_allocator import _require_valid_kw


def _number(value, name):
    try:
        return _require_valid_kw(value, name)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc


@dataclass(frozen=True)
class GridConfig:
    """Off by default. Enabled tariffs cover [0, 1440) exactly once.

    Band end is exclusive. A start greater than end wraps midnight. Prices
    are non-negative per-kWh test units, not an asserted residential tariff.
    The validated private bands are immutable snapshots of caller input.
    """

    enabled: bool = False
    import_limit_kw: float = 0.0
    export_limit_kw: float = 0.0
    tariff: list[dict] = field(default_factory=list)
    _bands: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        for name in ("import_limit_kw", "export_limit_kw"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        bands = []
        try:
            for band in self.tariff:
                start = _number(band["start_min"], "start_min")
                end = _number(band["end_min"], "end_min")
                buy = _number(band["import_price_per_kwh"], "import_price_per_kwh")
                sell = _number(band["export_price_per_kwh"], "export_price_per_kwh")
                if start >= 1440 or end > 1440 or start == end:
                    raise ValueError("tariff times require 0 <= start < 1440, end <= 1440")
                if start < end:
                    bands.append((start, end, buy, sell))
                else:
                    bands.append((start, 1440.0, buy, sell))
                    if end:
                        bands.append((0.0, end, buy, sell))
        except (KeyError, TypeError) as exc:
            raise ValueError("tariff must contain complete band dictionaries") from exc
        bands.sort()
        if bands or self.enabled:
            cursor = 0.0
            for start, end, _, _ in bands:
                if start != cursor:
                    raise ValueError("tariff must cover the day without gaps or overlaps")
                cursor = end
            if cursor != 1440:
                raise ValueError("tariff must cover the full day")
        object.__setattr__(self, "_bands", tuple(bands))


def settle_grid(shortfall_kw, surplus_kw, minute_of_day, dt_min, config) -> dict:
    """Cap terminal import/export and price at the interval's starting band.

    There is no import-to-battery charging. Shortfall and surplus describe
    the already-allocated step. The runner must deliver the returned import
    to unmet loads; export is taken only from otherwise curtailed solar.
    """
    if not isinstance(config, GridConfig):
        raise ValueError("config must be GridConfig")
    shortfall = _number(shortfall_kw, "shortfall_kw")
    surplus = _number(surplus_kw, "surplus_kw")
    minute = _number(minute_of_day, "minute_of_day")
    dt = _number(dt_min, "dt_min")
    if minute >= 1440 or dt == 0:
        raise ValueError("minute_of_day must be < 1440 and dt_min must be positive")
    result = dict.fromkeys(("import_kw", "export_kw", "import_cost", "export_credit"), 0.0)
    if not config.enabled:
        return result
    if shortfall > 0 and surplus > 0:
        raise ValueError("cannot settle simultaneous shortfall and surplus")
    buy, sell = next((buy, sell) for start, end, buy, sell in config._bands
                     if start <= minute < end)
    result["import_kw"] = min(shortfall, config.import_limit_kw)
    result["export_kw"] = min(surplus, config.export_limit_kw)
    result["import_cost"] = _number(result["import_kw"] * dt / 60 * buy, "import_cost")
    result["export_credit"] = _number(result["export_kw"] * dt / 60 * sell, "export_credit")
    return result
