"""Date helpers. Sources report dates as 'YYYY', 'YYYY-MM', 'YYYY-MM-DD' or 'YYYYMMDD'."""

from __future__ import annotations

import calendar
from datetime import date


def parse_partial(value: str | None, end_of_period: bool = False) -> date | None:
    """Parse a possibly partial date. Month-only dates resolve to the first (or last) day of the month."""
    if not value:
        return None
    v = value.strip()
    try:
        if len(v) == 8 and v.isdigit():
            return date(int(v[:4]), int(v[4:6]), int(v[6:]))
        parts = [int(p) for p in v.split("-")]
    except ValueError:
        return None
    if len(parts) == 3:
        return date(*parts)
    if len(parts) == 2:
        y, m = parts
        return date(y, m, calendar.monthrange(y, m)[1] if end_of_period else 1)
    if len(parts) == 1:
        return date(parts[0], 12, 31) if end_of_period else date(parts[0], 1, 1)
    return None


def iso(d: date | None) -> str | None:
    return d.isoformat() if d else None
