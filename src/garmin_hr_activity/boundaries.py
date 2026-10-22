"""Independent Python integer/datetime checks; no vectorized offset arithmetic."""

from datetime import datetime, timedelta, timezone

from garmin_inventory.privacy import AuditError


def expected_window(cutoff, step):
    if not float(cutoff).is_integer() or not 0 <= int(step) <= 1435:
        raise AuditError("Unexpected cutoff precision or window step.")
    cutoff_time = datetime.fromtimestamp(int(cutoff), tz=timezone.utc)
    end = cutoff_time - timedelta(minutes=int(step))
    start = end - timedelta(seconds=300)
    return start.timestamp(), end.timestamp()


def verify_grid(frame, cutoff_by_ema, lookbacks):
    """Check every row and every approved lookback it belongs to, independently."""
    previous = {}
    for row in frame.itertuples():
        cutoff = int(cutoff_by_ema[row.n_ema_id])
        step = int(row.n_step_index)
        a, b = expected_window(cutoff, step)
        if (
            row.n_window_start_epoch != a
            or row.n_window_end_epoch != b
            or b - a != 300
            or b > cutoff
        ):
            raise AuditError("Window differs from independent datetime boundaries.")
        if row.n_ema_id in previous:
            old_step, old_end = previous[row.n_ema_id]
            if step != old_step + 1 or old_end - b != 60:
                raise AuditError("Non-consecutive window grid.")
        elif step != 0:
            raise AuditError("Grid does not start at EMA cutoff.")
        previous[row.n_ema_id] = (step, b)
        for minutes in lookbacks:
            if step <= int(minutes) - 5 and not (cutoff - int(minutes) * 60 <= a < b <= cutoff):
                raise AuditError("Window escapes its approved lookback.")
    if any(step != 1435 for step, _ in previous.values()):
        raise AuditError("Incomplete 24-hour window grid.")
    return len(frame)
