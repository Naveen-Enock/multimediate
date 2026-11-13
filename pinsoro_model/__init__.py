"""PInSoRo simple-classifier baseline: window statistics -> XGBoost.

A non-deep control for the categorical PInSoRo task. Fixed-length statistics are computed over
overlapping windows of the raw SSI streams and cached as one ``.npz`` per (session, role) *inside
the session directory*; gradient-boosted trees are then fitted on those tables.

See ``README.md`` for the cache schema and the CLI.
"""
