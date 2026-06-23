"""PortfolioManagerAgent — DISSOLVED.

The Portfolio Manager has been replaced by the deterministic ``size_position``
tool in ``firm.tools.size_position``.  This module is retained as an empty
stub so that any lingering imports of ``PortfolioManagerAgent`` fail loudly
(ImportError) rather than silently using stale logic.

Direction is now decided solely by the Research Manager (recommendation + conviction).
Sizing is computed by ``size_position(recommendation, conviction, nav, price, policy)``.
"""

from __future__ import annotations

# ``PortfolioManagerAgent`` no longer exists.  Import ``size_position`` from
# ``firm.tools.size_position`` for sizing logic.
