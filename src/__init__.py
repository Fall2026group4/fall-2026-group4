"""SAE-Faithful source package.

Library modules log progress through ``logging.getLogger(__name__)`` instead of
``print``. This package-level handler keeps that progress visible when the
scripts in ``scripts/`` are run from a terminal, while code that imports the
package (for example the tests, or a notebook that configures its own logging)
can still change the level or add handlers on the ``"src"`` logger.
"""

import logging

_logger = logging.getLogger("src")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
