"""Environment capture for reproducible run metadata.

Every experiment records the library/platform versions it ran under (see
CLAUDE.md); this helper is shared by the load-test harness and the eval runner
so both write the same shape into their ``meta.json``.
"""

import importlib.metadata
import platform


def collect_versions(packages: tuple[str, ...]) -> dict[str, str]:
    """Return python/platform info plus the installed version of each package."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions
