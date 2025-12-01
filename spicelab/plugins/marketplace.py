"""Plugin marketplace client.

This module provides a client for interacting with the SpiceLab
plugin marketplace, allowing users to search, install, and manage
plugins from a central repository.

Example::

    from spicelab.plugins import PluginMarketplace

    marketplace = PluginMarketplace()

    # Search for plugins
    results = marketplace.search("memristor")

    # Get plugin details
    info = marketplace.get_plugin_info("spicelab-memristor")

    # Install a plugin
    marketplace.install("spicelab-memristor")

    # Update all plugins
    marketplace.update_all()
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import PluginType

logger = logging.getLogger(__name__)

# Default marketplace URL (could be PyPI or custom server)
DEFAULT_MARKETPLACE_URL = "https://pypi.org/simple/"


@dataclass
class MarketplacePluginInfo:
    """Information about a plugin from the marketplace.

    Attributes:
        name: Package name
        version: Latest version
        description: Short description
        author: Author name
        license: License identifier
        url: Project URL
        downloads: Download count
        rating: Average rating (0-5)
        keywords: Search keywords
        plugin_type: Type of plugin
        spicelab_versions: Compatible SpiceLab versions
        dependencies: Required plugins
        created: Creation date
        updated: Last update date
    """

    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""
    url: str = ""
    downloads: int = 0
    rating: float = 0.0
    keywords: list[str] = field(default_factory=list)
    plugin_type: str = "GENERIC"
    spicelab_versions: str = ">=0.1.0"
    dependencies: list[str] = field(default_factory=list)
    created: str | None = None
    updated: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "url": self.url,
            "downloads": self.downloads,
            "rating": self.rating,
            "keywords": self.keywords,
            "plugin_type": self.plugin_type,
            "spicelab_versions": self.spicelab_versions,
            "dependencies": self.dependencies,
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketplacePluginInfo:
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", ""),
            url=data.get("url", ""),
            downloads=data.get("downloads", 0),
            rating=data.get("rating", 0.0),
            keywords=data.get("keywords", []),
            plugin_type=data.get("plugin_type", "GENERIC"),
            spicelab_versions=data.get("spicelab_versions", ">=0.1.0"),
            dependencies=data.get("dependencies", []),
            created=data.get("created"),
            updated=data.get("updated"),
        )


@dataclass
class InstallResult:
    """Result of a plugin installation.

    Attributes:
        success: Whether installation succeeded
        plugin_name: Name of the plugin
        version: Installed version
        message: Status message
        dependencies_installed: List of dependencies that were installed
    """

    success: bool
    plugin_name: str
    version: str = ""
    message: str = ""
    dependencies_installed: list[str] = field(default_factory=list)


class PluginMarketplace:
    """Client for the SpiceLab plugin marketplace.

    Provides functionality to search, browse, install, and manage
    plugins from a central repository.

    By default, plugins are installed via pip from PyPI with the
    naming convention 'spicelab-{plugin_name}'.

    Example::

        marketplace = PluginMarketplace()

        # Search for plugins
        results = marketplace.search("rf measurement")

        # Install a plugin
        result = marketplace.install("spicelab-rf-measurements")

        # List installed plugins
        installed = marketplace.list_installed()

        # Update a plugin
        marketplace.update("spicelab-rf-measurements")
    """

    def __init__(
        self,
        marketplace_url: str = DEFAULT_MARKETPLACE_URL,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the marketplace client.

        Args:
            marketplace_url: URL of the marketplace server
            cache_dir: Directory for caching plugin info
        """
        self.marketplace_url = marketplace_url
        self.cache_dir = cache_dir or Path.home() / ".spicelab" / "marketplace_cache"
        self._plugins_cache: dict[str, MarketplacePluginInfo] = {}
        self._cache_updated: datetime | None = None

    def search(
        self,
        query: str,
        plugin_type: PluginType | None = None,
        limit: int = 20,
    ) -> list[MarketplacePluginInfo]:
        """Search for plugins in the marketplace.

        Args:
            query: Search query string
            plugin_type: Optional type filter
            limit: Maximum number of results

        Returns:
            List of matching plugins

        Note:
            This is a simplified implementation. A real marketplace
            would query a server API.
        """
        # For now, search PyPI for packages starting with 'spicelab-'
        results = []

        try:
            # Use pip search alternative (pip search is deprecated)
            # This queries pypi.org JSON API
            # Note: PyPI doesn't have a public search API, so we use a mock
            # In a real implementation, you'd have your own marketplace server

            # Return mock results for demonstration
            logger.info(f"Searching marketplace for: {query}")

            # Mock results (in production, query actual marketplace)
            mock_plugins = self._get_mock_plugins()
            query_lower = query.lower()

            for plugin in mock_plugins:
                if (
                    query_lower in plugin.name.lower()
                    or query_lower in plugin.description.lower()
                    or any(query_lower in kw.lower() for kw in plugin.keywords)
                ):
                    if plugin_type is None or plugin.plugin_type == plugin_type.name:
                        results.append(plugin)

                if len(results) >= limit:
                    break

        except Exception as e:
            logger.error(f"Search failed: {e}")

        return results

    def get_plugin_info(self, name: str) -> MarketplacePluginInfo | None:
        """Get detailed information about a plugin.

        Args:
            name: Plugin package name

        Returns:
            Plugin information, or None if not found
        """
        # Check cache first
        if name in self._plugins_cache:
            return self._plugins_cache[name]

        try:
            # Query PyPI JSON API
            import urllib.error
            import urllib.request

            url = f"https://pypi.org/pypi/{name}/json"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())

            info = data.get("info", {})
            plugin_info = MarketplacePluginInfo(
                name=info.get("name", name),
                version=info.get("version", ""),
                description=info.get("summary", ""),
                author=info.get("author", ""),
                license=info.get("license", ""),
                url=info.get("home_page", ""),
                keywords=info.get("keywords", "").split(",") if info.get("keywords") else [],
            )

            self._plugins_cache[name] = plugin_info
            return plugin_info

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"Plugin {name} not found on PyPI")
            else:
                logger.error(f"HTTP error getting plugin info: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get plugin info: {e}")
            return None

    def install(
        self,
        name: str,
        version: str | None = None,
        upgrade: bool = False,
    ) -> InstallResult:
        """Install a plugin from the marketplace.

        Args:
            name: Plugin package name
            version: Specific version to install (default: latest)
            upgrade: Upgrade if already installed

        Returns:
            Installation result
        """
        package_spec = name
        if version:
            package_spec = f"{name}=={version}"

        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package_spec)

        logger.info(f"Installing plugin: {package_spec}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                # Get installed version
                installed_version = self._get_installed_version(name)
                return InstallResult(
                    success=True,
                    plugin_name=name,
                    version=installed_version or version or "latest",
                    message="Installation successful",
                )
            else:
                return InstallResult(
                    success=False,
                    plugin_name=name,
                    message=f"Installation failed: {result.stderr}",
                )

        except subprocess.TimeoutExpired:
            return InstallResult(
                success=False,
                plugin_name=name,
                message="Installation timed out",
            )
        except Exception as e:
            return InstallResult(
                success=False,
                plugin_name=name,
                message=f"Installation error: {e}",
            )

    def uninstall(self, name: str) -> InstallResult:
        """Uninstall a plugin.

        Args:
            name: Plugin package name

        Returns:
            Uninstallation result
        """
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", name]

        logger.info(f"Uninstalling plugin: {name}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return InstallResult(
                    success=True,
                    plugin_name=name,
                    message="Uninstallation successful",
                )
            else:
                return InstallResult(
                    success=False,
                    plugin_name=name,
                    message=f"Uninstallation failed: {result.stderr}",
                )

        except Exception as e:
            return InstallResult(
                success=False,
                plugin_name=name,
                message=f"Uninstallation error: {e}",
            )

    def update(self, name: str) -> InstallResult:
        """Update a plugin to the latest version.

        Args:
            name: Plugin package name

        Returns:
            Update result
        """
        return self.install(name, upgrade=True)

    def update_all(self) -> list[InstallResult]:
        """Update all installed SpiceLab plugins.

        Returns:
            List of update results
        """
        installed = self.list_installed()
        results = []

        for name, _ in installed:
            result = self.update(name)
            results.append(result)

        return results

    def list_installed(self) -> list[tuple[str, str]]:
        """List all installed SpiceLab plugins.

        Returns:
            List of (name, version) tuples
        """
        installed = []

        try:
            # Get list of installed packages
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                packages = json.loads(result.stdout)
                for pkg in packages:
                    name = pkg.get("name", "")
                    # Look for spicelab plugins
                    if name.startswith("spicelab-") or name.startswith("spicelab_"):
                        installed.append((name, pkg.get("version", "")))

        except Exception as e:
            logger.error(f"Failed to list installed packages: {e}")

        return installed

    def check_updates(self) -> list[tuple[str, str, str]]:
        """Check for available updates.

        Returns:
            List of (name, current_version, latest_version) tuples
        """
        updates = []
        installed = self.list_installed()

        for name, current_version in installed:
            info = self.get_plugin_info(name)
            if info and info.version != current_version:
                updates.append((name, current_version, info.version))

        return updates

    def _get_installed_version(self, name: str) -> str | None:
        """Get the installed version of a package.

        Args:
            name: Package name

        Returns:
            Version string, or None if not installed
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", name],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()

        except Exception:
            pass

        return None

    def _get_mock_plugins(self) -> list[MarketplacePluginInfo]:
        """Get mock plugins for demonstration.

        Returns:
            List of mock plugin info objects
        """
        return [
            MarketplacePluginInfo(
                name="spicelab-memristor",
                version="1.0.0",
                description="Memristor component models for SpiceLab",
                author="SpiceLab Community",
                keywords=["memristor", "component", "memory"],
                plugin_type="COMPONENT",
            ),
            MarketplacePluginInfo(
                name="spicelab-rf-measurements",
                version="2.1.0",
                description="RF and microwave measurements (S-parameters, NF, IP3)",
                author="SpiceLab Community",
                keywords=["rf", "microwave", "s-parameters", "measurement"],
                plugin_type="MEASUREMENT",
            ),
            MarketplacePluginInfo(
                name="spicelab-qspice",
                version="0.5.0",
                description="QSPICE simulation engine integration",
                author="SpiceLab Community",
                keywords=["qspice", "engine", "simulator"],
                plugin_type="ENGINE",
            ),
            MarketplacePluginInfo(
                name="spicelab-smith-chart",
                version="1.2.0",
                description="Smith chart visualization for impedance matching",
                author="SpiceLab Community",
                keywords=["smith", "chart", "impedance", "visualization"],
                plugin_type="VISUALIZATION",
            ),
            MarketplacePluginInfo(
                name="spicelab-kicad-import",
                version="0.8.0",
                description="Import KiCad schematics and footprints",
                author="SpiceLab Community",
                keywords=["kicad", "import", "schematic"],
                plugin_type="IMPORT",
            ),
            MarketplacePluginInfo(
                name="spicelab-harmonic-balance",
                version="1.0.0",
                description="Harmonic balance analysis for RF circuits",
                author="SpiceLab Community",
                keywords=["harmonic", "balance", "rf", "analysis"],
                plugin_type="ANALYSIS",
            ),
        ]

    def get_info(self) -> dict[str, Any]:
        """Get marketplace client information.

        Returns:
            Dictionary with client state
        """
        return {
            "marketplace_url": self.marketplace_url,
            "cache_dir": str(self.cache_dir),
            "cached_plugins": list(self._plugins_cache.keys()),
            "installed": self.list_installed(),
        }


# Convenience functions


def search_plugins(query: str) -> list[MarketplacePluginInfo]:
    """Search for plugins in the marketplace.

    Args:
        query: Search query

    Returns:
        List of matching plugins
    """
    marketplace = PluginMarketplace()
    return marketplace.search(query)


def install_plugin(name: str, version: str | None = None) -> InstallResult:
    """Install a plugin from the marketplace.

    Args:
        name: Plugin package name
        version: Specific version (optional)

    Returns:
        Installation result
    """
    marketplace = PluginMarketplace()
    return marketplace.install(name, version)


def list_installed_plugins() -> list[tuple[str, str]]:
    """List installed SpiceLab plugins.

    Returns:
        List of (name, version) tuples
    """
    marketplace = PluginMarketplace()
    return marketplace.list_installed()
