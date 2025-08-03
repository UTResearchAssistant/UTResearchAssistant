"""Grant and conference monitor.

This module demonstrates how one might fetch information about
upcoming research funding opportunities and conferences.  The
implementation here is illustrative; production code would query
external APIs or scrape websites for up‑to‑date information.
"""

import logging
from datetime import date, timedelta
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class SystemMonitor:
    """System monitoring and status tracking for research services."""
    
    def __init__(self):
        """Initialize the system monitor."""
        self.services_status = {}
        logger.info("SystemMonitor initialized")
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health.
        
        Returns
        -------
        dict
            System health status
        """
        return {
            "status": "healthy",
            "timestamp": date.today().isoformat(),
            "services": self.services_status,
            "uptime": "Available"
        }
    
    def monitor_service(self, service_name: str, status: str) -> None:
        """Monitor a specific service status.
        
        Parameters
        ----------
        service_name : str
            Name of the service to monitor
        status : str
            Current status of the service
        """
        self.services_status[service_name] = {
            "status": status,
            "last_check": date.today().isoformat()
        }
        logger.info(f"Service {service_name} status: {status}")


def fetch_grants(topic: str) -> List[Tuple[str, date]]:
    """Retrieve a list of grants related to a topic.

    Parameters
    ----------
    topic : str
        Research topic to match grants against.

    Returns
    -------
    list[tuple[str, datetime.date]]
        A list of (grant name, deadline) pairs.
    """
    # TODO: integrate with funding agencies' APIs or scraping
    today = date.today()
    return [
        (f"{topic} Grant A", today + timedelta(days=30)),
        (f"{topic} Grant B", today + timedelta(days=60)),
    ]


def fetch_conferences(topic: str) -> List[Tuple[str, date]]:
    """Retrieve a list of conferences related to a topic.

    Parameters
    ----------
    topic : str
        Research topic to match conferences against.

    Returns
    -------
    list[tuple[str, datetime.date]]
        A list of (conference name, submission deadline) pairs.
    """
    today = date.today()
    return [
        (f"International Conference on {topic}", today + timedelta(days=45)),
        (f"{topic} Symposium", today + timedelta(days=90)),
    ]
