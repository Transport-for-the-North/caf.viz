# -*- coding: utf-8 -*-
"""Miscellaneous utility functions for visualisation."""

##### IMPORTS #####

# Built-Ins
import logging
import re

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


def normalise_name(name: str) -> str:
    """Convert name to lowercase and replace spaces with underscore."""
    name = name.lower().strip()
    return re.sub(r"\s+", "_", name)
