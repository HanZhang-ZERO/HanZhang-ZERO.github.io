#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
# sys.path.append('E:\VSCodeSpace\PythonWorkspace\Reliable-FLP')
sys.path.append('E:\VSCodeSpace\PythonWorkspace\djangoProject\mysite0203\cmdb\Algos')
from instanceGenerationRFLP import Instances
from instanceGenerationRRFLP import Instances


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite0203.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
