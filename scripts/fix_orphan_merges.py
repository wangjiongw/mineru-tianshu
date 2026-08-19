#!/usr/bin/env python3
"""Compatibility shim for the old orphan merge repair command."""
import sys


def main():
    print("fix_orphan_merges.py is retired. Use scripts/reconcile_parent_merges.py instead.")
    print("Default behavior is dry-run; add --apply to repair eligible parent merges.")
    print("Example: python scripts/reconcile_parent_merges.py --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
