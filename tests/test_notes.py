"""Test topic_notes for all 5 pilot topics. Run: python -m tests.test_notes"""
import sys, os
sys.path.insert(0, '.')
os.chdir(r'c:\Users\dg264\Desktop\Python\curric-agents')
from src.notes import topic_notes, topic_order

print("=== topic_order ===")
order = topic_order("python_programming")
print(f"  Order: {order}")

print("\n=== topic_notes (each topic) ===")
for t in order:
    n = topic_notes("python_programming", t)
    if n is None:
        print(f"  NONE   {t}")
    elif n.startswith("> *Note:"):
        print(f"  FALLBACK {t} (heading not matched) — first 80: {n[:80]}")
    else:
        first = n.strip().split('\n')[0]
        words = len(n.split())
        print(f"  OK     {t}: {first[:60]}  ({words} words)")
