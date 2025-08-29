
from collections import Counter
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

def combo_key(tup: Tuple[str, ...]) -> str:
    return ", ".join(tup)

def frequency_table(combos: List[Tuple[str, ...]]) -> Dict[str, int]:
    return dict(Counter(combo_key(c) for c in combos))

def plot_pie_or_bar(freq: Dict[str, int]):
    n = len(freq)
    labels = list(freq.keys())
    counts = list(freq.values())
    if n <= 10:
        plt.figure()
        plt.pie(counts, labels=labels, autopct='%1.1f%%')
        plt.title('Target-word combo distribution')
        plt.show()
    else:
        total = sum(counts)
        props = [c/total for c in counts]
        plt.figure()
        plt.bar(["All combos"], [1.0])
        y = 0.0
        top = sorted(zip(labels, props), key=lambda x: -x[1])[:10]
        for lab, p in top:
            y_next = y + p
            yc = (y + y_next) / 2
            plt.text(0, yc, f"{lab}: {p:.1%}", ha='center', va='center', rotation=90)
            y = y_next
        plt.title('Proportions (top 10 combos annotated)')
        plt.ylim(0, 1)
        plt.show()
