
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios":[1.15,1]})

# Panel 1: two-arm evals, per-RUN pass rate (3 repeats/case)
evals = ["Options\n(5 cases)", "Attribution\n(6 cases)", "AAPL PIT\n(8 cases)"]
agent   = [15/15, 17/18, 24/24]
vanilla = [0/15,  0/18,  12/24]
x = np.arange(len(evals)); w = 0.36
b1 = ax1.bar(x-w/2, [v*100 for v in agent],   w, label="agent",   color="#1D9E75")
b2 = ax1.bar(x+w/2, [v*100 for v in vanilla], w, label="vanilla", color="0.45")
for b, v in zip(list(b1)+list(b2), agent+vanilla):
    ax1.text(b.get_x()+b.get_width()/2, v*100+1.5, f"{v*100:.0f}%", ha="center", fontsize=9)
ax1.set_ylim(0, 112); ax1.set_ylabel("per-run pass rate (%)  [3 repeats/case]")
ax1.set_xticks(x); ax1.set_xticklabels(evals)
ax1.set_title("Agent vs vanilla — scored evals (Claude backend)")
ax1.legend(frameon=False)

# Panel 2: golden set by area (agent arm; no vanilla run exists)
areas = ["pit", "provenance", "quant", "methodology"]
passed = [15, 11, 15, 8]; total = [15, 11, 15, 9]
y = np.arange(len(areas))
ax2.barh(y, [p/t*100 for p,t in zip(passed,total)], 0.55, color="#1D9E75")
for i,(p,t) in enumerate(zip(passed,total)):
    ax2.text(p/t*100-2, i, f"{p}/{t}", va="center", ha="right", fontsize=10, color="white")
ax2.set_yticks(y); ax2.set_yticklabels(areas); ax2.invert_yaxis()
ax2.set_xlim(0, 105); ax2.set_xlabel("pass rate (%)")
ax2.set_title("Golden set, 50 cases — agent arm\n49/50 overall, $1.45 total ($0.029/case)")
fig.tight_layout()
fig.savefig("agent_eval_comparison.png", dpi=140, bbox_inches="tight")
print("saved agent_eval_comparison.png")
