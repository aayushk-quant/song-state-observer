import numpy as np
from song_state_observer.features import extract_features
from song_state_observer.preprocess import standardise
from song_state_observer.models import fit, compute_log_B, viterbi_log



Z, _, _ = standardise(extract_features("data/raw/sleep.flac"))
result = fit(Z, K=3, seed=0)
log_B = compute_log_B(Z, result["means"], result["vars"])
with np.errstate(divide="ignore"):
    states, _ = viterbi_log(np.log(result["pi"]), np.log(result["A"]), log_B)


import matplotlib.pyplot as plt
import numpy as np

# states, Z already computed; rebuild the envelope for the backdrop
rms_db = Z[:, 0]                          # standardised log-RMS column
window_times = np.arange(len(states)) * 0.5   # window hop = 0.5s

fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(window_times, rms_db, color="black", linewidth=0.7, zorder=3)

# shade the background by decoded state
colors = ["#cfe8ff", "#ffe0b3", "#ffb3b3"]   # 3 states
for k in range(3):
    ax.fill_between(window_times, rms_db.min(), rms_db.max(),
                    where=(states == k), color=colors[k], alpha=0.6,
                    step="mid", label=f"state {k}")

ax.set_xlabel("time (s)")
ax.set_ylabel("standardised log-RMS")
ax.set_title("Sleep — decoded states over energy envelope")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()