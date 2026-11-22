"""Plot a saved case report using actual hourly bin starts and masks."""
import argparse
from pathlib import Path
import numpy as np
from protocol import read_json

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = read_json(a.report)
    mask = np.asarray(report['valid_time_mask'], dtype=bool)
    attention = np.asarray(report['temporal_attention'])
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(np.arange(len(mask))[mask], attention[mask], width=1.0, align='edge')
    from generate_report import estimate_temporal_peak_time, estimate_attention_centroid_time
    # Recompute from the attention arrays so legacy reports that mislabeled
    # a centroid as peak_time_hours cannot silently retain that plot label.
    peak_time = estimate_temporal_peak_time(attention, mask)
    centroid_time = estimate_attention_centroid_time(attention, mask)
    if peak_time is not None:
        ax.axvline(peak_time, color='tab:red', linestyle='--', linewidth=1.2,
                   label=f"peak bin start = {peak_time:g} h")
    if centroid_time is not None:
        ax.axvline(centroid_time, color='tab:gray', linestyle=':', linewidth=1.2,
                   label=f"attention centroid = {centroid_time:.2f} h")
    if peak_time is not None or centroid_time is not None:
        ax.legend(loc='best', frameon=False)
    ax.set(xlabel='Hours since ICU admission (hourly bin starts)', ylabel='Predicted-class cross-attention',
           xlim=(0, report['hour']))
    fig.tight_layout()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180)
    plt.close(fig)

if __name__ == '__main__':
    main()
