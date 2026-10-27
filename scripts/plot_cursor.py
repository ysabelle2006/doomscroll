
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project directory (automatically detected)
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Folder containing your recordings
DATA_DIR = PROJECT_DIR / "data"

# Find every frames.csv recursively
csv_files = sorted(DATA_DIR.rglob("frames.csv"))

if not csv_files:
    print(f"No frames.csv files found in {DATA_DIR}")
    raise SystemExit(1)

print(f"Found {len(csv_files)} recording sessions:")

plt.figure(figsize=(14, 9))

for csv_file in csv_files:
    # Use the parent folder name as the session label
    session_name = csv_file.parent.name

    print(f"  - {session_name}")

    df = pd.read_csv(csv_file)

    # Check required columns
    if not {"cursor_x", "cursor_y"}.issubset(df.columns):
        print(f"Skipping {session_name}: missing cursor columns")
        continue

    # Remove rows with missing cursor positions
    df = df.dropna(subset=["cursor_x", "cursor_y"])

    if df.empty:
        print(f"Skipping {session_name}: no valid cursor data")
        continue

    # Plot cursor positions
    plt.scatter(
        df["cursor_x"],
        df["cursor_y"],
        s=3,
        alpha=0.3,
        label=session_name,
    )

# Screen coordinates: top-left is (0, 0)
plt.gca().invert_yaxis()

plt.xlabel("Screen X (pixels)")
plt.ylabel("Screen Y (pixels)")
plt.title("Doomscroll - Cursor Movement Map")

plt.legend(
    loc="upper right",
    markerscale=3,
    fontsize=8,
)

plt.grid(alpha=0.15)
plt.tight_layout()

# Save visualization in project folder
output_path = PROJECT_DIR / "cursor_movement_map.png"

plt.savefig(output_path, dpi=200)

print(f"\nMap saved to: {output_path}")

# Display the map
plt.show()