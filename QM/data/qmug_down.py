import requests
import pandas as pd

url = "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=structures.tar.gz"  # example: qmugs_0.parquet
out_path = "structures.tar.gz"

print(f"📥 Downloading {url}")
r = requests.get(url)
with open(out_path, "wb") as f:
    f.write(r.content)
print(f"✅ Saved to {out_path}")

