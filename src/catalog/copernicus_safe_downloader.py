from pathlib import Path
import requests
import zipfile
import shutil
from src.auth.copernicus_auth_manager import CopernicusAuthManager


class CopernicusSafeDownloader:
    """
    Downloads Sentinel SAFE products via Copernicus Zipper API.
    """

    DOWNLOAD_URL = (
        "https://download.dataspace.copernicus.eu/odata/v1/"
        "Products({product_id})/$value"
    )

    def __init__(self) -> None:
        self.auth = CopernicusAuthManager()

    def download_safe(self, product_id: str, out_dir: Path) -> Path:
        """
        Download and extract a SAFE product.

        Returns
        -------
        Path
            Path to extracted SAFE directory.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        token = self.auth.get_access_token()

        url = self.DOWNLOAD_URL.format(product_id=product_id)
        zip_path = out_dir / f"{product_id}.zip"

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/zip",
        }

        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()

            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # Extract
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)

        zip_path.unlink(missing_ok=True)

        safe_dirs = list(out_dir.glob("*.SAFE"))
        if not safe_dirs:
            raise RuntimeError("SAFE extraction failed")

        return safe_dirs[0]
