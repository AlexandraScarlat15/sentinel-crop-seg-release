import requests
from sentinel_crop_seg.core.auth.copernicus_auth_manager import CopernicusAuthManager


def resolve_uuid_from_odata(product_name: str) -> str:
    auth = CopernicusAuthManager()
    token = auth.get_access_token()

    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    params = {
        "$filter": f"Name eq '{product_name}.SAFE'",
        "$top": 1,
    }
    headers = {
        "Authorization": f"Bearer {token}"
    }

    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()

    items = r.json().get("value", [])
    if not items:
        raise RuntimeError(f"OData UUID not found for {product_name}")

    return items[0]["Id"]
