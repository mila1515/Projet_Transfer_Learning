import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from PIL import Image

from api import main


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def _request(self, method: str, path: str, **kwargs):
        transport = ASGITransport(app=main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    @staticmethod
    def _png() -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")
        return buffer.getvalue()

    async def test_health_reports_missing_model(self) -> None:
        with patch.object(main, "MODEL_PATH", Path("missing-model.keras")):
            response = await self._request("GET", "/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "model_missing")

    async def test_predict_rejects_non_image_upload(self) -> None:
        response = await self._request(
            "POST",
            "/predict",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    async def test_predict_returns_model_result(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".keras") as model_file:
            expected = {
                "prediction": "PNEUMONIA",
                "probability": 0.91,
                "threshold": 0.5,
            }
            with (
                patch.object(main, "MODEL_PATH", Path(model_file.name)),
                patch.object(main, "predict_from_pil", return_value=expected),
            ):
                response = await self._request(
                    "POST",
                    "/predict",
                    files={"file": ("xray.png", self._png(), "image/png")},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)


if __name__ == "__main__":
    unittest.main()
