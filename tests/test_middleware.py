import unittest
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from app.middleware import ExceptionHandler, TokenAuthentication, ResponseTime


async def _exception_handler_app(scope, receive, send):
    assert scope["type"] == "http"
    path = scope["path"]

    if path == "/http-error":
        raise HTTPException(status_code=403, detail="Forbidden")
    elif path == "/generic-error":
        raise RuntimeError("Boom")

    response = PlainTextResponse("OK")
    await response(scope, receive, send)


class TestExceptionHandler(unittest.TestCase):
    def setUp(self):
        self.app = ExceptionHandler(_exception_handler_app)
        self.client = TestClient(self.app)

    def test_ok_response(self):
        response = self.client.get("/ok")
        self.assertEqual(response.status_code, 200)

    def test_http_exception(self):
        response = self.client.get("/http-error")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"message": "Forbidden"})

    def test_generic_exception(self):
        with patch("app.middleware.logging.error"):
            response = self.client.get("/generic-error")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"message": "Something went wrong."})


class TestTokenAuthentication(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(TokenAuthentication, api_token="test-token")

        @self.app.get("/ok")
        async def ok():
            return {"status": "ok"}

        self.client = TestClient(self.app)

    def test_no_token(self):
        response = self.client.get("/ok")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"message": "Unauthorized."})

    def test_wrong_token(self):
        response = self.client.get(
            "/ok", headers={"Authorization": "Bearer wrong-token"}
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"message": "Unauthorized."})

    def test_valid_token(self):
        response = self.client.get(
            "/ok", headers={"Authorization": "Bearer test-token"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_init_with_empty_token(self):
        with self.assertRaises(ValueError):
            TokenAuthentication(app=None, api_token="")

    def test_init_with_valid_token(self):
        TokenAuthentication(app=None, api_token="valid-token")


class TestResponseTime(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.add_middleware(ResponseTime)

        @self.app.get("/ok")
        async def ok():
            return {"status": "ok"}

        self.client = TestClient(self.app)

    def test_response_time_header(self):
        response = self.client.get("/ok")
        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Response-Time", response.headers)
        float(response.headers["X-Response-Time"])
