import unittest
from unittest.mock import MagicMock, patch, sentinel

from app.model import ESM3Model


class TestESM3Model(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.parameters.return_value = []
        self.mock_model.to.return_value = self.mock_model

    @patch("app.model.cuda_is_available", return_value=False)
    @patch("app.model.mps_is_available", return_value=False)
    def test_invalid_model_name(self, mock_mps, mock_cuda):
        with self.assertRaises(ValueError) as ctx:
            ESM3Model("nonexistent", "cpu", False, 192, 1)
        self.assertIn("nonexistent", str(ctx.exception))

    @patch("app.model.cuda_is_available", return_value=False)
    def test_cuda_not_available(self, mock_cuda):
        with self.assertRaises(ValueError) as ctx:
            ESM3Model("esm3-open", "cuda", False, 192, 1)
        self.assertIn("CUDA", str(ctx.exception))

    @patch("app.model.cuda_is_available", return_value=True)
    @patch("app.model.mps_is_available", return_value=False)
    def test_mps_not_available(self, mock_mps, mock_cuda):
        with self.assertRaises(ValueError) as ctx:
            ESM3Model("esm3-open", "mps", False, 192, 1)
        self.assertIn("MPS", str(ctx.exception))

    @patch("app.model.cuda_is_available", return_value=False)
    @patch("app.model.mps_is_available", return_value=False)
    def test_max_concurrency_less_than_one(self, mock_mps, mock_cuda):
        with self.assertRaises(ValueError) as ctx:
            ESM3Model("esm3-open", "cpu", False, 192, 0)
        self.assertIn("Max concurrency", str(ctx.exception))

    @patch("app.model.quantize_")
    @patch("app.model.ESM3")
    @patch("app.model.cuda_is_available", return_value=False)
    @patch("app.model.mps_is_available", return_value=False)
    def test_successful_init_cpu(
        self, mock_mps, mock_cuda, mock_esm3_cls, mock_quantize_
    ):
        mock_esm3_cls.from_pretrained.return_value = self.mock_model

        instance = ESM3Model("esm3-open", "cpu", False, 192, 2)

        self.assertEqual(instance.name, "esm3-open")
        self.assertEqual(instance.device, "cpu")
        self.assertEqual(instance.max_concurrency, 2)
        self.mock_model.get_structure_encoder.assert_called_once()
        self.mock_model.get_structure_decoder.assert_called_once()
        self.mock_model.get_function_decoder.assert_called_once()
        self.mock_model.to.assert_any_call("cpu")
        self.mock_model.eval.assert_called_once()
        mock_quantize_.assert_not_called()

    @patch("app.model.quantize_")
    @patch("app.model.ESM3")
    @patch("app.model.cuda_is_available", return_value=True)
    @patch("app.model.mps_is_available", return_value=False)
    def test_successful_init_cuda_with_quantize(
        self, mock_mps, mock_cuda, mock_esm3_cls, mock_quantize_
    ):
        mock_esm3_cls.from_pretrained.return_value = self.mock_model

        instance = ESM3Model("esm3-open", "cuda", True, 64, 1)

        self.assertEqual(instance.device, "cuda")
        mock_quantize_.assert_called_once()
        self.mock_model.eval.assert_called_once()

    @patch("app.model.quantize_")
    @patch("app.model.ESM3")
    @patch("app.model.cuda_is_available", return_value=False)
    @patch("app.model.mps_is_available", return_value=False)
    def test_num_parameters(
        self, mock_mps, mock_cuda, mock_esm3_cls, mock_quantize_
    ):
        mock_model = MagicMock()
        mock_model.parameters.return_value = [
            MagicMock(numel=lambda: 50),
            MagicMock(numel=lambda: 30),
        ]
        mock_model.to.return_value = mock_model
        mock_esm3_cls.from_pretrained.return_value = mock_model

        instance = ESM3Model("esm3-open", "cpu", False, 192, 1)
        self.assertEqual(instance.num_parameters, 80)

    @patch("app.model.quantize_")
    @patch("app.model.ESM3")
    @patch("app.model.cuda_is_available", return_value=False)
    @patch("app.model.mps_is_available", return_value=False)
    def test_generate(
        self, mock_mps, mock_cuda, mock_esm3_cls, mock_quantize_
    ):
        mock_model = MagicMock()
        mock_model.generate.return_value = sentinel.result
        mock_model.to.return_value = mock_model
        mock_esm3_cls.from_pretrained.return_value = mock_model

        instance = ESM3Model("esm3-open", "cpu", False, 192, 1)

        protein = MagicMock()
        config = MagicMock()
        result = instance.generate(protein, config)

        self.assertIs(result, sentinel.result)
        mock_model.generate.assert_called_once_with(protein, config)
