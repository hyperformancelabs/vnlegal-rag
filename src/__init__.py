"""VNLegal RAG — full retrieval pipeline v1.3.

Importable package extracted from ``pipeline_v1.3/test-pipeline-v1_3.ipynb`` and
organized into subpackages (``data``, ``models``, ``retrieval``, ``evaluation``,
``utils``) with a top-level :mod:`~src.factory` and :mod:`~src.pipeline`.

Quick start::

    from src import PipelineConfig, RetrievalPipeline
    pipe = RetrievalPipeline.from_config(PipelineConfig())
    results = pipe.retrieve_tfidf_textcnn("Hợp đồng lao động thời vụ ...")
    metrics = pipe.evaluate("test")
"""

from __future__ import annotations


def _warmup_torch_conv_backend() -> None:
    """Initialize torch's conv/OpenMP backend before sklearn is imported.

    On some Windows hosts, the first torch ``Conv1d`` forward executed *after*
    ``sklearn`` is imported hits a conflicting OpenMP runtime and crashes with an
    uncatchable stack overflow. Running a trivial conv here — before the
    submodule imports below pull in sklearn — claims the backend first and makes
    the package importable everywhere. No-op cost (~1 tiny tensor); never fatal.
    """
    try:
        import torch

        # Use realistic channel/length dims so the multithreaded oneDNN conv
        # path (not just the trivial single-thread kernel) is initialized.
        with torch.no_grad():
            torch.nn.Conv1d(200, 100, 3)(torch.zeros(1, 200, 256))
    except Exception:
        pass


_warmup_torch_conv_backend()

from .evaluation import build_markdown_table, evaluate_retriever
from .models import LSTMEncoder, SiameseLSTM, TextCNN
from .pipeline import PipelineConfig, RetrievalPipeline, seed_everything
from .utils import default_device

__all__ = [
    "PipelineConfig",
    "default_device",
    "RetrievalPipeline",
    "seed_everything",
    "evaluate_retriever",
    "build_markdown_table",
    "TextCNN",
    "LSTMEncoder",
    "SiameseLSTM",
]
