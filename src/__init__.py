"""
src package — ISOT Fake News Detection project modules.

Exposes the primary data-pipeline, model, and training entry points
for convenient imports:

    from src import create_dataloaders, build_model, train, load_config
"""

from src.utils import load_config, set_seed, build_vocab, clean_text, tokenize
from src.dataset import create_dataloaders, FakeNewsDataset, load_and_merge_data
from src.model import FakeNewsLSTM, build_model
from src.matrics import compute_metrics, log_metrics, get_classification_report
from src.trainer import train
