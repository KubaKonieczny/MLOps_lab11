import argparse

from src.scripts.download_artifacts import download_artifacts
from src.scripts.export_classifier_to_onnx import export_classifier_to_onnx
from src.scripts.export_sentence_transformer_to_onnx import export_model_to_onnx
from src.scripts.settings import Settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load environment variables from specified.env file."
    )
    parser.add_argument(
        "--type",
        type=str,
        default="download",
        help="download or export",
    )
    args = parser.parse_args()

    settings = Settings()

    if args.type == 'download':
        download_artifacts(settings)

    if args.type == 'export':
        export_classifier_to_onnx(settings)
        export_model_to_onnx(settings)


