from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    classifier_joblib_path: str = './model/classifier.joblib'
    embedding_dim:int = 384
    s3_bucket: str = 'mlops-lab11-models-jk'
    onnx_classifier_path: str= './model/classifier.onnx'
    sentence_transformer_dir: str = './model/sentence_transformer.model'
    onnx_embedding_model_path: str = './model/embedding_model.onnx'
    onnx_tokenizer_path: str = './model/tokenizer.json'

