"""Pinned BGE model assets shared by the downloader and index builder."""

EMBEDDINGS = {
    "small": ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
    "base": ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
}

BGE_FILES = [
    "config.json", "config_sentence_transformers.json", "model.safetensors",
    "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json",
]
