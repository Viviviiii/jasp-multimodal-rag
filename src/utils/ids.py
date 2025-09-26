from pathlib import Path

def source_name_from_path(path: str) -> str:
    return Path(path).name

def text_chunk_id(source: str, page: int, t_idx: int) -> str:
    return f"{source}:p{page}:t{t_idx}"

def image_id(source: str, page: int, i_idx: int) -> str:
    return f"{source}:p{page}:i{i_idx}"

def image_caption_id(source: str, page: int, i_idx: int) -> str:
    return f"{source}:p{page}:ic{i_idx}"
