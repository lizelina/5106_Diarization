import shutil
from pathlib import Path
from loguru import logger


def delete_all_temp_files():
    """删除指定路径下的所有文件和文件夹，但保留 'data' 文件夹"""
    directories = [
        Path("./data/temp_audios/"),
        Path("./data/transcripts/"),
        Path("./data/transcripts_aligned/"),
        Path("./data/merged_transcript/"),
        Path("./data/diarization/")
    ]

    for directory in directories:
        # 检查目录是否存在
        if directory.exists() and directory.is_dir():
            # 删除整个目录及其内容
            shutil.rmtree(directory)
            logger.info(f"Directory {directory} has been deleted.")
        else:
            logger.warning(f"Directory {directory} does not exist.")

