"""
内置数据集管理 - 支持音频文件下载
优先使用本地缓存，无则联网下载
"""
import os
import json
import urllib.request
import zipfile
import shutil
from typing import Dict, List, Optional, Union

# 数据集下载地址
DATASET_URLS = {
    "zh-en-littleprince": "https://github.com/sjtuayj/OpenST/releases/download/v0.1.0/zh-en-littleprince.zip",
}

# 默认缓存目录
# DEFAULT_CACHE_DIR = os.path.expanduser("~/.datasets")
DEFAULT_CACHE_DIR = "./datasets"

class Dataset:
    """内置数据集类"""
    
    def __init__(self, data: List[Dict], base_dir: str):
        self._data = data
        self._base_dir = base_dir
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self._data[idx].copy()
        if "source_speech_path" in item:
            filename = os.path.basename(item["source_speech_path"])
            item["source_speech_path"] = os.path.join(self._base_dir, "audio", filename)
        return item
    
    @property
    def ids(self) -> List[str]:
        return [item.get("id", f"sample_{i}") for i, item in enumerate(self._data)]
    
    @property
    def source_texts(self) -> List[str]:
        return [item["source_text"] for item in self._data]
    
    @property
    def reference_texts(self) -> List[str]:
        return [item["reference_text"] for item in self._data]
    
    @property
    def audio_paths(self) -> List[str]:
        return [self[i].get("source_speech_path", "") for i in range(len(self))]
    
    def verify_audio_files(self) -> Dict[str, Union[int, List[str]]]:
        """验证音频文件完整性"""
        missing = [p for p in self.audio_paths if not os.path.exists(p)]
        return {
            "total": len(self),
            "found": len(self) - len(missing),
            "missing": len(missing),
            "missing_files": missing,
        }


def list_datasets() -> List[str]:
    """列出所有可用数据集"""
    return list(DATASET_URLS.keys())


def _is_dataset_cached(name: str, cache_dir: str) -> bool:
    """检查数据集是否已下载"""
    dataset_dir = os.path.join(cache_dir, name)
    data_file = os.path.join(dataset_dir, "dataset_paired.json")
    audio_dir = os.path.join(dataset_dir, "audio")
    return os.path.exists(data_file) and os.path.exists(audio_dir)


def get_dataset_info(name: str, cache_dir: Optional[str] = None) -> Dict:
    """获取数据集信息"""
    if name not in DATASET_URLS:
        raise ValueError(f"未知数据集: {name}")
    
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    is_cached = _is_dataset_cached(name, cache_dir)
    
    info = {
        "name": name,
        "url": DATASET_URLS[name],
        "cache_dir": os.path.join(cache_dir, name),
        "is_downloaded": is_cached,
    }
    
    if is_cached:
        dataset = load_dataset(name, cache_dir=cache_dir)
        info["num_samples"] = len(dataset)
        verify = dataset.verify_audio_files()
        info["audio_complete"] = verify["missing"] == 0
    
    return info


def load_dataset(
    name: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> Dataset:
    """
    加载数据集（优先本地，无则下载）
    
    Args:
        name: 数据集名称
        cache_dir: 缓存目录
        force_download: 强制重新下载
    
    Returns:
        Dataset 对象
    """
    if name not in DATASET_URLS:
        available = ", ".join(DATASET_URLS.keys())
        raise ValueError(f"未知数据集: {name}。可用: {available}")
    
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    dataset_dir = os.path.join(cache_dir, name)
    data_file = os.path.join(dataset_dir, "dataset_paired.json")
    
    # 检查本地缓存
    if _is_dataset_cached(name, cache_dir) and not force_download:
        print(f"✅ [Local] 使用本地数据集: {name}")
        print(f"   路径: {dataset_dir}")
    else:
        print(f"⏳ [Online] 下载数据集: {name}")
        _download_dataset(name, cache_dir)
    
    # 加载数据
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = Dataset(data=data, base_dir=dataset_dir)
    
    # 验证完整性
    verify = dataset.verify_audio_files()
    if verify["missing"] > 0:
        print(f"   ⚠️ 缺少 {verify['missing']} 个音频文件")
    else:
        print(f"   ✅ 数据完整 ({verify['total']} 条样本, 音频齐全)")
    
    return dataset


def _download_dataset(name: str, cache_dir: str):
    """下载数据集"""
    url = DATASET_URLS[name]
    dataset_dir = os.path.join(cache_dir, name)
    zip_path = os.path.join(cache_dir, f"{name}.zip")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    print(f"   URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, zip_path, _download_progress)
        print()
        
        print(f"   📦 解压中...")
        os.makedirs(dataset_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dataset_dir)
        
        os.remove(zip_path)
        print(f"   ✅ 下载完成: {dataset_dir}")
        
    except Exception as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        raise RuntimeError(f"下载失败: {e}")


def _download_progress(block_num, block_size, total_size):
    """下载进度条"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        bar_len = 40
        filled = int(bar_len * percent // 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r   [{bar}] {percent}%", end="", flush=True)


def create_dataset_from_json(json_path: str) -> Dataset:
    """从本地 JSON 创建数据集"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(json_path))
    return Dataset(data=data, base_dir=base_dir)
