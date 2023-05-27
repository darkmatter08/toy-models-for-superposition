from typing import Optional, List, Dict, Any

from safetensors import safe_open
from safetensors.torch import save_file
import json
from pathlib import Path

def save_model(
    pt_model_state_dict,
    file_path_str:str,
    metadata_dict: Optional[
        Dict[str, Any]
    ]=None
):   

    file_path = Path(file_path_str)
    folder_path = file_path.parent

    if not folder_path.exists():
        folder_path.mkdir()

    save_file(
        tensors=pt_model_state_dict,
        filename=file_path_str,
        metadata=metadata_dict
    )


def save_tensor_dict(
    tensor_dict,
    file_path_str:str,
):   

    file_path = Path(file_path_str)
    folder_path = file_path.parent

    if not folder_path.exists():
        folder_path.mkdir()

    save_file(
        tensors=tensor_dict,
        filename=file_path_str,
    )


def load_tensor_dict(
    file_path_str: str
) -> Dict[str, Any]:
    tensor_dict = {}

    with safe_open(
        file_path_str,
        framework='pt',
        device='cpu'
    ) as f:
        for key in f.keys():
            tensor_dict[key] = f.get_tensor(key)

    return tensor_dict


def load_model(
    file_path_str: str
) -> Dict[str, Any]:
    tensor_dict = {}

    with safe_open(
        file_path_str,
        framework='pt',
        device='cpu'
    ) as f:
        for key in f.keys():
            tensor_dict[key] = f.get_tensor(key)

    return tensor_dict


def load_partial_model(
    file_path_str: str,
    model_key_list: List[str]
) -> Dict[str, Any]:
    tensor_dict = {}
    with safe_open(
        file_path_str,
        framework='pt',
        device='cpu'
    ) as f:
        for key in model_key_list:
            tensor_dict[key] = f.get_tensor(key)
    
    return tensor_dict


def save_dict_to_json(
    json_file_name_str: str,
    data_dict: Dict[str, Any]
):
    with open(json_file_name_str, 'w') as f:
        json.dump(
            data_dict, f, 
            ensure_ascii=False, 
            indent=4
        )


def load_dict_from_json(
    json_file_name_str: str,
):
    with open(json_file_name_str, 'r') as f:
        data_dict = json.load(f)
    return data_dict

if __name__ == "__main__":
    save_tensor_dict(
        tensor_dict={
            "test": 1
        },
        file_path_str='src/superposition_original/test.safetensors'
    )