import argparse
import os
import re
import traceback
import torch
from tqdm import tqdm
from funasr.utils import version_checker
from funasr import AutoModel

path_asr = "tools/asr/models/SenseVoiceSmall"
path_vad  = 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
path_punc = 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
path_asr  = path_asr if os.path.exists(path_asr)  else "iic/SenseVoiceSmall"
path_vad  = path_vad  if os.path.exists(path_vad)  else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_punc = path_punc if os.path.exists(path_punc) else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

version_checker.check_for_update = lambda: None

def execute_asr(input_folder, output_folder, language, device):
    try:
        model = AutoModel(model=path_asr,
                          vad_model=path_vad,
                          vad_kwargs={"max_single_segment_time": 2000},
                          punc_model=path_punc
                          )
    except Exception as e:
        print("Error loading model:", e)
        return
    
    input_file_names = os.listdir(input_folder)
    input_file_names.sort()
    output = []
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for file_name in tqdm(input_file_names):
        try:
            file_path = os.path.join(input_folder, file_name)
            res = model.generate(
                input=file_path,
                cache={},
                language=language.lower(),  # "zn", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False,
                batch_size_s=0,
                device=device
            )[0]['text']
            text = re.sub(r'<[^<>]*>', '', res).replace('  ', '')
            if language.lower() != "en":
                text = text.replace(' ', '')
            output.append(f"{file_path}|{text}")
        except Exception as e:
            print(f"Error processing file {file_name}:", e)
            print(traceback.format_exc())

    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(f'{output_folder}/output.txt')

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        print(f"ASR task completed -> Output file path: {output_file_path}\n")
    
    return output_file_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="Path to the folder containing WAV files.")
    parser.add_argument("-o", "--output_folder", type=str, required=True,
                        help="Output folder to store transcriptions.")
    parser.add_argument("-l", "--language", type=str, default='auto',
                        choices=['auto', 'zh', 'en', 'ja', 'yue', 'ko'],
                        help="Language of the audio files.")
    parser.add_argument("-d", "--device", type=str, default=None, choices=['cpu', 'cuda'],
                        help="CPU or CUDA")
    
    cmd = parser.parse_args()
    output_file_path = execute_asr(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        language=cmd.language,
        device=cmd.device,
    )
