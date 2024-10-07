import argparse
import os
import traceback

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

# Define the language map
language_map = {
    "Afrikaans": "af",
    "Amharic": "am",
    "Arabic": "ar",
    "Assamese": "as",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Tibetan": "bo",
    "Breton": "br",
    "Bosnian": "bs",
    "Catalan": "ca",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Basque": "eu",
    "Persian": "fa",
    "Finnish": "fi",
    "Faroese": "fo",
    "French": "fr",
    "Galician": "gl",
    "Gujarati": "gu",
    "Hausa": "ha",
    "Hawaiian": "haw",
    "Hebrew": "he",
    "Hindi": "hi",
    "Croatian": "hr",
    "Haitian Creole": "ht",
    "Hungarian": "hu",
    "Armenian": "hy",
    "Indonesian": "id",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jw",
    "Georgian": "ka",
    "Kazakh": "kk",
    "Khmer": "km",
    "Kannada": "kn",
    "Korean": "ko",
    "Latin": "la",
    "Luxembourgish": "lb",
    "Lingala": "ln",
    "Lao": "lo",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Malagasy": "mg",
    "Maori": "mi",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Mongolian": "mn",
    "Marathi": "mr",
    "Malay": "ms",
    "Maltese": "mt",
    "Burmese": "my",
    "Nepali": "ne",
    "Dutch": "nl",
    "Norwegian Nynorsk": "nn",
    "Norwegian": "no",
    "Occitan": "oc",
    "Punjabi": "pa",
    "Polish": "pl",
    "Pashto": "ps",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Sanskrit": "sa",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Shona": "sn",
    "Somali": "so",
    "Albanian": "sq",
    "Serbian": "sr",
    "Sundanese": "su",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Telugu": "te",
    "Tajik": "tg",
    "Thai": "th",
    "Turkmen": "tk",
    "Tagalog": "tl",
    "Turkish": "tr",
    "Tatar": "tt",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Chinese": "zh",
    "Cantonese": "yue",
    "Automatic Detection": "auto"
}

def execute_asr(input_folder, output_folder, model_size, language_name, precision):
    language = language_map.get(language_name, "auto")  # Get the language code from the selected language name
    
    if '-local' in model_size:
        model_size = model_size[:-6]
        model_path = f'tools/asr/models/faster-whisper-{model_size}'
    else:
        # Use "faster-whisper-large-v3-turbo-ct2" as the custom model
        model_path = "deepdml/faster-whisper-large-v3-turbo-ct2" if model_size == "faster-whisper-large-v3-turbo-ct2" else model_size
    
    if language == 'auto':
        language = None  # Don't set language, let the model automatically determine the language
    
    print(f"Loading Faster Whisper model: {model_size} ({model_path})")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = WhisperModel(model_path, device=device, compute_type=precision)
    except:
        return print(traceback.format_exc())
    
    input_file_names = os.listdir(input_folder)
    input_file_names.sort()

    output = []
    output_file_name = os.path.basename(input_folder)
    
    for file_name in tqdm(input_file_names):
        try:
            file_path = os.path.join(input_folder, file_name)
            segments, info = model.transcribe(
                audio=file_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=language
            )
            text = ''
            
            for segment in segments:
                text += segment.text

            output.append(f"{file_path}|{output_file_name}|{info.language.upper()}|{text}")
        except:
            print(traceback.format_exc())
    
    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(f'{output_folder}/{output_file_name}.txt')

    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in output:
            file_path, _, _, transcription_text = line.split('|')
            relative_file_path = os.path.relpath(file_path, input_folder)
            # Strip any leading/trailing whitespace from the transcription text
            f.write(f"wavs/{relative_file_path}|{transcription_text.strip()}\n")
        print(f"ASR task completed -> Output file path: {output_file_path}\n")
    
    return output_file_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="Path to the folder containing WAV files.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, 
                        help="Output folder to store transcriptions.")
    parser.add_argument("-s", "--model_size", type=str, default='faster-distil-whisper-large-v3', 
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "faster-whisper-large-v3-turbo-ct2"],
                        help="Model size for Faster Whisper or 'faster-whisper-large-v3-turbo-ct2' for the custom Hugging Face model.")
    parser.add_argument("-l", "--language_name", type=str, default='English',
                        choices=list(language_map.keys()),
                        help="Language of the audio files (full name).")
    parser.add_argument("-p", "--precision", type=str, default='float16', choices=['float16','float32','int8'],
                        help="Precision options: float16, float32, or int8")

    cmd = parser.parse_args()
    output_file_path = execute_asr(
        input_folder=cmd.input_folder,
        output_folder=cmd.output_folder,
        model_size=cmd.model_size,
        language_name=cmd.language_name,
        precision=cmd.precision,
    )
