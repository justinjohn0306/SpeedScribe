
# SpeedScribe

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justinjohn0306/SpeedScribe/blob/main/SpeedScribe.ipynb)

<a href="https://fakeyou.com/"><img src="https://fakeyou.com/fakeyou/FakeYou-Logo.png" alt="FakeYou Logo. Click here to go to the official website." width="200"></a>

SpeedScribe is a high-performance, easy-to-use automated speech recognition (ASR) tool powered by Faster Whisper. It allows for quick transcription of audio files in multiple languages with support for custom models and real-time processing feedback.

## Features

- **Custom Model Support**: Integrates with Faster Whisper, including the `faster-distil-whisper-large-v3` model.
- **Multi-Language Support**: Automatically detects and transcribes audio in different languages.
- **Flexible Input Options**: Upload audio files from Google Drive or your local system.
- **Real-Time Processing**: Provides live updates while transcribing audio files.
- **Automated Dot Processing**: Ensures each transcript line ends with a dot, skipping lines ending with punctuation like `?` or `!`.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/justinjohn0306/SpeedScribe.git
cd SpeedScribe
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Run the ASR Script

To start transcribing your audio files, run:

```bash
python SpeedScribe.py -i <input_folder> -o <output_folder> -s <model_size> -l <language_name> -p <precision>
```

#### Example:

```bash
python SpeedScribe.py -i dataset/audio -o output -s faster-distil-whisper-large-v3 -l English -p float16
```

### 4. Process Transcripts with Dot Processing

If you want to ensure each line in your transcript ends with a dot, use the `dot_processing.py` script:

```bash
python dot_processing.py -i path/to/your/transcript.txt -o path/to/save/processed_transcript.txt
```

This script will automatically add a dot at the end of each line that doesn’t already end with . (dot), ? (question mark), or ! (exclamation mark).

### Usage in Google Colab

SpeedScribe can also be run directly in Google Colab, allowing you to:

- **Upload Audio Files**: Choose to upload files from Google Drive or locally.
- **Real-Time Transcription**: See output as it is being processed.
- **Flexible Language Support**: Easily switch between different languages.

## Customization

- **Model Selection**: Choose from different Faster Whisper models, including the custom `faster-distil-whisper-large-v3` model.
- **Precision Options**: Select between `float16`, `float32`, or `int8` for faster performance or higher accuracy.

## Contributing

Feel free to fork this repository and submit pull requests. Contributions and suggestions are always welcome!

## License

This project is licensed under the MIT License.
