import argparse
import os

def dots_adder_to_txt_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_input, open(output_file, 'w', encoding='utf-8') as f_output:
        for line in f_input:
            line = line.strip()
            # Add a dot only if the line doesn't end with '.', '?', or '!'
            if not line.endswith(('.', '?', '!')):
                line += '.'
            f_output.write(line + '\n')
    print(f"Dots Processing complete! Processed file saved as: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a transcript to ensure each line ends with a dot.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input transcript file.")
    parser.add_argument('-o', '--output', type=str, default="processed_transcript.txt", help="Path to save the processed transcript file.")

    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.isfile(args.input):
        print(f"Error: The file {args.input} does not exist.")
    else:
        dots_adder_to_txt_file(args.input, args.output)