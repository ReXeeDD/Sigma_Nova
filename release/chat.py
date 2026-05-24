"""
SigmaNOVA Standalone Chat
Run this file to load the weights and chat with the model!
"""
import argparse
import os
import sys

# Import the local stripped-down model
from sigma_nova import SigmaNOVA

def main():
    parser = argparse.ArgumentParser(description="SigmaNOVA Chat Interface")
    parser.add_argument("--weights", type=str, default="model/nova_v6_weights_instruct.pt",
                       help="Path to the model weights (.pt file)")
    parser.add_argument("--tokenizer", type=str, default="model/nova_tokenizer.json",
                       help="Path to the tokenizer JSON file")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  Booting SigmaNOVA Brain...")
    print("="*50)
    
    if not os.path.exists(args.weights):
        print(f"Error: Could not find weights at {args.weights}")
        print("Please download the weights and place them in the 'model' folder.")
        sys.exit(1)

    # Initialize model architecture
    model = SigmaNOVA(vocab_size=32000, context_dim=512, pc_dims=[512, 1024, 512], tokenizer_path=args.tokenizer)
    
    # Load weights
    print(f"Loading weights from {args.weights}...")
    model.load(args.weights)
    print("Brain loaded successfully!\n")

    print("SigmaNOVA Enhanced Conversational Chat")
    print("=" * 50)
    print("Commands:")
    print("  'quit' or 'exit' - End conversation")
    print("  'temp X' - Set temperature (0.1-2.0, default 0.3)")
    print("  'len X'  - Set response length (10-100, default 60)")
    print()

    temperature = 0.3
    response_length = 60

    while True:
        try:
            prompt = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not prompt:
            continue

        if prompt.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if prompt.lower().startswith("temp "):
            try:
                temperature = float(prompt[5:].strip())
                temperature = max(0.1, min(2.0, temperature))
                print(f"Temperature set to {temperature}")
                continue
            except ValueError:
                print("Please provide a valid number for temperature (e.g., 'temp 0.8')")
                continue

        if prompt.lower().startswith("len "):
            try:
                response_length = int(prompt[4:].strip())
                response_length = max(10, min(200, response_length))
                print(f"Response length set to {response_length}")
                continue
            except ValueError:
                print("Please provide a valid number for length (e.g., 'len 50')")
                continue

        # Format using the SEP token
        formatted_prompt = f"{prompt.strip()} {model.SEP_TOKEN}"
        
        # Generate response
        response = model.generate(formatted_prompt, n_words=response_length, temperature=temperature)

        # Strip SEP token if it leaked into the response
        response = response.replace(model.SEP_TOKEN, "").strip()
        if model.SEP_TOKEN in response:
            response = response.split(model.SEP_TOKEN)[0].strip()

        print(f"SigmaNOVA > {response}\n")

if __name__ == "__main__":
    main()
