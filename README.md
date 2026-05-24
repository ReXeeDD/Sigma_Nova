# SigmaNOVA 

SigmaNOVA is a custom 18-Million parameter language model trained from scratch using a novel predictive coding hierarchical architecture. It achieves 100% recall on factual information and identity in benchmark testing despite its extremely small footprint.

Because of its recurrent Gated Context Memory architecture, it scales linearly `O(N)` with sequence length, making it incredibly fast and memory efficient compared to traditional Transformer models.

## Installation

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pre-trained weights and tokenizer from our Hugging Face repository:
   - **[Hugging Face Repo: ReXeeD / SigmaNOVA-Instruct-18M](https://huggingface.co/ReXeeD/SigmaNOVA-Instruct-18M)**
4. Create a folder named `model` inside the project directory.
5. Place the downloaded files inside the `model/` folder so your directory structure looks like this:
   - `model/nova_v6_weights_instruct.pt`
   - `model/nova_tokenizer.json`

## Chatting with the Model

To launch the interactive chat interface, simply run:

```bash
python chat.py
```

### In-Chat Commands:
- `temp 0.3` - Adjust the generation temperature (lower = more deterministic, higher = more creative).
- `len 50` - Set the maximum number of tokens to generate per response.
- `quit` - Exit the chat.

## Training Data & Purpose
This version of SigmaNOVA is primarily a **proof-of-concept test model** to demonstrate the capabilities of its novel architecture.
- **Base Brain (Grammar):** Pre-trained on the TinyStories dataset to learn basic English grammar and sentence structure(~35,302,519 words).
- **Instruction Tuning:** Fine-tuned on a micro-dataset of just **298 question-answer pairs** (~4,000 words).

Despite this extremely limited fine-tuning data, the model's new architecture allows it to achieve 100% recall on factual information and identity benchmarks!

## Example Interactions
Here are some real examples of the model's responses during benchmark testing:

**Greetings & Identity**
- **User:** What is your name?
- **SigmaNOVA:** My name is SigmaNOVA. I was built to help you.

**Factual Recall**
- **User:** What is the sun?
- **SigmaNOVA:** The sun is a star that gives Earth light and heat.
- **User:** What is the moon?
- **SigmaNOVA:** The moon is a rocky body that orbits the Earth.

**Reasoning & Concepts**
- **User:** What is the opposite of hot?
- **SigmaNOVA:** The opposite of hot is cold.
- **User:** What color is the sky?
- **SigmaNOVA:** The sky is blue during the day.

## Architecture Highlights
- **Parameters:** 18,221,568
- **Vocab Size:** 32,000 (Custom BPE Tokenizer)
- **Context Dimension:** 512
- **Topology:** Gated Context Memory + 3-Layer Predictive Coding Hierarchy (512 -> 1024 -> 512)
- **Training:** Full Backpropagation Through Time (BPTT) with `<|sep|>` and `<|end|>` boundary tokens.


## ⚖️ License
This project uses a split-license approach to best protect both the software and the data:

* **Codebase & Scripts:** Licensed under the [Apache License 2.0](LICENSE).
* **Model Weights:** Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to use, modify, and build commercial products with SigmaNOVA

## 📝 How to Give Credit
If you use this model or code in your project, you must provide proper attribution:
1. For the **code**, you must include the contents of the `NOTICE` file in your documentation or app credits.
2. For the **model weights**, you must credit **[Albin Thomas]** and link back to this repository.
