<h1 id="top" align="center">Awesome LLM</h1>
<p align="center"><a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome" /></a></p>
<p align="center">Awesome series for Large Language Model(LLM)s</p>

<p align="center"><img width="880" src="./cover.png" /></p>

## Contents

- [Models](#models)
   - [Overview](#overview)
   - [Open models](#open-models)
   - [Projects](#projects)
   - [GitHub repositories](#github-repositories)
   - [HuggingFace repositories](#huggingface-repositories)
   - [Commercial models](#commercial-models)
      - [GPT](#gpt)
      - [Bard](#bard)
      - [Codex](#codex)
- [Datasets](#datasets)
- [Benchmarks](#benchmarks)
- [Materials](#materials)
   - [Papers](#papers)
   - [Posts](#posts)
   - [Projects](#projects)
   - [Reading materials](#reading-materials)
- [Contributing](#contributing)

## Models

### Overview

| Name                                      | Parameter Size                     | Announcement Date   | Provider                                      |
|-------------------------------------------|------------------------------------|---------------------|-----------------------------------------------|
| Grok‑3                                    | Undisclosed                        | Feburary 2025       | xAI                                           |
| DeepSeek R1                               | 671 billion                        | January 2025        | DeepSeek                                      |
| DeepSeek V3                               | 671 billion                        | December 2024       | DeepSeek                                      |
| Qwen 2.5 Max                              | Undisclosed                        | June 2024           | Alibaba Cloud                                 |
| Gemma 2                                   | 27 billion                         | June 2024           | Google                                        |
| DeepSeek VL2                              | 4.5 billion                        | June 2024           | DeepSeek                                      |
| Qwen 1.5‑MoE‑A2.7B                        | 14.9B (2.7B active)                | May 2024            | Alibaba Cloud                                 |
| Yi‑1.5‑6B‑Chat                            | 6 billion                          | May 2024            | 01.AI                                         |
| Yi‑1.5‑9B                                 | 9 billion                          | May 2024            | 01.AI                                         |
| DeepSeek V2                               | 238 billion (21B active)           | May 2024            | DeepSeek                                      |
| DeepSeek VL                               | 7 billion                          | May 2024            | DeepSeek                                      |
| Microsoft Phi‑2                           | 2.7 billion                        | April 2024          | Microsoft                                     |
| Llama 3 (8B, 70B)                         | 8B, 70B                            | April 2024          | Meta                                          |
| Gemma 1.1                                 | 7 billion                          | April 2024          | Google                                        |
| Claude 3 (Opus, Sonnet, Haiku)            | Undisclosed                        | March 2024          | Anthropic                                     |
| DBRX (132B total, 36B active)             | 132 billion (36B active)           | March 2024          | Databricks                                    |
| Grok‑1                                    | 314 billion                        | March 2024          | xAI                                           |
| Gemma (2B, 7B)                            | 2B, 7B                             | February 2024       | Google                                        |
| Qwen 1.5 (0.5B, 1.8B, 4B, 7B, 14B, 72B)   | 0.5B, 1.8B, 4B, 7B, 14B, 72B       | February 2024       | Alibaba Cloud                                 |
| Solar                                     | 10.7 billion                       | December 2023       | Upstage                                       |
| Microsoft Phi‑2                           | 2.7 billion                        | December 2023       | Microsoft                                     |
| Google Gemini (Pro, Ultra, Nano)          | Undisclosed                        | December 2023       | Google                                        |
| Mixtral 8x7B                              | 46.7 billion (active)              | December 2023       | Mistral AI                                    |
| Grok‑0                                    | 33 billion                         | November 2023       | xAI                                           |
| Yi‑34B‑200K                               | 34 billion                         | November 2023       | 01.AI                                         |
| Qwen‑VL / Qwen‑VL‑Chat                    | Undisclosed                        | Oct/Nov 2023        | Alibaba Cloud                                 |
| Zephyr‑7b‑beta                            | 7 billion                          | October 2023        | HuggingFace H4                                |
| Mistral 7B                                | 7.3 billion                        | September 2023      | Mistral AI                                    |
| Qwen‑14B‑Chat                             | 14 billion (est.)                  | September 2023      | Alibaba Cloud                                 |
| Qwen‑14B                                  | 14 billion (est.)                  | September 2023      | Alibaba Cloud                                 |
| Qwen‑7B‑Chat                              | 7 billion (est.)                   | August 2023         | Alibaba Cloud                                 |
| Qwen‑7B                                   | 7 billion (est.)                   | August 2023         | Alibaba Cloud                                 |
| Llama 2 (7B, 13B, 70B)                    | 7B, 13B, 70B                       | July 2023           | Meta                                          |
| XGen (7B)                                 | 7 billion                          | July 2023           | Salesforce                                    |
| Falcon (7B, 40B, 180B)                    | 7B, 40B, 180B                      | June/Sept 2023      | Technology Innovation Institute (UAE)         |
| MPT (7B, 30B)                             | 7B, 30B                            | May/June 2023       | MosaicML                                      |
| LIMA (65B)                                | 65 billion                         | May 2023            | Meta AI                                       |
| PaLM 2                                    | Undisclosed                        | May 2023            | Google                                        |
| Vicuna (7B, 13B, 33B)                     | 7B, 13B, 33B                       | March 2023          | LMSYS ORG                                     |
| Koala (13B)                               | 13 billion                         | April 2023          | UC Berkeley                                   |
| OpenAssistant (LLaMA 30B)                 | 30 billion                         | April 2023          | LAION                                         |
| Jurassic‑2                                | Undisclosed                        | April 2023          | AI21 Labs                                     |
| Dolly (Databricks) (6B and 12B)           | 6 & 12 billion                     | March/April 2023    | Databricks                                    |
| BloombergGPT                              | 50 billion                         | March 2023          | Bloomberg                                     |
| GPT‑4                                     | Undisclosed                        | March 2023          | OpenAI                                        |
| Bard                                      | Undisclosed                        | March 2023          | Google                                        |
| Stanford Alpaca (7B)                      | 7 billion                          | March 2023          | Stanford University                           |
| LLaMA (7B, 13B, 33B, 65B)                 | 7B, 13B, 33B, 65B                  | February 2023       | Meta                                          |
| ChatGPT                                   | Undisclosed                        | November 2022       | OpenAI                                        |
| GPT‑3.5 (series)                          | 175 billion (for largest models)   | November 2022       | OpenAI                                        |
| Jurassic‑1 (178B)                         | 178 billion                        | November 2022       | AI21                                          |
| Galactica (120B)                          | 120 billion                        | November 2022       | Meta                                          |
| Sparrow (70B)                             | 70 billion                         | September 2022      | DeepMind                                      |
| NLLB (54.5B)                              | 54.5 billion                       | July 2022           | Meta                                          |
| BLOOM (176B)                              | 176 billion                        | July 2022           | BigScience (Hugging Face)                     |
| AlexaTM (20B)                             | 20 billion                         | August 2022         | Amazon                                        |
| UL2 (20B)                                 | 20 billion                         | May 2022            | Google                                        |
| OPT‑175B                                  | 175 billion                        | May 2022            | Meta (Facebook)                               |
| PaLM (540B)                               | 540 billion                        | April 2022          | Google                                        |
| AlphaCode (41.4B)                         | 41.4 billion                       | February 2022       | DeepMind                                      |
| Chinchilla (70B)                          | 70 billion                         | March 2022          | DeepMind                                      |
| GLaM (1.2T)                               | 1.2 trillion                       | December 2021       | Google                                        |
| Macaw (11B)                               | 11 billion                         | October 2021        | Allen Institute for AI                        |
| T0 (11B)                                  | 11 billion                         | October 2021        | Hugging Face                                  |
| Megatron‑Turing NLG (530B)                | 530 billion                        | January 2022        | Microsoft & NVIDIA                            |
| LaMDA (137B)                              | 137 billion                        | January 2022        | Google                                        |
| Gopher (280B)                             | 280 billion                        | December 2021       | DeepMind                                      |
| GPT‑J (6B)                                | 6 billion                          | June 2021           | EleutherAI                                    |
| GPT‑NeoX 2.0 (20B)                        | 20 billion                         | February 2022       | EleutherAI                                    |
| T5 (11B)                                  | 11 billion                         | October 2019        | Google                                        |
| BERT‑Large                                | 336 million                        | October 2018        | Google                                        |


[:arrow_up: Go to top](#top)

### Open models

- [T5 (11B)](https://huggingface.co/docs/transformers/model_doc/t5) - Announced by Google / 2020
- [T5 FLAN (540B)](https://huggingface.co/google/flan-t5-xxl) - Announced by Google / 2022
- [T0 (11B)](https://huggingface.co/bigscience/T0pp) - Announced by BigScience (HuggingFace) / 2021
- [OPT-175B (175B)](https://huggingface.co/docs/transformers/model_doc/opt) - Announced by Meta / 2022
- [UL2 (20B)](https://ai.googleblog.com/2022/10/ul2-20b-open-source-unified-language.html) - Announced by Google / 2022
- [Bloom (176B)](https://huggingface.co/bigscience/bloom) - Announced by BigScience (HuggingFace) / 2022
- [BERT-Large (336M)](https://huggingface.co/bert-large-uncased) - Announced by Google / 2018
- [GPT-NeoX 2.0 (20B)](https://github.com/EleutherAI/gpt-neox) - Announced by EleutherAI / 2023
- [GPT-J (6B)](https://huggingface.co/EleutherAI/gpt-j-6B) - Announced by EleutherAI / 2021
- [Macaw (11B)](https://macaw.apps.allenai.org/) - Announced by AI2 / 2021
- [Stanford Alpaca (7B)](https://crfm.stanford.edu/2023/03/13/alpaca.html) - Announced by Stanford University / 2023

[:arrow_up: Go to top](#top)

### Projects

- [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) - Announced by Microsoft / 2023
- [LMOps](https://github.com/microsoft/lmops) - Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities.

[:arrow_up: Go to top](#top)

### Commercial models

#### GPT

- [GPT 4 (Parameter size unannounced, gpt-4-32k)](https://openai.com/product/gpt-4) - Announced by OpenAI / 2023
- [ChatGPT (175B)](https://openai.com/blog/chatgpt/) - Announced by OpenAI / 2022
- [ChatGPT Plus (175B)](https://openai.com/blog/chatgpt-plus/) - Announced by OpenAI / 2023
- [GPT 3.5 (175B, text-davinci-003)](https://platform.openai.com/docs/models/gpt-3) - Announced by OpenAI / 2022

[:arrow_up: Go to top](#top)

#### Gemini

- [Gemini](https://deepmind.google/technologies/gemini/) - Announced by Google Deepmind / 2023

#### Bard

- [Bard](https://bard.google.com/) - Announced by Google / 2023

[:arrow_up: Go to top](#top)

#### Codex

- [Codex (11B)](https://openai.com/blog/openai-codex/) - Announced by OpenAI / 2021

[:arrow_up: Go to top](#top)

## Datasets

- [Sphere](https://github.com/facebookresearch/Sphere) - Announced by Meta / 2022
   - `134M` documents split into `906M` passages as the web corpus.
- [Common Crawl](https://commoncrawl.org/)
   - `3.15B` pages and over than `380TiB` size dataset, public, free to use.
- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
   - `100,000+` question dataset for QA.
- [Pile](https://pile.eleuther.ai/)
   - `825 GiB diverse`, open source language modelling data set.
- [RACE](https://www.cs.cmu.edu/~glai1/data/race/)
   - A large-scale reading comprehension dataset with more than `28,000` passages and nearly `100,000` questions.
- [Wikipedia](https://huggingface.co/datasets/wikipedia)
   - Wikipedia dataset containing cleaned articles of all languages.

[:arrow_up: Go to top](#top)

## Benchmarks

Below are key websites and references used for evaluating and comparing large language models (LLMs) and their benchmarks:

- **Chatbot Arena**  
  [https://chatbotarena.com/](https://chatbotarena.com/)  
  A platform for head-to-head evaluations of AI chatbots.

- **LLM Leaderboard 2025 – Verified AI Rankings**  
  [https://llm-stats.com/](https://llm-stats.com/)  
  Comparative rankings of leading AI models based on quality, price, and performance.

- **Artificial Analysis LLM Leaderboards**  
  [https://artificialanalysis.ai/leaderboards/models](https://artificialanalysis.ai/leaderboards/models)  
  Detailed comparisons across multiple metrics (output speed, latency, context window, etc.).

- **MMLU – Wikipedia**  
  [https://en.wikipedia.org/wiki/MMLU](https://en.wikipedia.org/wiki/MMLU)  
  Information about the Measuring Massive Multitask Language Understanding benchmark.

- **Language Model Benchmark – Wikipedia**  
  [https://en.wikipedia.org/wiki/Language_model_benchmark](https://en.wikipedia.org/wiki/Language_model_benchmark)  
  Overview of various benchmarks used for evaluating LLM performance.


[:arrow_up: Go to top](#top)

## Materials

### Papers

- [Megatron-Turing NLG (530B)](https://arxiv.org/abs/2201.11990) - Announced by NVIDIA and Microsoft / 2021
- [LaMDA (137B)](https://arxiv.org/abs/2201.08239) - Announced by Google / 2021
- [GLaM (1.2T)](https://arxiv.org/pdf/2112.06905.pdf) - Announced by Google / 2021
- [PaLM (540B)](https://arxiv.org/abs/2204.02311) - Announced by Google / 2022
- [AlphaCode (41.4B)](https://www.deepmind.com/blog/competitive-programming-with-alphacode) - Announced by DeepMind / 2022
- [Chinchilla (70B)](https://arxiv.org/abs/2203.15556) - Announced by DeepMind / 2022
- [Sparrow (70B)](https://www.deepmind.com/blog/building-safer-dialogue-agents) - Announced by DeepMind / 2022
- [NLLB (54.5B)](https://arxiv.org/abs/2207.04672) - Announced by Meta / 2022
- [LLaMA (65B)](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) - Announced by Meta / 2023
- [AlexaTM (20B)](https://arxiv.org/abs/2208.01448) - Announced by Amazon / 2022
- [Gopher (280B)](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval) - Announced by DeepMind / 2021
- [Galactica (120B)](https://arxiv.org/abs/2211.09085) - Announced by Meta / 2022
- [PaLM2 Tech Report](https://ai.google/static/documents/palm2techreport.pdf) - Announced by Google / 2023
- [LIMA](https://arxiv.org/abs/2305.11206) - Announced by Meta / 2023
- [DeekSeek-R1 (631B)](https://arxiv.org/pdf/2501.12948) - Announced by DeepSeek-AI / 2025

### Posts

- [Llama 2 (70B)](https://about.fb.com/news/2023/07/llama-2/) - Announced by Meta / 2023
- [Luminous (13B)](https://www.aleph-alpha.com/luminous-explore-a-model-for-world-class-semantic-representation) - Announced by Aleph Alpha / 2021
- [Turing NLG (17B)](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/) - Announced by Microsoft / 2020
- [Claude (52B)](https://www.anthropic.com/index/introducing-claude) - Announced by Anthropic / 2021
- [Minerva (Parameter size unannounced)](https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html) - Announced by Google / 2022
- [BloombergGPT (50B)](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) - Announced by Bloomberg / 2023
- [AlexaTM (20B](https://www.amazon.science/publications/alexatm-20b-few-shot-learning-using-a-large-scale-multilingual-seq2seq-model) - Announced by Amazon / 2023
- [Dolly (6B)](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html) - Announced by Databricks / 2023
- [Jurassic-1](https://www.ai21.com/blog/announcing-ai21-studio-and-jurassic-1) - Announced by AI21 / 2022
- [Jurassic-2](https://www.ai21.com/blog/introducing-j2) - Announced by AI21 / 2023
- [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/) - Announced by Berkeley Artificial Intelligence Research(BAIR) / 2023
- [Gemma](https://blog.google/technology/developers/gemma-open-models/) - Gemma: Introducing new state-of-the-art open models / 2024
- [Grok-1](https://x.ai/blog/grok-os) - Open Release of Grok-1 / 2023
- [Grok-1.5](https://x.ai/blog/grok-1.5) - Announced by XAI / 2024
- [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) - Announced by Databricks / 2024
- [Grok-2](https://x.ai/blog/grok-3) - Announced by XAI / 2025

[:arrow_up: Go to top](#top)

### Projects

- [BigScience](https://bigscience.huggingface.co/) - Maintained by HuggingFace ([Twitter](https://twitter.com/BigScienceLLM)) ([Notion](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4))
- [HuggingChat](https://www.producthunt.com/posts/hugging-chat) - Maintained by HuggingFace / 2023
- [OpenAssistant](https://open-assistant.io/) - Maintained by Open Assistant / 2023
- [StableLM](https://github.com/Stability-AI/StableLM) - Maintained by Stability AI / 2023
- [Eleuther AI Language Model](https://www.eleuther.ai/language-modeling)- Maintained by Eleuther AI / 2023
- [Falcon LLM](https://falconllm.tii.ae/) - Maintained by Technology Innovation Institute / 2023
- [Gemma](https://ai.google.dev/gemma) - Maintained by Google / 2024

### GitHub repositories

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) - ![Repo stars of tatsu-lab/stanford_alpaca](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=social) - A repository of Stanford Alpaca project,  a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations.
- [Dolly](https://github.com/databrickslabs/dolly) - ![Repo stars of databrickslabs/dolly](https://img.shields.io/github/stars/databrickslabs/dolly?style=social) - A large language model trained on the Databricks Machine Learning Platform.
- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) - ![Repo stars of Significant-Gravitas/Auto-GPT](https://img.shields.io/github/stars/Significant-Gravitas/Auto-GPT?style=social) - An experimental open-source attempt to make GPT-4 fully autonomous.
- [dalai](https://github.com/cocktailpeanut/dalai) - ![Repo stars of cocktailpeanut/dalai](https://img.shields.io/github/stars/cocktailpeanut/dalai?style=social) - The cli tool to run LLaMA on the local machine.
- [LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter) - ![Repo stars of ZrrSkywalker/LLaMA-Adapter](https://img.shields.io/github/stars/ZrrSkywalker/LLaMA-Adapter?style=social) - Fine-tuning LLaMA to follow Instructions within 1 Hour and 1.2M Parameters.
- [alpaca-lora](https://github.com/tloen/alpaca-lora) - ![Repo stars of tloen/alpaca-lora](https://img.shields.io/github/stars/tloen/alpaca-lora?style=social) - Instruct-tune LLaMA on consumer hardware.
- [llama_index](https://github.com/jerryjliu/llama_index) - ![Repo stars of jerryjliu/llama_index](https://img.shields.io/github/stars/jerryjliu/llama_index?style=social) - A project that provides a central interface to connect your LLM's with external data.
- [openai/evals](https://github.com/openai/evals) - ![Repo stars of openai/evals](https://img.shields.io/github/stars/openai/evals?style=social) - A curated list of reinforcement learning with human feedback resources.
- [trlx](https://github.com/CarperAI/trlx) - ![Repo stars of promptslab/Promptify](https://img.shields.io/github/stars/CarperAI/trlx?style=social) - A repo for distributed training of language models with Reinforcement Learning via Human Feedback. (RLHF)
- [pythia](https://github.com/EleutherAI/pythia) - ![Repo stars of EleutherAI/pythia](https://img.shields.io/github/stars/EleutherAI/pythia?style=social) - A suite of 16 LLMs all trained on public data seen in the exact same order and ranging in size from 70M to 12B parameters.
- [Embedchain](https://github.com/embedchain/embedchain) - ![Repo stars of embedchain/embedchain](https://img.shields.io/github/stars/embedchain/embedchain.svg?style=social) - Framework to create ChatGPT like bots over your dataset.
- [google-deepmind/gemma](https://github.com/google-deepmind/gemma) - ![Repo stars of google-deepmind/gemma](https://img.shields.io/github/stars/google-deepmind/gemma.svg?style=social) - Open weights LLM from Google DeepMind.
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) - ![Repo stars of deepseek-ai/DeepSeek-R1](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-R1.svg?style=social) - A first-generation reasoning model from DeepSeek-AI.

[:arrow_up: Go to top](#top)

### HuggingFace repositories

- [OpenAssistant SFT 6](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor) - 30 billion LLaMa-based model made by HuggingFace for the chatting conversation.
- [Vicuna Delta v0](https://huggingface.co/lmsys/vicuna-13b-delta-v0) - An open-source chatbot trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT.
- [MPT 7B](https://huggingface.co/mosaicml/mpt-7b) - A decoder-style transformer pre-trained from scratch on 1T tokens of English text and code. This model was trained by MosaicML.
- [Falcon 7B](https://huggingface.co/tiiuae/falcon-7b) - A 7B parameters causal decoder-only model built by TII and trained on 1,500B tokens of RefinedWeb enhanced with curated corpora.

[:arrow_up: Go to top](#top)

### Reading materials

- [Phi-2: The surprising power of small language models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
- [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama)
- [PaLM2](https://ai.google/discover/palm2)
- [PaLM2 and Future work: Gemini model](https://blog.google/technology/ai/google-palm-2-ai-large-language-model/)

[:arrow_up: Go to top](#top)

## Contributing

We welcome contributions to the Awesome LLMOps list! If you'd like to suggest an addition or make a correction, please follow these guidelines:

1. Fork the repository and create a new branch for your contribution.
2. Make your changes to the README.md file.
3. Ensure that your contribution is relevant to the topic of LLM.
4. Use the following format to add your contribution:
  ```markdown
  [Name of Resource](Link to Resource) - Description of resource
  ```
5. Add your contribution in alphabetical order within its category.
6. Make sure that your contribution is not already listed.
7. Provide a brief description of the resource and explain why it is relevant to LLM.
8. Create a pull request with a clear title and description of your changes.

We appreciate your contributions and thank you for helping to make the Awesome LLM list even more awesome!

[:arrow_up: Go to top](#top)
