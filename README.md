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
| Gemma-3                                   | 1B, 4B, 12B, 27B                   | March 2025          | Google                                        |
| GPT-4.5                                   | Undisclosed                        | Feburary 2025       | OpenAI                                        |
| Grok‑3                                    | Undisclosed                        | Feburary 2025       | xAI                                           |
| Gemini-2                                  | Undisclosed                        | Feburary 2025       | Google                                        |
| DeepSeek-VL2                              | 4.5B                               | Feburary 2025       | DeepSeek                                      |
| DeepSeek-R1                               | 671B                               | January 2025        | DeepSeek                                      |
| DeepSeek-V3                               | 671B                               | December 2024       | DeepSeek                                      |
| GPT‑o1                                    | Undisclosed                        | September 2024      | OpenAI                                        |
| Qwen-2.5                                  | 0.5B, 1.5B, 3B, 7B, 14B, 72B       | September 2024      | Alibaba Cloud                                 |
| Gemma-2                                   | 2B, 9B, 27B                        | June 2024           | Google                                        |
| Qwen-2                                    | 0.5B, 1.5B, 7B, 57B, 72B           | June 2024           | Alibaba Cloud                                 |
| GPT‑4o                                    | Undisclosed                        | May 2024            | OpenAI                                        |
| Yi‑1.5                                    | 6B, 9B, 34B                        | May 2024            | 01.AI                                         |
| DeepSeek-V2                               | 238B (21B active)                  | April 2024          | DeepSeek                                      |
| Llama-3                                   | 8B, 70B                            | April 2024          | Meta                                          |
| Gemma-1.1                                 | 2B, 7B                             | April 2024          | Google                                        |
| DeepSeek-VL                               | 7B                                 | March 2024          | DeepSeek                                      |
| Claude-3                                  | Undisclosed                        | March 2024          | Anthropic                                     |
| Grok‑1                                    | 314B                               | March 2024          | xAI                                           |
| DBRX                                      | 132B (36B active)                  | March 2024          | Databricks                                    |
| Gemma                                     | 2B, 7B                             | February 2024       | Google                                        |
| Qwen-1.5                                  | 0.5B, 1.8B, 4B, 7B, 14B, 72B       | February 2024       | Alibaba Cloud                                 |
| Qwen‑VL                                   | Undisclosed                        | January 2024        | Alibaba Cloud                                 |
| Phi‑2                                     | 2.7B                               | December 2023       | Microsoft                                     |
| Gemini                                    | Undisclosed                        | December 2023       | Google                                        |
| Mixtral                                   | 46.7B                              | December 2023       | Mistral AI                                    |
| Grok‑0                                    | 33B                                | November 2023       | xAI                                           |
| Yi                                        | 6B, 34B                            | November 2023       | 01.AI                                         |
| Zephyr‑7b‑beta                            | 7B                                 | October 2023        | HuggingFace H4                                |
| Solar                                     | 10.7B                              | September 2023      | Upstage                                       |
| Mistral                                   | 7.3B                               | September 2023      | Mistral AI                                    |
| Qwen                                      | 1.8B, 7B, 14B, 72B                 | August 2023         | Alibaba Cloud                                 |
| Llama-2                                   | 7B, 13B, 70B                       | July 2023           | Meta                                          |
| XGen                                      | 7B                                 | July 2023           | Salesforce                                    |
| Falcon                                    | 7B, 40B, 180B                      | June/Sept 2023      | Technology Innovation Institute (UAE)         |
| MPT                                       | 7B, 30B                            | May/June 2023       | MosaicML                                      |
| LIMA                                      | 65B                                | May 2023            | Meta AI                                       |
| PaLM-2                                    | 340B                               | May 2023            | Google                                        |
| Vicuna                                    | 7B, 13B, 33B                       | March 2023          | LMSYS ORG                                     |
| Koala                                     | 13B                                | April 2023          | UC Berkeley                                   |
| OpenAssistant                             | 30B                                | April 2023          | LAION                                         |
| Jurassic‑2                                | Undisclosed                        | April 2023          | AI21 Labs                                     |
| Dolly                                     | 6B, 12B                            | March/April 2023    | Databricks                                    |
| BloombergGPT                              | 50B                                | March 2023          | Bloomberg                                     |
| GPT‑4                                     | Undisclosed                        | March 2023          | OpenAI                                        |
| Bard                                      | Undisclosed                        | March 2023          | Google                                        |
| Stanford-Alpaca                           | 7B                                 | March 2023          | Stanford University                           |
| LLaMA                                     | 7B, 13B, 33B, 65B                  | February 2023       | Meta                                          |
| ChatGPT                                   | Undisclosed                        | November 2022       | OpenAI                                        |
| GPT‑3.5                                   | 175B                               | November 2022       | OpenAI                                        |
| Jurassic‑1                                | 178B                               | November 2022       | AI21                                          |
| Galactica                                 | 120B                               | November 2022       | Meta                                          |
| Sparrow                                   | 70B                                | September 2022      | DeepMind                                      |
| NLLB                                      | 54.5B                              | July 2022           | Meta                                          |
| BLOOM                                     | 176B                               | July 2022           | BigScience (Hugging Face)                     |
| AlexaTM                                   | 20B                                | August 2022         | Amazon                                        |
| UL2                                       | 20B                                | May 2022            | Google                                        |
| OPT                                       | 175B                               | May 2022            | Meta (Facebook)                               |
| PaLM                                      | 540B                               | April 2022          | Google                                        |
| AlphaCode                                 | 41.4B                              | February 2022       | DeepMind                                      |
| Chinchilla                                | 70B                                | March 2022          | DeepMind                                      |
| GLaM                                      | 1.2T                               | December 2021       | Google                                        |
| Macaw                                     | 11B                                | October 2021        | Allen Institute for AI                        |
| T0                                        | 11B                                | October 2021        | Hugging Face                                  |
| Megatron‑Turing-NLG                       | 530B                               | January 2022        | Microsoft & NVIDIA                            |
| LaMDA                                     | 137B                               | January 2022        | Google                                        |
| Gopher                                    | 280B                               | December 2021       | DeepMind                                      |
| GPT‑J                                     | 6B                                 | June 2021           | EleutherAI                                    |
| GPT‑NeoX-2.0                              | 20B                                | February 2022       | EleutherAI                                    |
| T5                                        | 60M, 220M, 770M, 3B, 11B           | October 2019        | Google                                        |
| BERT                                      | 108M, 334M, 1.27B                  | October 2018        | Google                                        |


[:arrow_up: Go to top](#top)

### Open models

- [Gemma 3 (1B, 4B, 12B, 27B)](https://huggingface.co/google/gemma-3-1b-it) - Announced by DeepSeek / 2025
- [DeepSeek-R1 (671B)](https://github.com/deepseek-ai/DeepSeek-R1) - Announced by DeepSeek / 2025
- [LLaMA 3 (8B, 70B)](https://huggingface.co/meta-llama/Llama-3) - Announced by Meta / 2024
- [Gemma 2 (2B, 9B, 27B)](https://ai.google.dev/gemma) - Announced by Google / 2024
- [DeepSeek-V2 (238B)](https://huggingface.co/deepseek-ai/deepseek-v2) - Announced by DeepSeek / 2024
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B) - Announced by Mistral AI / 2023
- [Solar (10.7B)](https://huggingface.co/upstage/solar-10.7B) - Announced by Upstage / 2023
- [DBRX (132B)](https://huggingface.co/databricks/dbrx) - Announced by Databricks / 2024
- [Falcon (7B, 40B, 180B)](https://huggingface.co/tiiuae/falcon-7b) - Announced by Technology Innovation Institute / 2023
- [MPT (7B, 30B)](https://huggingface.co/mosaicml/mpt-7b) - Announced by MosaicML / 2023
- [Dolly (6B, 12B)](https://huggingface.co/databricks/dolly-v2-12b) - Announced by Databricks / 2023
- [Phi-2 (2.7B)](https://huggingface.co/microsoft/Phi-2) - Announced by Microsoft / 2023
- [GPT-NEOX 20B](https://huggingface.co/EleutherAI/gpt-neox-20b) - Announced by EleutherAI / 2023
- [GPT-J (6B)](https://huggingface.co/EleutherAI/gpt-j-6B) - Announced by EleutherAI / 2021
- [Stanford Alpaca (7B)](https://crfm.stanford.edu/2023/03/13/alpaca.html) - Announced by Stanford University / 2023
- [OpenAssistant (30B)](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor) - Announced by LAION / 2023

[:arrow_up: Go to top](#top)

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

- **Price Per Token – LLM Pricing Comparison**
  [https://pricepertoken.com/](https://pricepertoken.com/)
  Compare LLM API pricing across 200+ models with token counters, cost calculators, and benchmark data.

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

- [Introducing Gemma 3: The most capable model you can run on a single GPU or TPU](https://blog.google/technology/developers/gemma-3/)
- [Phi-2: The surprising power of small language models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
- [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama)
- [PaLM2](https://ai.google/discover/palm2)
- [PaLM2 and Future work: Gemini model](https://blog.google/technology/ai/google-palm-2-ai-large-language-model/)

[:arrow_up: Go to top](#top)

## Contributing

We welcome contributions to the [Awesome LLM](https://github.com/KennethanCeyer/awesome-llm/) list! If you'd like to suggest an addition or make a correction, please follow these guidelines:

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
