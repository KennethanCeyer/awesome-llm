<h1 align="center">Awesome LLM</h1>
<p align="center"><a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome" /></a></p>
<p align="center">Awesome series for Large Language Model(LLM)s</p>

<p align="center"><img width="880" src="./cover.png" /></p>

## Contents

- [Models](#models)
   - [Overview](#overview)
   - [Open models](#open-models)
   - [Projects](#projects)
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

## Models

### Overview

| Name                       | Parameter size    | Announcement date |
|----------------------------|-------------------|-------------------|
| BERT-Large (336M)          | 336 million       | 2018              |
| T5 (11B)                   | 11 billion        | 2020              |
| Gopher (280B)              | 280 billion       | 2021              |
| GPT-J (6B)                 | 6 billion         | 2021              |
| LaMDA (137B)               | 137 billion       | 2021              |
| Megatron-Turing NLG (530B) | 530 billion       | 2021              |
| T0 (11B)                   | 11 billion        | 2021              |
| Macaw (11B)                | 11 billion        | 2021              |
| T5 FLAN (540B)             | 540 billion       | 2022              |
| OPT-175B (175B)            | 175 billion       | 2022              |
| ChatGPT (175B)             | 175 billion       | 2022              |
| GPT 3.5 (175B)             | 175 billion       | 2022              |
| AlexaTM (20B)              | 20 billion        | 2022              |
| Bloom (176B)               | 176 billion       | 2022              |
| Bard                       | Not yet announced | 2023              |
| GPT 4                      | Not yet announced | 2023              |
| AlphaCode (41.4B)          | 41.4 billion      | 2022              |
| Chinchilla (70B)           | 70 billion        | 2022              |
| Sparrow (70B)              | 70 billion        | 2022              |
| PaLM (540B)                | 540 billion       | 2022              |
| NLLB (54.5B)               | 54.5 billion      | 2022              |
| UL2 (20B)                  | 20 billion        | 2022              |
| LLaMA (65B)                | 65 billion        | 2023              |
| Stanford Alpaca (7B)       | 7 billion         | 2023              |
| GPT-NeoX 2.0 (20B)         | 20 billion        | 2023              |
| ChatGPT Plus (175B)        | 175 billion       | 2023              |

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

### Projects

- [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) - Announced by Microsoft / 2023
- [LMOps](https://github.com/microsoft/lmops) - Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities.

### Commercial models

#### GPT

- [GPT 4 (Parameter size unannounced, gpt-4-32k)](https://openai.com/product/gpt-4) - Announced by OpenAI / 2023
- [ChatGPT (175B)](https://openai.com/blog/chatgpt/) - Announced by OpenAI / 2022
- [ChatGPT Plus (175B)](https://openai.com/blog/chatgpt-plus/) - Announced by OpenAI / 2023
- [GPT 3.5 (175B, text-davinci-003)](https://platform.openai.com/docs/models/gpt-3) - Announced by OpenAI / 2022

#### Bard

- [Bard](https://bard.google.com/) - Announced by Google / 2023

#### Codex

- [Codex (11B)](https://openai.com/blog/openai-codex/) - Announced by OpenAI / 2021

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

## Benchmarks

- [BIG-bench](https://github.com/google/BIG-bench)

## Materials

### Papers

- [Megatron-Turing NLG (530B)](https://arxiv.org/abs/2201.11990) - Announced by NVIDIA and Microsoft / 2021
- [LaMDA (137B)](https://arxiv.org/abs/2201.08239) - Announced by Google / 2021
- [PaLM (540B)](https://arxiv.org/abs/2204.02311) - Announced by Google / 2022
- [AlphaCode (41.4B)](https://www.deepmind.com/blog/competitive-programming-with-alphacode) - Announced by DeepMind / 2022
- [Chinchilla (70B)](https://arxiv.org/abs/2203.15556) - Announced by DeepMind / 2022
- [Sparrow (70B)](https://www.deepmind.com/blog/building-safer-dialogue-agents) - Announced by DeepMind / 2022
- [NLLB (54.5B)](https://arxiv.org/abs/2207.04672) - Announced by Meta / 2022
- [LLaMA (65B)](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) - Announced by Meta / 2023
- [AlexaTM (20B)](https://arxiv.org/abs/2208.01448) - Announced by Amazon / 2022
- [Gopher (280B)](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval) - Announced by DeepMind / 2021

### Posts

- [Luminous (13B)](https://www.aleph-alpha.com/luminous-explore-a-model-for-world-class-semantic-representation) - Announced by Aleph Alpha / 2021
- [Turing NLG (17B)](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/) - Announced by Microsoft / 2020
- [Claude (52B)](https://www.anthropic.com/index/introducing-claude) - Announced by Anthropic / 2021
- [Minerva (Parameter size unannounced)](https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html) - Announced by Google / 2022

### Projects

- [BigScience](https://bigscience.huggingface.co/) - Maintained by HuggingFace ([Twitter](https://twitter.com/BigScienceLLM)) ([Notion](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4))
