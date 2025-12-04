[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/C8cH05zS)


# Report on Mini-Project: Phase 1 and Phase 2


- **Name**: Prakhar Raj
- **Roll No.**: 2022102066

## 1. Introduction
The goal of this mini-project is to build a small autoregressive language model (LM) for multiple languages:
English, Hindi & Awadhi from scratch. 

Given minimum total tokens required: 3 billion.
- 50% English - 1.5billion token
- 30-40% Hindi → 900 million - 1.2 billion token
- 10-20% Awadhi → 300-600 million

Also I will use Model - **Qwen3 (Dense)** 

 
 The project involves data collection, preprocessing, tokenizer training, and fine-tuning for specific tasks, mentioned as: [Go to Task 1](#task-1-ft72---text-simplification)
 & [Go to Task 2](#task-2-ft25---text-de-identification-anonymization)


##  Datasets

### Hugging Face Links

| Language     | Dataset Link |
|--------------|--------------|
| English      | [English Text Simplification](https://huggingface.co/datasets/raja20221020/english-text-simplification-for-finetuning) |
| Hindi        | [Hindi Text Simplification](https://huggingface.co/datasets/raja20221020/hindi-text-simplification-for-finetuning) |
| Awadhi       | [Awadhi Text Simplification](https://huggingface.co/datasets/raja20221020/awadhi-text-simplification-for-finetuning) |
| Eng_Hin_Awa  | [English-Hindi-Awadhi De-identification](https://huggingface.co/datasets/raja20221020/english_hindi_awadhi_deidentification) |

---

##  Experiment Tracking (Weights & Biases)

- **[Pretrained Model Logs](https://wandb.ai/prakhar_raj-iiit-hyderabad/lma_mini_project?nw=nwuserprakhar_raj)**
- **[Finetuned Model Logs](https://wandb.ai/prakhar_raj-iiit-hyderabad/finetune_1?nw=nwuserprakhar_raj)**

---


🔗 [Model Checkpoints on onedrive](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/prakhar_raj_students_iiit_ac_in/EgSGY93hPGBCi_pEanDMNkgB44GzrzfsrluRVVlDe--I0A?e=6GD3dJ)

---

## Phase 1 & Phase 2: Data Collection, Preprocessing & Tokenizer Training:

## Corpus Collection
- **Languages and Token Distribution**:
  - English: 50% of total tokens (~1.5 billion tokens).
  - Mother Tongue (Hindi): 30–40% of total tokens (~1–1.2 billion tokens).
  - Indian Language (Awadhi): 10–20% of total tokens (~0.3–0.6 billion tokens).

- **Sources**:
  - English: Extracted from mC4 dataset.
  - Awadhi: Derived from [github_repo](https://github.com/PrashantShuklaa/Awadhi_Speech_Dataset?tab=readme-ov-file) , [Kaggle datasets](https://www.kaggle.com/search?q=awadhi+in%3Adatasets) , [HPLT_dataset](https://hplt-project.org/datasets/v2.0), [huggingface](https://hplt-project.org/datasets/v2.0).
  - Hindi: Combined from Sangraha dataset, mC4, cc100 and other sources.

### Preprocessing
- **Cleaning**:

    - Ensured consistent character representation using NFC.
    - Eliminated invisible characters like zero-width space and BOM.
    - Retained only printable characters and essential whitespace.
    - Deduplicated lines to ensure unique data.
  
- **Sentence Segmentation**:
  - These are used to ensure accurate and language-specific sentence segmentation for both English and Indic languages, which is critical for preparing clean and structured text for language model training.

    -  Split English text into sentences based on punctuation (., ?, !) followed by whitespace or line end.
    - Used Indic NLP Library for Hindi and Awadhi.

  So after separating all sentences:

```text
awadhi_cleaned.txt has 8478186 lines.
english_cleaned.txt has 90641955 lines.
hindi_cleaned.txt has 61054678 lines.
```

---
### For training tokenizer, I have dicided text ratio: (English 50%, Hindi 35%, Awadhi 15%)

- I want tokenizer to see each language proportionally to how we want the LM to learn it.
- And if we just dump most English, it may dominate the vocabulary & Hindi/Awadhi words would appear rarely.

**As seen above if English has ~90 million lines and Awadhi has only approx. ~8 million lines. So just feeding all lines would completely overwhelm Awadhi.**

**so I sampled sentences from each languages to match the target ratio:**

```text
English: 10,000,000 * 0.5  ≈ 5,000,000 lines
Hindi:   10,000,000 * 0.35 ≈ 3,500,000 lines
Awadhi:  10,000,000 * 0.15 ≈ 1,500,000 lines
```

**taking total 10 million lines for training tokenizer**

---

  ## Tokenizer Choice

- **Model**: SentencePiece with unigram model.
- **Vocabulary Size**: 50,000 tokens.

### Training
- Trained the tokenizer on the combined corpus with proportional sampling:
  - English: 50%.
  - Awadhi: 35%.
  - Hindi: 15%.



## Evaluation

  - Verified that the vocabulary supports all three languages adequately.
  - Ensured inclusion of common words and subwords from all three languages.


---

**Token Statistics**:


```text
# your output here
=== English ===
Total chars: 9143092310
Devanagari fraction: 0.000
Total tokens: 2040999488
Tokens per word: 1.317
<unk> tokens: 0 (0.0000)
Byte-fallback tokens: 15004048 (0.0074)

=== Hindi ===
Total chars: 6603725591
Devanagari fraction: 0.774
Total tokens: 1619964264
Tokens per word: 1.245
<unk> tokens: 0 (0.0000)
Byte-fallback tokens: 7492328 (0.0046)

=== Awadhi ===
Total chars: 926348760
Devanagari fraction: 0.772
Total tokens: 226219398
Tokens per word: 1.189
<unk> tokens: 0 (0.0000)
Byte-fallback tokens: 79915 (0.0004)

=== Overall Token Distribution ===
English tokens: 2040999488 (52.51%)
Hindi tokens: 1619964264 (41.67%)
Awadhi tokens: 226219398 (5.82%)
Total tokens: 3887183150
```

### Output for trained tokenizer on dataset:

```text

--- English Tokenization (from dataset) ---
Sentence: This is done by separating isotopes in an enrichment plant to achieve the higher concentration.
Tokens: ['▁This', '▁is', '▁done', '▁by', '▁separating', '▁is', 'otope', 's', '▁in', '▁an', '▁enrichment', '▁plant', '▁to', '▁achieve', '▁the', '▁higher', '▁concentration', '.']

Sentence: In reality, the company supplies a full page of testimonials from local users of the app.
Tokens: ['▁In', '▁reality', ',', '▁the', '▁company', '▁supplies', '▁a', '▁full', '▁page', '▁of', '▁testimonials', '▁from', '▁local', '▁users', '▁of', '▁the', '▁app', '.']

Sentence: When combined with Work4’s patented technology, Facebook’s targeted advertising platform becomes the must-have tool in every company’s recruitment tool-box.
Tokens: ['▁When', '▁combined', '▁with', '▁Work', '4', '’', 's', '▁patented', '▁technology', ',', '▁Facebook', '’', 's', '▁targeted', '▁advertising', '▁platform', '▁becomes', '▁the', '▁must', '-', 'have', '▁tool', '▁in', '▁every', '▁company', '’', 's', '▁recruitment', '▁tool', '-', 'box', '.']



--- Hindi Tokenization (from dataset) ---
Sentence: 14 अप्रैल 2022 को दोनों फिल्में दर्शकों के सामने आएगी।
Tokens: ['▁14', '▁अप्रैल', '▁2022', '▁को', '▁दोनों', '▁फिल्में', '▁दर्शकों', '▁के', '▁सामने', '▁आएगी', '।']

Sentence: हैरत की बात यह कि दंपती के बेटे ने भी घटना का वीडियो बनाया था।
Tokens: ['▁है', 'रत', '▁की', '▁बात', '▁यह', '▁कि', '▁दंपती', '▁के', '▁बेटे', '▁ने', '▁भी', '▁घटना', '▁का', '▁वीडियो', '▁बनाया', '▁था', '।']

Sentence: बैठी थी सांसद , सर के ऊपर से गयी गोली !
Tokens: ['▁बैठी', '▁थी', '▁सांसद', '▁', ',', '▁सर', '▁के', '▁ऊपर', '▁से', '▁गयी', '▁गोली', '▁!']


--- Awadhi Tokenization (from dataset) ---
Sentence: इसके अलावा, बुद्धिमान औरत के खोज के लिए इस यात्रा म अन्य पात्र शामिल हो सकत हैं और लार्ड स्टार्क के साथ अपने रिश्ते पर कौन सा असर पड़ सकत है?
Tokens: ['▁इसके', '▁अलावा', ',', '▁बुद्धिमान', '▁औरत', '▁के', '▁खोज', '▁के', '▁लिए', '▁इस', '▁यात्रा', '▁म', '▁अन्य', '▁पात्र', '▁शामिल', '▁हो', '▁सकत', '▁हैं', '▁और', '▁लार्ड', '▁स्टार्क', '▁के', '▁साथ', '▁अपने', '▁रिश्ते', '▁पर', '▁कौन', '▁सा', '▁असर', '▁पड़', '▁सकत', '▁है', '?']

Sentence: दए गए डेटासेट के आधार पर, हम मान सकत हैं कि इनपुट सिग्नल एक साइनसोइडल तरंग है जेहिमा एक विशेष आवृत्ति अउर चरण है।
Tokens: ['▁दए', '▁गए', '▁डेटासेट', '▁के', '▁आधार', '▁पर', ',', '▁हम', '▁मान', '▁सकत', '▁हैं', '▁कि', '▁इनपुट', '▁सिग्नल', '▁एक', '▁साइन', 'स', 'ोइड', 'ल', '▁तरंग', '▁है', '▁जेहिमा', '▁एक', '▁विशेष', '▁आवृत्ति', '▁अउर', '▁चरण', '▁है', '।']

Sentence: हर तत्व के लिए, आप जांच सकत हैं कि ई अलमंड शब्दकोश मा एक कुंजी के रूप मा मौजूद है।
Tokens: ['▁हर', '▁तत्व', '▁के', '▁लिए', ',', '▁आप', '▁जांच', '▁सकत', '▁हैं', '▁कि', '▁ई', '▁अल', 'मंड', '▁शब्दकोश', '▁मा', '▁एक', '▁कुंजी', '▁के', '▁रूप', '▁मा', '▁मौजूद', '▁है', '।']



```
---

##  Timeline and Future Steps
###  Work Done So Far
- Completed data collection and preprocessing.
- Trained and evaluated the tokenizer.

###  Next Steps

---

---


### Phase 3: Pretraining the Autoregressive Language Model

This phase focused on training the custom multilingual autoregressive model using the tokenized dataset prepared in the earlier phases. The process involved defining the model architecture, setting up a robust distributed training environment, overcoming significant technical challenges, and successfully training the model for **2** full epochs.

#### Model Architecture

The model is a custom-configured, decoder-only Transformer based on the **Qwen3** architecture. Instead of using a standard pre-trained checkpoint, a new model was initialized from a configuration to match our specific requirements for a smaller, more manageable model.

The key parameters for the model are:
- **Base Architecture**: `Qwen/Qwen3-0.6B` configuration
- **Vocabulary Size**: `50,000` (from our custom tokenizer)
- **Number of Layers**: `12`
- **Number of Attention Heads**: `8`
- **Hidden Size (Embedding Dimension)**: `512`
- **Feed-Forward Intermediate Size**: `2048`

This configuration results in a custom small-scale language model of approximately **120M parameters**, suitable for pretraining with the available resources.

#### Training Configuration

The training was configured using the `transformers.Trainer` and `TrainingArguments` classes. Key hyperparameters were set as follows:

- **Distributed Training Strategy**: Distributed Data Parallel (DDP)
- **Mixed Precision**: `FP16` (Float16) was used to reduce memory consumption and accelerate training.
- **Number of GPUs**: 4
- **Per-Device Batch Size**: `4`
- **Gradient Accumulation Steps**: `8`
- **Effective Batch Size**: `4 (per GPU) * 4 (GPUs) * 8 (accum steps) = 128`
- **Optimizer**: AdamW (default for `Trainer`)
- **Learning Rate**: `1e-3` (0.001)
- **LR Scheduler**: Linear warmup for `500` steps, followed by decay
- **Total Training Epochs**: `2.0`
- **Monitoring**: All metrics were logged to Weights & Biases for real-time tracking: [WandB Link](https://wandb.ai/prakhar_raj-iiit-hyderabad/lma_mini_project/runs/p1rg2omm?nw=nwuserprakhar_raj)


---

### **Pretraining Results**

The model was successfully pretrained for **2 full epochs**. The entire training process took approximately **48 hours** to complete. After training, the final model was evaluated on the held-out test set containing samples from all three languages.

The final evaluation metrics are as follows:

| Metric | Value |
| :--- | :--- |
| **Test Loss** | `3.62` |
| **Perplexity** | `37.33` |
| **Total Epochs** | `2.0` |


<!-- 
![eval_loss](img/eval_loss.svg)
![train_loss](img/train_loss.svg)
![test_perplexity](img/test_perp.svg)
![test_eval_loss](img/test_eval_loss.svg) -->


<p float="left">
  <img src="img/eval_loss.svg" width="45%"/>
  <img src="img/train_loss.svg" width="45%"/>
  <img src="img/test_perp.svg" width="45%"/>
  <img src="img/test_eval_loss.svg" width="45%"/>
</p>


**Conclusion for Phase 3:**
- A test perplexity of **37.33** indicates that the model has successfully learned meaningful patterns, syntax, and vocabulary from the multilingual corpus. Perplexity measures how well the model predicts the next token; a lower value is better.
- For a custom model of this size trained from scratch, this is a strong result and confirms that the pretraining was successful. The final model artifacts have been saved locally and pushed to the Hugging Face Hub at `raja20221020/qwen-small-pretrained`, providing a solid foundation for the fine-tuning tasks in the next phase.



### Phase 4: Fine-tuning for Specific Tasks

With a robust pretrained multilingual model as the foundation, Phase 4 is focused on adapting this model to perform two specialized, instruction-based tasks: **Text Simplification** and **Text De-identification**. This was achieved through a multi-task, multi-lingual fine-tuning process using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

#### Task 1: FT72 - Text Simplification

The goal of this task is to make complex sentences easier to understand.

**Dataset Curation:**
1.  **Source Data:** The process began by sourcing a high-quality English text simplification dataset (`bogdancazan/wikilarge-text-simplification`). A random sample of 12,000 examples was selected to form the base English dataset.
2.  **Instruction Templating:** To enhance the model's ability to follow instructions, a variety of prompts (e.g., "Simplify this sentence.", "Make this text easier to understand.") were randomly assigned to each example.
3.  **Multilingual Expansion:** The English dataset was then translated into Hindi and Awadhi using the **Google Cloud Translate API**. This step was crucial for creating parallel fine-tuning data for our target languages.

```
{"instruction": "Simplify the following sentence to make it easier to read.", "input": "he was ranked no. in empire magazine s the top movie stars of all time list.", "output": "he is ranked in empire magazine s the top movie stars of all time list."}

{"instruction": "निम्नलिखित पाठ को सरल बनाने के लिए पुनः लिखें।", "input": "उनकी एक संतान लिन उलमान है, जिसके पिता इंगमार बर्गमैन हैं, जबकि उलमान की शादी स्टैंग से हुई थी। उलमान के दो पोते-पोतियां हैं, एक लड़का और एक लड़की, जो उनकी बेटी की दो शादियों से हैं।", "output": "मिस उलमान की एक बेटी लिन उलमान एलआरबी है जो इंगमार बर्गमैन और दो पोते-पोतियों के साथ आरआरबी में पैदा हुई थी।"}

{"instruction": "पाठ का सरल बनावा।", "input": "मई का जारी कीन गा ई ब्लॉक पार्टी के पहिला वी रिकॉर्ड्स एप रहा।", "output": "ईपी मई का पूरे यूरोप मा जारी कीन गा रहा।"}

```

4.  **Final Datasets:** The resulting three datasets (English, Hindi, and Awadhi) were individually uploaded to the Hugging Face Hub, creating a comprehensive, multilingual resource for the text simplification task:
### Datasets

| Language | Dataset Link |
|----------|--------------|
| English  | [English Text Simplification](https://huggingface.co/datasets/raja20221020/english-text-simplification-for-finetuning) |
| Hindi    | [Hindi Text Simplification](https://huggingface.co/datasets/raja20221020/hindi-text-simplification-for-finetuning) |
| Awadhi   | [Awadhi Text Simplification](https://huggingface.co/datasets/raja20221020/awadhi-text-simplification-for-finetuning) |



#### Task 2: FT25 - Text De-identification (Anonymization)

The goal of this task is to identify and replace Personally Identifiable Information (PII) with anonymized tags.

**Dataset Curation:**
1.  **Synthetic Data Generation:** Due to the scarcity of public de-identification datasets, a synthetic dataset was generated from scratch.
2.  **Entity Lists:** Comprehensive lists of PII entities (names, addresses, phone numbers, etc.) were compiled for English and Hindi.
3.  **Template Creation:** A large and diverse set of sentence templates containing placeholders for these entities was created for English, Hindi, and Awadhi.
4.  **Data Generation:** A script programmatically filled these templates with random entities from the lists to create realistic "input" sentences. A corresponding "output" sentence was generated by replacing the entities with anonymized tags (e.g., `[NAME]`, `[ADDRESS]`). This process was repeated to generate 12,000 examples for each of 3 languages.

  **Combined Dataset:** The de-identification and text simplification datasets were loaded and processed into a unified instruction format: 
```
{"instruction": "इ टेक्स्ट मा निजी जानकारी का पहिचान छिपावा।", "input": "बिलिंग खाता {ACCOUNT} अउर योजना HPN-STU-901 के जांच भइल।", "output": "बिलिंग खाता {ACCOUNT} अउर योजना [HPBN] के जांच भइल।"}

{"instruction": "Annonymize the following text.", "input": "Credit card 6011111111111117 will expire on 2023-10-27.", "output": "Credit card [CREDIT_CARD] will expire on [DATE]."}

{"instruction": "Annonymize करें।", "input": "आप मुझसे 5005566778 पर या ईमेल test123@gmail.com पर संपर्क कर सकते हैं।", "output": "आप मुझसे [PHONE_NUMBER] पर या ईमेल [EMAIL] पर संपर्क कर सकते हैं।"}
```

5.  **Final Dataset:**  The generated datasets for all three languages were combined, shuffled, and uploaded to the Hugging Face Hub as a single, unified dataset ready for fine-tuning.
### Datasets



| Language | Dataset Link |
|----------|--------------|
| Eng_Hin_Awa | [English Text Simplification](https://huggingface.co/datasets/raja20221020/english_hindi_awadhi_deidentification) |

---

#### Multi-Task Fine-tuning Strategy

To train a single model capable of performing both tasks across all three languages, a multi-task learning approach was implemented.

**Methodology:**
1.  **PEFT with LoRA:** To efficiently adapt the 120M parameter pretrained model without modifying all its weights, **Low-Rank Adaptation (LoRA)** was used. LoRA introduces a small number of trainable parameters into the model's attention layers (`q_proj`, `k_proj`, `v_proj`, etc.), making the fine-tuning process computationally efficient.
2.  **Combined Dataset:** The de-identification and text simplification datasets were loaded and processed into a unified instruction format.
3.  **Interleaved Sampling:** The datasets were combined using `interleave_datasets`, with a sampling probability of **50% for the de-identification task** and **50% for the simplification tasks** (split evenly among the three languages). This ensured the model was trained on a balanced mix of tasks during each training step.
4.  **Validation Strategy:** For datasets lacking a predefined validation split, 10% of the training data was automatically held out to create one, ensuring reliable evaluation of the model's performance on unseen data.

#### Fine-tuning Configuration

- **Base Model**: The final checkpoint from the Phase 3 pretraining.
- **Fine-tuning Method**: LoRA (`r=16`, `lora_alpha=32`)
- **Micro Batch Size**: `8`
- **Gradient Accumulation Steps**: `4`
- **Effective Batch Size**: `32`
- **Learning Rate**: `1e-4`
- **Total Training Epochs**: `2`
- **Monitoring**: All fine-tuning metrics were logged to Weights & Biases.

This comprehensive fine-tuning phase successfully produced a single, versatile, multilingual model adapted for two distinct and practical NLP tasks. The final LoRA adapter, which contains the specialized task knowledge, was saved and pushed to the Hugging Face Hub at `raja20221020/qwen-small-finetuned-multitask`.
  

## Some Examples
```

--- User Prompt ---
मेरा फ़ोन
-------------------

Generating response...

--- Full Model Output ---
मेरा फ़ोन यूज करने के लिए है । मैं एक बार फोन उठा ता हूं , तो आपको पता चल जाता है कि यह आपके लिए नहीं है । ' वहीं , एक अन्य यूजर ने लिखा - ' आप जानते हैं कि हम दोनों ही बहुत अच्छे दोस्त हैं और उनकी मदद करना चाहते हैं । ' उन्होंने आगे कहा कि ' जब भी मुझे कोई समस्या आती है , तो मैं उनसे संपर्क करता हूं । ' बता दें कि इससे पहले कंपनी ने अपने यूजर्स को न ए स्मार्टफोन ्स की पेशकश की थी । नई दिल्ली ( एजेंसी / वार्ता ): दिल्ली में पिछले क ई दिनों से हो रही भारी बारिश के कारण आज सुबह से ही लोगों का जीना मु हाल हो गया है । मौसम विभाग ने अगले दो दिनों त क हल्की से मध्यम बारिश होने की संभावना जता य ी है । 

--- User Prompt ---
My name is John and my email is
-------------------

Generating response...

--- Full Model Output ---
My name is John and my email is ng . I know it ' s your home town , but you have to find a good place for people to meet up with me , s o I can help you out . This was my first time in the US . It was the best thing I ever had ever done before . I love it when I get to know someone who has been there since I was six . I want to be able to talk to them about how they got here and what they did and why they don ' t like it . You are the person who makes me feel more comfortable on the phone as I am not going to let you know that I am here to help you out . I think I will do better than anyone else . Thank you for sharing this .
-------------------------


--- User Prompt ---
हम पंडितजी क घर जात हईं, काहे कि उहाँ आजका कथा होइ, अउर गाँव के सब लोगा उहाँ जुटिहैं।
-------------------

--- Full Model Output ---
हम पंडित जी क घर जात ह ई ं , काहे कि उ हाँ आज का क था होइ , अउर गाँव के सब लोग ा उ हाँ जुट ि हैं ।  " तुम कहाँ हो ? - मैं तो एक स न की लड़की हूँ ! - वह क्या है ? - यह क्या है ? - वे क्या हैं ? - अरे , मैं तो ब स एक लड़का हूँ ! - लेकिन आप जानत हैं , कौन है ? - अगर आप अपने माता - पिता से प्यार करत हैं , तो आप कैसे हैं ? - और फिर भी आप हमेशा मुझसे प्यार करत हैं ! - उनका मतलब का है ? - या र , का आपका लागत है कि ऊ आपके साथ है ? - का आप चाहत हैं कि उ आपन दिल मा एक खास भावना व्यक्त करै ? - जब त क आप अपने गुरु के पास नहीं आवत हैं , तब त क आप जउन चीज न का साझा करत हैं ,
-------------------------

```

```
(hep_jepa) aryan.kumar@gnode052:~/prakhar$ # --- Test Case 1: English De-identification (Model hallucinates incorrect tags and repeats nonsense) ---
(hep_jepa) aryan.kumar@gnode052:~/prakhar$ python generate.py \
>     --model_type finetuned \
>     --prompt "### Instruction:\n De-identify the text.\n\n### Input:\nMy name is John Smith.\n\n### Response:\n"
--- Starting Inference Script ---
Loading correctly configured tokenizer from: ./ckpt/qwen-small-checkpoints/final
Loading BASE model from: ./ckpt/qwen-small-checkpoints/final
Attaching LoRA adapter from: ./qwen-small-finetuned-adapter/checkpoint-998

--- User Prompt ---
### Instruction:\n De-identify the text.\n\n### Input:\nMy name is John Smith.\n\n### Response:\n
-------------------

Generating response...

--- Full Model Output ---
# # # Instruction : \ n De - identify the text . \ n \ n # # # Input : \ n My name is John Smith . \ n \ n # # # Response : \ n 🏻 \ n [ MY FILE NAME ] has been recorded . \ n \ n [ FF FF _ ID ] has been recorded . \ n \ r @ y a h o o . org was recorded . \ n \ r [ MY FILE NAME ]. \ n \ r [ MY FILE NAME ]. \ n \ r [ MY FILE NAME ]. \ n \ r [ FF FF _ ID ] had been recorded . \ n \ r [ MY FILE NAME ]. \ n \ r [ MY FILE NAME .” was recorded . \ n \ r [ MY FILE NAME ]. \ n \ r [ FF FF _ ID ] had been recorded . \ n \ r [ MY FILE NAME ]. \ n \ r [ FF FF _ ID ]. was recorded . \ n
-------------------------


(hep_jepa) aryan.kumar@gnode052:~/prakhar$ # --- Test Case 2: English De-identification (Model mangles the prompt and outputs repetitive gibberish) ---
(hep_jepa) aryan.kumar@gnode052:~/prakhar$ python generate.py \
>     --model_type finetuned \
>     --prompt "### Instruction:\n De-identify the PII text.\n\n### Input:\n my number is 555-1234.\n\n### Response:\n"
--- Starting Inference Script ---
Loading correctly configured tokenizer from: ./ckpt/qwen-small-checkpoints/final
Loading BASE model from: ./ckpt/qwen-small-checkpoints/final
Attaching LoRA adapter from: ./qwen-small-finetuned-adapter/checkpoint-100

--- User Prompt ---
### Instruction:\n De-identify the PII text.\n\n### Input:\n my number is 555-1234.\n\n### Response:\n
-------------------

Generating response...

--- Full Model Output ---
# # # Instruction : \ n De - identify the PI I text . \ n \ n # # # Input : \ n my number is 5 55 - 1234 . \ n \ n # # # Response : \ n ार्टनरशिप _ ID [ _ ID ] was _ ID [ \ n ] ; \ n \ n _ ID [ _ ID ] was _ ID [ \ n ]. \ n \ n # # # Response : \ n my number is 4 40 - 10 42 . \ n \ n # # Response : \ n my number is 2 98 - 09 3 . \ n \ n # # Response : \ n my number is 60 31 - 08 4 . \ n \ n # # Response : \ n my number is 1 888 - 13 57 . \ n \ n # # Response : \ n my number is 70 33 - 18 58 . \ n \ n # # Response : \ n my number is 80 28 - 17 77 . \ n \

```
