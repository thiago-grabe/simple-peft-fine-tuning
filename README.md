# Lightweight Fine-Tuning Project

This project focuses on fine-tuning foundation models with minimal computational resources. By leveraging lightweight fine-tuning techniques, we can adapt large-scale models like GPT-2 efficiently for specific tasks without extensive GPU requirements.

## Dataset Overview

We use the **Hate Speech Twitter** dataset ([view here](https://huggingface.co/datasets/thefrankhsu/hate_speech_twitter/viewer)) to train and evaluate our model. The dataset consists of tweets labeled as hate speech or non-hate speech, with further classification into nine categories.

### Key Features
- **Tweets**: The textual content of tweets.
- **Labels**: Binary values (1 = Hate Speech, 0 = Non-Hate Speech).
- **Categories**: Hate speech is classified into nine distinct groups, including behavior, class, disability, ethnicity, gender, physical appearance, race, religion, and sexual orientation.

### Dataset Statistics
- **Training Set**:
  - 5,679 tweets (1,516 hate speech, 4,163 non-hate speech)
- **Testing Set**:
  - 1,000 tweets (500 hate speech, 500 non-hate speech)

## Fine-Tuning Approach

### Model Choice
We use **GPT-2** as the base model and fine-tune it for binary classification using the PEFT (Parameter Efficient Fine-Tuning) method **LoRA (Low-Rank Adaptation)**. 

### Preprocessing Steps
1. **Tokenizer Initialization**:
   ```python
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   tokenizer.pad_token = tokenizer.eos_token
   ```
2. **Tokenization & Encoding**:
   ```python
   def tokenize_and_encode(examples):
       tokenized_inputs = tokenizer(
           examples['tweet'],
           padding="max_length",
           truncation=True,
           max_length=512
       )
       tokenized_inputs['labels'] = examples['label']
       return tokenized_inputs
   ```
3. **Dataset Preparation**:
   ```python
   train_dataset = train_dataset.map(tokenize_and_encode, batched=True)
   val_dataset = val_dataset.map(tokenize_and_encode, batched=True)
   ```

### Training Configuration

The model is fine-tuned using **Hugging Face Trainer** with the following configurations:

```python
training_args = TrainingArguments(
    output_dir="./results_normal_model",
    evaluation_strategy="epoch",
    learning_rate=2.5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs_normal_model',
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=150,
    warmup_ratio=0.1,
    eval_accumulation_steps=50
)
```

### Model Evaluation

After training, we evaluate the model's performance using accuracy, precision, recall, and F1-score:

```python
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
```

## Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Datasets
- CUDA (if using GPU)

### Installation Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/lightweight-finetuning.git
   cd lightweight-finetuning
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Run the fine-tuning notebook

## Results & Performance

The fine-tuned model achieves the following performance on the validation set:
- **Accuracy**: ~86.5%
- **F1-score**: ~0.85
- **Precision**: ~0.87
- **Recall**: ~0.84

## Example Classifications

Here are some example classifications from the model:

- **Tweet**: "I can't believe people still think like this in 2024. So ignorant!"
  - **Predicted Label**: Non-Hate Speech

- **Tweet**: "This group of people should not be allowed to live in this country."
  - **Predicted Label**: Hate Speech

- **Tweet**: "I love how diverse and inclusive this community is!"
  - **Predicted Label**: Non-Hate Speech

- **Tweet**: "Go back to where you came from!"
  - **Predicted Label**: Hate Speech

## Future Improvements
- Experiment with **longer training epochs** to improve performance.
- Test different **PEFT techniques** such as AdapterFusion or Prompt Tuning.
- Optimize **hyperparameters** for better generalization.

## Acknowledgments
- **Hugging Face** for Transformers and PEFT libraries.
- **Dataset Contributors** for the Hate Speech Twitter dataset.
- **LoRA Authors** for developing lightweight fine-tuning methodologies.
