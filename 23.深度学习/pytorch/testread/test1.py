import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset

# 参数配置
MODEL_NAME = "deepseek-ai/deepseek-llm-1.3b"  # 根据实际模型调整
DATASET_NAME = "your_dataset"
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.texts = data["text"]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }


# 加载数据
dataset = load_dataset(DATASET_NAME)
train_dataset = TextDataset(dataset["train"], tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练循环
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# 保存模型
model.save_pretrained("trained_deepseek")
tokenizer.save_pretrained("trained_deepseek")