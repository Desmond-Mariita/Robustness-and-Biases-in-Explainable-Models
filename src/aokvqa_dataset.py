import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from transformers import CLIPProcessor

class AOKVQADataset(Dataset):
    """
    PyTorch-compatible dataset adapter for the A-OKVQA benchmark.
    Loads images, questions, answer choices, correct label, and rationales.
    """
    def __init__(self, split="train", image_dir="/path/to/coco2017/train2017", processor_name="openai/clip-vit-base-patch32"):
        """
        Args:
            split (str): one of "train", "validation", "test"
            image_dir (str): path to COCO 2017 images
            processor_name (str): name of the CLIP processor
        """
        self.dataset = load_dataset("allenai/aokvqa", split=split)
        self.image_dir = image_dir
        self.processor = CLIPProcessor.from_pretrained(processor_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_path = os.path.join(self.image_dir, sample['image_id'] + ".jpg")
        image = Image.open(image_path).convert("RGB")
        processed = self.processor(images=image, return_tensors="pt")

        question = sample['question']
        choices = [c['text'] for c in sample['choices']]
        label = sample['correct_choice_idx']
        rationale = sample.get('rationale', None)

        return {
            "pixel_values": processed['pixel_values'].squeeze(0),
            "question": question,
            "choices": choices,
            "label": label,
            "rationale": rationale
        }

def aokvqa_collate_fn(batch):
    """Pads and batches AOKVQA samples."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    questions = [item["question"] for item in batch]
    choices = [item["choices"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    rationales = [item["rationale"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "questions": questions,
        "choices": choices,
        "labels": labels,
        "rationales": rationales
    }

