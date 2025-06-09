import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


class PJXAttention(nn.Module):
    """
    Implements soft attention between visual and textual features.
    Computes attention weights over image regions given the encoded question.
    """
    def __init__(self, image_dim, text_dim, hidden_dim):
        super(PJXAttention, self).__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attn_proj = nn.Linear(hidden_dim, 1)

    def forward(self, image_feats, question_feat):
        """
        Args:
            image_feats: (B, N, D_img) where N is # of image regions
            question_feat: (B, D_txt) encoded question vector
        Returns:
            attn_weights: (B, N) attention over image regions
        """
        B, N, _ = image_feats.size()

        image_proj = self.image_proj(image_feats)  # (B, N, H)
        question_proj = self.text_proj(question_feat).unsqueeze(1)  # (B, 1, H)

        fused = torch.tanh(image_proj + question_proj)  # (B, N, H)
        attn_logits = self.attn_proj(fused).squeeze(-1)  # (B, N)
        attn_weights = F.softmax(attn_logits, dim=1)
        return attn_weights


class ExplanationDecoder(nn.Module):
    """
    Transformer-based decoder to generate textual justifications.
    Uses Mistral-7B-Instruct for local generation.
    """
    def __init__(self, model_name='mistralai/Mistral-7B-Instruct-v0.2'):
        super(ExplanationDecoder, self).__init__()
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                          bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.decoder = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, context_vector, prompt_text=None, max_len=60):
        """
        Args:
            context_vector: (B, D) vector from attention-weighted image features
            prompt_text: list of strings to prime decoder
        Returns:
            generated_texts: list of decoded strings
        """
        B, D = context_vector.size()

        # Convert context vector into textual embedding prompt if needed
        if prompt_text is None:
            prompt_text = ["Explain the answer based on this visual context:"] * B

        # Basic example of conditioning: append a summary token from context vector
        # In production, use learned adapters or prefix-tuning
        inputs = self.tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True).to(context_vector.device)
        outputs = self.decoder.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_len,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        generated_texts = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        return generated_texts


class PJXModel(nn.Module):
    """
    PJ-X Model combining visual question answering with explanations.
    Uses CLIP vision encoder, a transformer for text encoding, and Mistral-7B for decoding.
    """
    def __init__(self, vision_encoder, text_encoder_model='bert-base-uncased', hidden_dim=512, num_answers=4):
        super(PJXModel, self).__init__()
        self.vision_encoder = vision_encoder  # e.g., CLIP vision tower

        self.text_encoder = AutoModel.from_pretrained(text_encoder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_model)

        self.attention = PJXAttention(
            image_dim=vision_encoder.output_dim,
            text_dim=self.text_encoder.config.hidden_size,
            hidden_dim=hidden_dim
        )

        self.answer_head = nn.Sequential(
            nn.Linear(vision_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_answers)
        )

        self.decoder = ExplanationDecoder(model_name='mistralai/Mistral-7B-Instruct-v0.2')

    def forward(self, image, question_texts, answer_choices=None):
        """
        Args:
            image: tensor of shape (B, 3, H, W)
            question_texts: list of questions as strings
        Returns:
            answer_logits: (B, num_answers)
            attn_weights: (B, N)
            explanations: list of generated explanations
        """
        # Step 1: Visual features
        image_feats = self.vision_encoder(image)  # (B, N, D_img)

        # Step 2: Encode question with transformer
        inputs = self.tokenizer(question_texts, return_tensors='pt', padding=True, truncation=True).to(image.device)
        text_outputs = self.text_encoder(**inputs)
        question_vec = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token (B, D_txt)

        # Step 3: Attention map over image regions
        attn_weights = self.attention(image_feats, question_vec)  # (B, N)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), image_feats).squeeze(1)  # (B, D_img)

        # Step 4: Answer prediction
        answer_logits = self.answer_head(context_vector)  # (B, num_answers)

        # Step 5: Generate justification
        explanations = self.decoder(context_vector)

        return answer_logits, attn_weights, explanations


