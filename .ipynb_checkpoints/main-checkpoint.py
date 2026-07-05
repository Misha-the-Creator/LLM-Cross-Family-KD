from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    print("Hello from knowledge-distillation!")

    student_name = "LiquidAI/LFM2.5-230M"
    student_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=student_name, device_map="auto", dtype="bfloat16"
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=student_name
    )

    prompt = "What is C. elegans?"

    input_ids = student_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )["input_ids"].to(student_model.device)


if __name__ == "__main__":
    main()
