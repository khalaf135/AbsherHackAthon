# train.py

from __future__ import annotations

import random
from typing import List, Optional

from generator import MaliciousGenerator, GeneratorConfig
from decrmenator import RLDiscriminator, RLDiscriminatorConfig


def adversarial_train(
    csv_path: str,
    base_model_id: str,
    adapter_dir: Optional[str] = None,
    num_steps: int = 10,
    prompts: Optional[List[str]] = None,
) -> None:
    disc_cfg = RLDiscriminatorConfig()
    discriminator = RLDiscriminator.from_csv(
        csv_path=csv_path,
        config=disc_cfg,
    )

    gen_cfg = GeneratorConfig(
        base_model_id=base_model_id,
        adapter_dir=adapter_dir,
    )
    generator = MaliciousGenerator(config=gen_cfg)

    if prompts is None:
        prompts = [
            "Write a short SMS that might look like a bank smishing attempt:",
            "Write a suspicious promotional SMS that could be spam:",
            "Write an SMS that tries to trick a user into clicking a link:",
        ]

    print("\nStarting adversarial loop...")
    for step in range(1, num_steps + 1):
        prompt = random.choice(prompts)
        sms = generator.generate(prompt)

        pred_label, probs, loss_value = discriminator.predict_and_self_update(
            sms,
            do_learn=True,
        )

        label_str = "ham" if pred_label == 0 else "malicious"

        print("\n" + "=" * 60)
        print(f"Step {step}/{num_steps}")
        print("Prompt:", prompt)
        print("SMS:", sms)
        print(
            f"Pred: {label_str} | "
            f"P(ham)={probs[0]:.4f}, P(malicious)={probs[1]:.4f}"
        )
        if loss_value is not None:
            print(f"Loss: {loss_value:.6f}")


if __name__ == "__main__":
    CSV_PATH = "Dataset_5971.csv"
    BASE_MODEL_ID = "dphn/Dolphin3.0-Llama3.1-8B"
    ADAPTER_DIR = None

    adversarial_train(
        csv_path=CSV_PATH,
        base_model_id=BASE_MODEL_ID,
        adapter_dir=ADAPTER_DIR,
        num_steps=5,
    )