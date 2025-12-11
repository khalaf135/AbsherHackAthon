# main.py

from __future__ import annotations

from typing import Optional

from generator import MaliciousGenerator, GeneratorConfig
from decrmenator import RLDiscriminator, RLDiscriminatorConfig


def build_models(
    csv_path: str,
    base_model_id: str,
    adapter_dir: Optional[str] = None,
) -> tuple[MaliciousGenerator, RLDiscriminator]:
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

    return generator, discriminator


def main() -> None:
    CSV_PATH = "Dataset_5971.csv"
    BASE_MODEL_ID = "dphn/Dolphin3.0-Llama3.1-8B"
    ADAPTER_DIR = None

    print("Loading models...")
    generator, discriminator = build_models(
        csv_path=CSV_PATH,
        base_model_id=BASE_MODEL_ID,
        adapter_dir=ADAPTER_DIR,
    )

    print("\nRL + LLM SMS System")
    print("-------------------")

    menu = """
[1] Classify a custom SMS
[2] Generate an SMS with the LLM and classify it
[3] Exit
"""

    while True:
        print(menu)
        choice = input("Your choice: ").strip()

        if choice == "1":
            msg = input("Enter SMS text: ").strip()
            if not msg:
                print("Empty message.")
                continue

            result = discriminator.classify(msg)
            print("\n--- Result ---")
            print("Message:", msg)
            print("Label:", result["label"])
            print(
                f"P(ham)={result['probs']['ham']:.4f}, "
                f"P(malicious)={result['probs']['malicious']:.4f}"
            )

        elif choice == "2":
            prompt = input(
                "Seed prompt for the generator "
                "(default: 'Write a suspicious SMS'): "
            ).strip()
            if not prompt:
                prompt = "Write a suspicious SMS that might be spam:"

            sms = generator.generate(prompt)
            result = discriminator.classify(sms)

            print("\n--- Generated SMS ---")
            print("Prompt:", prompt)
            print("SMS:", sms)
            print("Label:", result["label"])
            print(
                f"P(ham)={result['probs']['ham']:.4f}, "
                f"P(malicious)={result['probs']['malicious']:.4f}"
            )

        elif choice == "3":
            print("Bye.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()